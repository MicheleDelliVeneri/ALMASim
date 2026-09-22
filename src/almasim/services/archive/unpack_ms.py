"""Import ALMA ASDM directories into raw MeasurementSets.

This module intentionally performs only the raw ASDM-to-MS import via
``casatasks.importasdm``. It does not calibrate, split, image, or restore
pipeline products.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import socket
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from almasim.services.extraction import is_extraction_tmp_dir

logger = logging.getLogger(__name__)


LogFn = Callable[[str], None] | None


def _emit(logger_fn: LogFn, message: str) -> None:
    if logger_fn is not None:
        logger_fn(message)


def has_casa_runtime_data(path: str | os.PathLike[str]) -> bool:
    """Return true when ``path`` looks like populated CASA runtime data."""
    data_path = Path(path)
    return (data_path / "readme.txt").is_file() and (data_path / "geodetic").is_dir()


def find_existing_casa_data(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    casa_data_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Choose a CASA runtime data directory for standalone ``casatasks`` use."""
    if casa_data_root is not None:
        return Path(casa_data_root).expanduser().resolve()

    output_casa_data = Path(output_root).expanduser().resolve() / ".casa-data"
    if has_casa_runtime_data(output_casa_data):
        return output_casa_data

    input_path = Path(input_root).expanduser().resolve()
    for candidate in input_path.rglob(".casa-data"):
        if candidate.is_dir() and has_casa_runtime_data(candidate):
            return candidate

    return output_casa_data


def casa_log_path(output_root: str | os.PathLike[str], stage: str, uid: str | None) -> Path:
    """Return the CASA log file used for one ``stage`` (unpack/calibrate) of one UID."""
    safe_uid = (uid or "all").replace("/", "_").replace(":", "_")
    return Path(output_root).expanduser().resolve() / "logs" / f"casa-{stage}-{safe_uid}.log"


def _site_config_name() -> str:
    """Site config file name unique to this process, also across NFS clients."""
    host = socket.gethostname().split(".", 1)[0] or "localhost"
    return f"casasiteconfig-{host}-{os.getpid()}.py"


def _write_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that no reader can ever observe it partially written."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


_EXIT_CLEANUP: set[Path] = set()


def _remove_at_exit(path: Path) -> None:
    """Delete ``path`` when the interpreter exits (registered once per path)."""
    if path in _EXIT_CLEANUP:
        return
    _EXIT_CLEANUP.add(path)
    atexit.register(path.unlink, missing_ok=True)


def configure_casa_environment(
    output_root: str | os.PathLike[str],
    casa_data: str | os.PathLike[str],
    workspace_root: str | os.PathLike[str] | None = None,
    log_file: str | os.PathLike[str] | None = None,
    log_to_terminal: bool = False,
) -> Path:
    """Create and point CASA at a local site config and Matplotlib cache.

    ``log_file`` redirects the CASA logger away from ``casa-<timestamp>.log`` in
    the current directory; ``log_to_terminal`` additionally echoes every CASA log
    message to the console so long-running tasks show progress.

    The site config is **per process**: it is written to
    ``.casa-config/casasiteconfig-<host>-<pid>.py``, atomically (temp file +
    rename), exported through ``CASASITECONFIG`` unconditionally, and removed
    at interpreter exit. It used to be one shared ``casasiteconfig.py`` that
    every worker truncated and rewrote; with eight children importing casatools
    concurrently, one would read the file while another was rewriting it, see
    an empty config, fall back to ``~/.casa/data`` and die with ``measures data
    is not available`` before doing any work. The content is also per UID (the
    ``logfile`` line), so a shared file was wrong even without the race.
    """
    output_path = Path(output_root).expanduser().resolve()
    casa_data_path = Path(casa_data).expanduser().resolve()
    workspace_path = (
        Path(workspace_root).expanduser().resolve() if workspace_root is not None else output_path
    )

    workspace_path.mkdir(parents=True, exist_ok=True)

    mpl_config = workspace_path / ".matplotlib"
    casa_config_dir = workspace_path / ".casa-config"
    casa_site_config = casa_config_dir / _site_config_name()

    casa_data_path.mkdir(parents=True, exist_ok=True)
    mpl_config.mkdir(parents=True, exist_ok=True)
    casa_config_dir.mkdir(parents=True, exist_ok=True)

    config_lines = [
        f"measurespath = {str(casa_data_path)!r}",
        "data_auto_update = False",
        "measures_auto_update = False",
    ]
    if log_file is not None:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        config_lines.append(f"logfile = {str(log_path)!r}")
    config_lines.append(f"log2term = {bool(log_to_terminal)!r}")
    _write_atomically(casa_site_config, "\n".join(config_lines) + "\n")
    _remove_at_exit(casa_site_config)

    # Not setdefault: casatools reads this variable at import time and the
    # config we just wrote is the only one that describes *this* process.
    os.environ["CASASITECONFIG"] = str(casa_site_config)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return casa_data_path


def ensure_casa_runtime_data(
    casa_data: Path,
    skip_update: bool = False,
    logger_fn: LogFn = None,
) -> None:
    """Populate CASA runtime data when needed."""
    if has_casa_runtime_data(casa_data):
        logger.info("Using CASA runtime data: %s", casa_data)
        _emit(logger_fn, f"Using CASA runtime data: {casa_data}")
        return

    if skip_update:
        raise RuntimeError(
            "CASA runtime data is missing at {0}. Re-run without "
            "--skip-casa-data-update or pass --casa-data-root pointing to a "
            "populated data directory.".format(casa_data)
        )

    logger.info("Populating CASA runtime data: %s", casa_data)
    _emit(logger_fn, f"Populating CASA runtime data: {casa_data}")
    from casaconfig import update_all

    # casaconfig expects a CASA logsink object, not a Python logger — pass None
    # so it falls back to its own stdout/stderr output.
    update_all(path=str(casa_data), logger=None)


def find_asdm_directories(
    input_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
) -> list[Path]:
    """Find ASDM directories below ``input_root``."""
    input_path = Path(input_root).expanduser().resolve()
    if not input_path.is_dir():
        raise RuntimeError(f"Input root does not exist or is not a directory: {input_path}")

    # Walk rather than rglob, pruning descent into matched directories. An ASDM
    # holds thousands of files, so rglob("*.asdm.sdm") spends all its time
    # walking *inside* the very directories it has already matched: on a tree of
    # ~2600 ASDMs over NFS that is minutes per lookup, paid once per Slurm task.
    wanted = f"{asdm_uid}.asdm.sdm" if asdm_uid is not None else None
    asdm_dirs = []
    for dirpath, dirnames, _ in os.walk(input_path):
        # A download job extracts each tar into a hidden staging directory and
        # publishes it with one rename. Whatever is inside staging is a
        # half-written tree that looks like an ASDM and is not one yet.
        dirnames[:] = [name for name in dirnames if not is_extraction_tmp_dir(name)]
        matched = [name for name in dirnames if name.endswith(".asdm.sdm")]
        for name in matched:
            if wanted is None or name == wanted:
                asdm_dirs.append(Path(dirpath) / name)
        # Never descend into an ASDM; nothing below one is another ASDM.
        dirnames[:] = [name for name in dirnames if not name.endswith(".asdm.sdm")]

    if not asdm_dirs:
        if asdm_uid is None:
            raise RuntimeError(f"No *.asdm.sdm directories found under {input_path}")
        raise RuntimeError(f"No ASDM named {asdm_uid}.asdm.sdm found under {input_path}")

    return sorted(asdm_dirs)


def asdm_name(asdm_path: str | os.PathLike[str]) -> str:
    """Return the ASDM UID without the ``.asdm.sdm`` suffix."""
    return Path(asdm_path).name.replace(".asdm.sdm", "")


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _entity_id_to_filename(entity_id: str) -> str:
    """``uid://A002/X11b5555/X43a9`` -> ``uid___A002_X11b5555_X43a9``."""
    return entity_id.strip().replace("://", "___").replace("/", "_")


def verify_asdm_directory(asdm_path: str | os.PathLike[str]) -> None:
    """Raise ``RuntimeError`` unless ``asdm_path`` is a complete ASDM.

    Directory existence is not evidence of completeness. An ASDM tar is
    extracted over minutes, a killed extraction leaves whatever had landed, and
    ``importasdm`` reports the gap only as an opaque ``ASDMUtilsException: File
    not found`` after copying the working set. Extraction is atomic now, so this
    should never fire; it is the backstop for ASDMs that reached the input root
    by any other route.

    Checks, in the order the data would be needed:

    - ``ASDM.xml`` exists and parses;
    - every table it lists with ``NumberRows > 0`` has its ``<Name>.xml`` or
      ``<Name>.bin`` file;
    - every ``dataUID`` referenced from ``Main.xml`` has a non-empty file under
      ``ASDMBinary/`` — that is the visibility data, the bulk of the archive
      and the last thing a truncated extraction would have written.
    """
    root = Path(asdm_path)
    if not root.is_dir():
        raise RuntimeError(f"ASDM directory does not exist: {root}")

    index = root / "ASDM.xml"
    if not index.is_file():
        raise RuntimeError(f"ASDM is incomplete, ASDM.xml is missing: {root}")
    try:
        index_root = ET.parse(index).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"ASDM is incomplete, ASDM.xml does not parse ({exc}): {root}")

    missing_tables: list[str] = []
    for table in index_root.iter():
        if _xml_local_name(table.tag) != "Table":
            continue
        name = rows = None
        for child in table:
            local = _xml_local_name(child.tag)
            if local == "Name" and child.text:
                name = child.text.strip()
            elif local == "NumberRows" and child.text:
                rows = int(child.text.strip())
        if not name or not rows:
            continue
        if not ((root / f"{name}.xml").is_file() or (root / f"{name}.bin").is_file()):
            missing_tables.append(name)
    if missing_tables:
        shown = ", ".join(missing_tables[:5])
        more = f" (+{len(missing_tables) - 5} more)" if len(missing_tables) > 5 else ""
        raise RuntimeError(
            f"ASDM is incomplete, {len(missing_tables)} table file(s) missing: "
            f"{shown}{more}: {root}"
        )

    main = root / "Main.xml"
    if not main.is_file():
        # Only reachable if ASDM.xml lists Main with 0 rows; nothing to import.
        raise RuntimeError(f"ASDM is incomplete, Main.xml is missing: {root}")
    try:
        main_root = ET.parse(main).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"ASDM is incomplete, Main.xml does not parse ({exc}): {root}")

    binary_dir = root / "ASDMBinary"
    missing_bdfs: list[str] = []
    expected = 0
    for element in main_root.iter():
        if _xml_local_name(element.tag) != "dataUID":
            continue
        for ref in element.iter():
            entity_id = ref.get("entityId")
            if entity_id is None:
                continue
            expected += 1
            bdf = binary_dir / _entity_id_to_filename(entity_id)
            if not bdf.is_file() or bdf.stat().st_size == 0:
                missing_bdfs.append(bdf.name)
    if missing_bdfs:
        shown = ", ".join(missing_bdfs[:3])
        more = f" (+{len(missing_bdfs) - 3} more)" if len(missing_bdfs) > 3 else ""
        raise RuntimeError(
            f"ASDM is incomplete, {len(missing_bdfs)}/{expected} binary data file(s) "
            f"missing or empty under ASDMBinary: {shown}{more}: {root}"
        )


def raw_ms_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the raw MeasurementSet path for ``uid``."""
    return Path(output_root).expanduser().resolve() / "working" / f"{uid}.ms"


def raw_ms_marker_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the completion marker written next to a finished raw MS."""
    return raw_ms_path(output_root, uid).with_name(f"{uid}.ms.done")


def raw_ms_failure_marker_path(output_root: str | os.PathLike[str], uid: str) -> Path:
    """Return the failure marker written when ``uid`` could not be imported."""
    return raw_ms_path(output_root, uid).with_name(f"{uid}.ms.failed")


def measurement_set_row_count(ms_path: str | os.PathLike[str]) -> int:
    """Return the number of rows in the MAIN table of ``ms_path``.

    A killed ``importasdm`` leaves a directory that looks complete: the schema
    and tens of GB of storage-manager files are on disk, but the row count was
    never committed, so the table opens cleanly and reports zero rows. Checking
    the directory exists is therefore not enough to call an import finished.
    """
    from casatools import table

    tb = table()
    try:
        if not tb.open(str(ms_path)):
            raise RuntimeError(f"Could not open MeasurementSet: {ms_path}")
        return int(tb.nrows())
    finally:
        try:
            tb.close()
        except Exception:  # pragma: no cover - close failures are not actionable
            pass


def verify_measurement_set(ms_path: str | os.PathLike[str]) -> int:
    """Raise unless ``ms_path`` is a MeasurementSet with at least one row."""
    path = Path(ms_path)
    if not path.is_dir():
        raise RuntimeError(f"Expected MeasurementSet was not created: {path}")
    rows = measurement_set_row_count(path)
    if rows <= 0:
        raise RuntimeError(
            f"MeasurementSet has no rows, the import did not finish: {path}. "
            "Re-import it from its ASDM; the directory on disk is not usable."
        )
    return rows


def is_unpack_complete(output_root: str | os.PathLike[str], uid: str) -> bool:
    """Return True when ``uid`` has both a raw MS directory and its marker."""
    return raw_ms_path(output_root, uid).is_dir() and raw_ms_marker_path(output_root, uid).is_file()


def write_raw_ms_marker(output_root: str | os.PathLike[str], uid: str, rows: int) -> Path:
    """Record that ``uid`` imported successfully, with the row count as proof."""
    marker = raw_ms_marker_path(output_root, uid)
    marker.write_text(
        json.dumps(
            {
                "uid": uid,
                "output": str(raw_ms_path(output_root, uid)),
                "rows": rows,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    raw_ms_failure_marker_path(output_root, uid).unlink(missing_ok=True)
    return marker


def write_raw_ms_failure_marker(
    output_root: str | os.PathLike[str],
    uid: str,
    error: str,
    *,
    log_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Record why ``uid`` could not be imported."""
    marker = raw_ms_failure_marker_path(output_root, uid)
    payload = {
        "uid": uid,
        "stage": "unpack",
        "error": error,
        "failed_at": datetime.now(timezone.utc).isoformat(),
    }
    if log_path is not None:
        payload["log"] = str(log_path)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return marker


def create_measurement_set(
    importasdm: Callable[..., object],
    raw_asdm: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> Path:
    """Create one raw MeasurementSet from one ASDM directory.

    An import is only considered finished once the resulting MS has at least one
    row and a ``<uid>.ms.done`` marker. Directory existence is deliberately not
    enough: a killed ``importasdm`` leaves a complete-looking directory holding
    tens of GB whose row count was never committed, and treating that as done is
    what let truncated MeasurementSets flow into calibration, where they abort
    CASA with ``minMax - Array has no elements``.

    An existing MS without a marker is verified rather than trusted: if it has
    rows it is grandfathered in and marked, otherwise it is re-imported.
    """
    raw_asdm_path = Path(raw_asdm).expanduser().resolve()
    asdm_uid = asdm_name(raw_asdm_path)
    working_dir = Path(output_root).expanduser().resolve() / "working"

    if not raw_asdm_path.is_dir():
        raise RuntimeError(f"Cannot find raw ASDM directory: {raw_asdm_path}")

    working_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Working directory: %s", working_dir)

    output_root_path = Path(output_root).expanduser().resolve()
    output_ms = working_dir / f"{asdm_uid}.ms"

    if not overwrite and output_ms.exists():
        if raw_ms_marker_path(output_root_path, asdm_uid).is_file():
            logger.info("MeasurementSet already exists, skipping import: %s", output_ms)
            _emit(
                logger_fn,
                f"Raw MeasurementSet already exists, skipping import: {output_ms}",
            )
            return output_ms
        # No marker: either an import from before markers existed, or one that
        # was killed mid-write. Only the row count tells the two apart, and
        # trusting the directory is what let truncated MSs reach calibration.
        try:
            rows = verify_measurement_set(output_ms)
        except RuntimeError as exc:
            logger.warning("Replacing unusable MeasurementSet %s: %s", output_ms, exc)
            _emit(
                logger_fn,
                f"Existing MeasurementSet is unusable, re-importing: {output_ms} ({exc})",
            )
        else:
            write_raw_ms_marker(output_root_path, asdm_uid, rows)
            logger.info("Verified existing MeasurementSet (%d rows): %s", rows, output_ms)
            _emit(
                logger_fn,
                f"Verified existing raw MeasurementSet ({rows:,} rows), skipping import: "
                f"{output_ms}",
            )
            return output_ms

    logger.info("Creating MeasurementSet from %s", raw_asdm_path)
    _emit(logger_fn, f"Creating raw MeasurementSet from {raw_asdm_path}")
    # Reaching here means we have decided to import: either the caller asked for
    # it, or whatever is on disk is unusable. importasdm must be allowed to
    # replace it, so ``overwrite`` is unconditionally True below.
    raw_ms_marker_path(output_root_path, asdm_uid).unlink(missing_ok=True)
    raw_ms_failure_marker_path(output_root_path, asdm_uid).unlink(missing_ok=True)
    current_dir = Path.cwd()
    try:
        # Refuse a half-extracted ASDM up front, as a recorded failure. Left to
        # importasdm it becomes an opaque "File not found" after minutes of work.
        verify_asdm_directory(raw_asdm_path)
        try:
            os.chdir(working_dir)
            importasdm(
                asdm=str(raw_asdm_path),
                vis=str(output_ms),
                overwrite=True,
            )
        finally:
            os.chdir(current_dir)
        rows = verify_measurement_set(output_ms)
    except Exception as exc:
        write_raw_ms_failure_marker(output_root_path, asdm_uid, f"{type(exc).__name__}: {exc}")
        raise

    write_raw_ms_marker(output_root_path, asdm_uid, rows)
    logger.info("Created MeasurementSet (%d rows): %s", rows, output_ms)
    _emit(logger_fn, f"Created raw MeasurementSet ({rows:,} rows): {output_ms}")
    return output_ms


def create_measurement_sets(
    input_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    asdm_uid: str | None = None,
    casa_data_root: str | os.PathLike[str] | None = None,
    skip_casa_data_update: bool = False,
    overwrite: bool = False,
    logger_fn: LogFn = None,
) -> list[Path]:
    """Create raw MeasurementSets for all matching ASDMs below ``input_root``.

    Each UID that finishes carries a ``<uid>.ms.done`` marker recording its row
    count; each that fails carries a ``<uid>.ms.failed`` marker recording the
    error. See :func:`create_measurement_set` for why the row count matters.
    """
    asdm_dirs = find_asdm_directories(input_root, asdm_uid)

    # Decide what is already finished *before* importing CASA. A UID whose MS
    # carries a ``.ms.done`` marker needs no casatools at all, and importing
    # them anyway is what exposed finished UIDs to CASA start-up failures
    # (which the subprocess wrapper then recorded as ``.ms.failed``).
    finished: dict[Path, Path] = {}
    if not overwrite:
        for raw_asdm in asdm_dirs:
            uid = asdm_name(raw_asdm)
            if is_unpack_complete(output_root, uid):
                ms = raw_ms_path(output_root, uid)
                finished[raw_asdm] = ms
                logger.info("MeasurementSet already exists, skipping import: %s", ms)
                _emit(logger_fn, f"Raw MeasurementSet already exists, skipping import: {ms}")
    if asdm_dirs and len(finished) == len(asdm_dirs):
        return [finished[raw_asdm] for raw_asdm in asdm_dirs]

    casa_data = find_existing_casa_data(input_root, output_root, casa_data_root)
    configure_casa_environment(
        output_root,
        casa_data,
        log_file=casa_log_path(output_root, "unpack", asdm_uid),
        log_to_terminal=True,
    )
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update, logger_fn=logger_fn)

    from casatasks import importasdm

    logger.info(
        "Found %d ASDM director%s",
        len(asdm_dirs),
        "y" if len(asdm_dirs) == 1 else "ies",
    )
    _emit(
        logger_fn,
        f"Found {len(asdm_dirs)} ASDM director{'y' if len(asdm_dirs) == 1 else 'ies'}",
    )
    return [
        finished[raw_asdm]
        if raw_asdm in finished
        else create_measurement_set(
            importasdm,
            raw_asdm,
            output_root,
            overwrite=overwrite,
            logger_fn=logger_fn,
        )
        for raw_asdm in asdm_dirs
    ]
