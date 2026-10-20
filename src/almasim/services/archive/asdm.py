"""Find and check ALMA ASDM directories (MADIV-28 T4.8)."""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from ..extraction import is_extraction_tmp_dir

logger = logging.getLogger(__name__)


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
        raise RuntimeError(f"ASDM is incomplete, ASDM.xml does not parse ({exc}): {root}") from exc

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
        raise RuntimeError(f"ASDM is incomplete, Main.xml does not parse ({exc}): {root}") from exc

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
