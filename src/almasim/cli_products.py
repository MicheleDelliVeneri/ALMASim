"""Product resolution and download commands for the ALMASim CLI."""

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import sleep, time
from typing import Any, List, Optional

import typer
from tqdm.auto import tqdm

from .cli_shared import dedupe_keep_order, default_output_path, split_csv_values

MAX_PARALLEL_PER_MIRROR = 10
MAX_PARALLEL_TOTAL = 30
PRODUCT_TYPES = {
    "all",
    "raw",
    "calibration",
    "scripts",
    "weblog",
    "qa_reports",
    "auxiliary",
    "cubes",
    "continuum",
    "fits",
    "other",
}

products_app = typer.Typer(
    help="Data product resolution and download commands.",
    no_args_is_help=True,
)


@lru_cache(maxsize=1)
def _download_contract() -> dict[str, Any]:
    from .services.download import (
        MAX_PARALLEL_PER_MIRROR as _MAX_PARALLEL_PER_MIRROR,
    )
    from .services.download import (
        MAX_PARALLEL_TOTAL as _MAX_PARALLEL_TOTAL,
    )
    from .services.download import (
        PRODUCT_TYPES as _PRODUCT_TYPES,
    )
    from .services.download import (
        download_products as _download_products,
    )
    from .services.download import (
        filter_products as _filter_products,
    )
    from .services.download import (
        format_bytes as _format_bytes,
    )
    from .services.download import (
        load_products_csv as _load_products_csv,
    )
    from .services.download import (
        resolve_products as _resolve_products,
    )
    from .services.download import (
        save_products_csv as _save_products_csv,
    )

    return {
        "MAX_PARALLEL_PER_MIRROR": _MAX_PARALLEL_PER_MIRROR,
        "MAX_PARALLEL_TOTAL": _MAX_PARALLEL_TOTAL,
        "PRODUCT_TYPES": _PRODUCT_TYPES,
        "download_products": _download_products,
        "filter_products": _filter_products,
        "format_bytes": _format_bytes,
        "load_products_csv": _load_products_csv,
        "resolve_products": _resolve_products,
        "save_products_csv": _save_products_csv,
    }


def create_backend(*args, **kwargs):
    from .services.compute import create_backend as _create_backend

    return _create_backend(*args, **kwargs)


def download_products(*args, **kwargs):
    return _download_contract()["download_products"](*args, **kwargs)


def filter_products(*args, **kwargs):
    return _download_contract()["filter_products"](*args, **kwargs)


def format_bytes(*args, **kwargs):
    return _download_contract()["format_bytes"](*args, **kwargs)


def load_products_csv(*args, **kwargs):
    return _download_contract()["load_products_csv"](*args, **kwargs)


def resolve_products(*args, **kwargs):
    return _download_contract()["resolve_products"](*args, **kwargs)


def save_products_csv(*args, **kwargs):
    return _download_contract()["save_products_csv"](*args, **kwargs)


def _future_status(future: Any) -> str:
    status = getattr(future, "status", None)
    if callable(status):
        try:
            status = status()
        except TypeError:
            status = None
    return str(status).lower() if status is not None else ""


def _future_done(future: Any) -> bool:
    done_attr = getattr(future, "done", None)
    if callable(done_attr):
        try:
            return bool(done_attr())
        except TypeError:
            return False
    if isinstance(done_attr, bool):
        return done_attr

    status = _future_status(future)
    return status in {"finished", "done", "error", "failed", "cancelled"}


_NO_PROGRESS_TIMEOUT_S = 3600
_PROGRESS_HEARTBEAT_S = 60.0
_FAILED_FUTURE_STATES = {"error", "failed", "cancelled", "lost"}


@dataclass(frozen=True)
class StageFailure:
    """One UID that failed during an unpack/calibrate stage."""

    uid: str
    error: str
    log_path: Optional[str] = None


def _safe_uid(uid: str) -> str:
    return uid.replace("/", "_").replace(":", "_")


def _stage_log_path(output_root: Any, uid: str, stage: str) -> Path:
    """Per-UID log file written by the worker subprocess wrappers.

    The file lives under ``<output_root>/logs`` so that, on Slurm, the submit
    node can follow it while the job runs on a compute node.
    """
    return Path(str(output_root)).expanduser().resolve() / "logs" / f"{_safe_uid(uid)}.{stage}.log"


def _tail_log_line(
    path: Path, max_bytes: int = 4096, modified_since: Optional[float] = None
) -> Optional[str]:
    """Return the last non-empty line of ``path`` without reading the whole file.

    Files last modified before ``modified_since`` (a ``time.time()`` stamp) are
    ignored so that a log left behind by an earlier run is not mistaken for a
    job that is running now.
    """
    try:
        if modified_since is not None and path.stat().st_mtime < modified_since:
            return None
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            chunk = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.strip() for line in chunk.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _error_headline(error: str, limit: int = 240) -> str:
    """First informative line of an error message, trimmed for console output."""
    for line in str(error).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"
    return str(error)[:limit]


def _report_stage_failures(stage_label: str, failures: list[StageFailure]) -> None:
    if not failures:
        return
    typer.echo(f"{stage_label}: {len(failures)} UID(s) failed and were skipped:", err=True)
    for failure in failures:
        typer.echo(f"  {failure.uid}: {_error_headline(failure.error)}", err=True)
        if failure.log_path:
            typer.echo(f"    log: {failure.log_path}", err=True)


def _exit_if_failures(stage_label: str, failures: list[StageFailure]) -> None:
    """Print a failure summary and exit non-zero once every UID has been attempted."""
    if not failures:
        return
    _report_stage_failures(stage_label, failures)
    raise typer.Exit(code=1)


def _compute_jobs_with_progress(
    *,
    backend: Any,
    jobs: list[Any],
    job_uids: list[str],
    stage_label: str,
    no_progress_timeout: float = _NO_PROGRESS_TIMEOUT_S,
    continue_on_error: bool = False,
    log_paths: Optional[list[Path]] = None,
    failures: Optional[list[StageFailure]] = None,
    heartbeat_interval: float = _PROGRESS_HEARTBEAT_S,
) -> list[Any]:
    """Run a stage and show per-UID progress for asynchronous backends.

    With ``continue_on_error`` the result list contains ``None`` for every job
    that failed (details are appended to ``failures``) instead of raising on the
    first failure. When ``log_paths`` is given, the last line of every running
    UID's log is printed every ``heartbeat_interval`` seconds so long stages are
    not silent.
    """
    if not jobs:
        return []

    futures = backend.compute(jobs, sync=False)
    if not isinstance(futures, list):
        futures = [futures]

    def _uid(index: int) -> str:
        return job_uids[index] if index < len(job_uids) else f"job-{index + 1}"

    def _log_path(index: int) -> Optional[Path]:
        if log_paths is not None and index < len(log_paths):
            return log_paths[index]
        return None

    def _record_failure(index: int, error: str) -> None:
        if failures is not None:
            log_path = _log_path(index)
            failures.append(
                StageFailure(_uid(index), error, str(log_path) if log_path is not None else None)
            )

    completed: set[int] = set()
    failed: set[int] = set()
    stage_started_at = time()
    last_progress_time = stage_started_at
    last_heartbeat_time = stage_started_at
    with tqdm(total=len(futures), desc=stage_label, unit="uid", leave=True) as progress_bar:
        progress_bar.set_postfix_str(f"completed 0/{len(futures)}")
        while len(completed) < len(futures):
            prev_completed = len(completed)
            for index, future in enumerate(futures):
                if index in completed:
                    continue

                state = _future_status(future)
                if state in _FAILED_FUTURE_STATES:
                    failed.add(index)

                if not _future_done(future):
                    continue

                completed.add(index)
                progress_bar.update(1)
                uid = _uid(index)
                status_text = state if state else "finished"
                progress_bar.write(f"{stage_label} completed for {uid} [{status_text}]")
                if index in failed:
                    error_text = f"job ended in state {status_text!r}"
                    try:
                        exc = future.exception()
                        if exc is not None:
                            error_text = str(exc)
                    except Exception:
                        pass
                    _record_failure(index, error_text)
                    progress_bar.write(f"  {uid} error: {_error_headline(error_text)}")
                    log_path = _log_path(index)
                    if log_path is not None:
                        progress_bar.write(f"  {uid} log: {log_path}")

            if len(completed) > prev_completed:
                last_progress_time = time()

            progress_bar.set_postfix_str(
                f"completed {len(completed)}/{len(futures)} failed {len(failed)}"
            )
            if len(completed) < len(futures):
                now = time()
                if now - last_progress_time > no_progress_timeout:
                    pending = len(futures) - len(completed)
                    raise RuntimeError(
                        f"{stage_label}: {pending} future(s) made no progress for "
                        f"{no_progress_timeout:.0f}s — workers may have died. "
                        "Re-run with --skip-existing to resume."
                    )
                if log_paths is not None and now - last_heartbeat_time >= heartbeat_interval:
                    last_heartbeat_time = now
                    running = 0
                    for index in range(len(futures)):
                        if index in completed:
                            continue
                        log_path = _log_path(index)
                        last_line = (
                            _tail_log_line(log_path, modified_since=stage_started_at - 1.0)
                            if log_path is not None
                            else None
                        )
                        if last_line is None:
                            continue
                        running += 1
                        progress_bar.write(f"  [{_uid(index)}] {last_line}")
                    queued = len(futures) - len(completed) - running
                    progress_bar.write(
                        f"{stage_label}: {running} running, {queued} waiting for a worker, "
                        f"{len(completed)} done"
                    )
                sleep(0.5)

    if not continue_on_error:
        return backend.gather(futures)

    results: list[Any] = []
    for index, future in enumerate(futures):
        if index in failed:
            results.append(None)
            continue
        try:
            results.append(backend.gather([future])[0])
        except Exception as exc:
            failed.add(index)
            _record_failure(index, str(exc))
            results.append(None)
    return results


def _run_uid_stage(
    *,
    backend: Any,
    postprocess_backend: str,
    task_fn: Any,
    job_kwargs: list[dict[str, Any]],
    job_uids: list[str],
    stage_label: str,
    log_paths: list[Path],
    continue_on_error: bool,
    failures: list[StageFailure],
) -> list[list[str]]:
    """Run one per-UID task for every UID, returning one output list per UID.

    Failed UIDs yield an empty list and are recorded in ``failures`` when
    ``continue_on_error`` is set; otherwise the first failure propagates.
    """
    if not job_uids:
        return []

    if postprocess_backend == "slurm":
        task = backend.delayed(task_fn)
        jobs = [task(**kwargs) for kwargs in job_kwargs]
        results = _compute_jobs_with_progress(
            backend=backend,
            jobs=jobs,
            job_uids=job_uids,
            stage_label=stage_label,
            continue_on_error=continue_on_error,
            log_paths=log_paths,
            failures=failures,
        )
        return [list(result) if result else [] for result in results]

    outputs: list[list[str]] = []
    total = len(job_uids)
    for index, (uid, kwargs) in enumerate(zip(job_uids, job_kwargs), start=1):
        log_path = log_paths[index - 1] if index - 1 < len(log_paths) else None
        typer.echo(
            f"{stage_label} [{index}/{total}] starting {uid}"
            + (f" (log: {log_path})" if log_path else "")
        )
        try:
            result = task_fn(**kwargs)
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append(StageFailure(uid, str(exc), str(log_path) if log_path else None))
            typer.echo(
                f"{stage_label} [{index}/{total}] FAILED {uid}: {_error_headline(str(exc))}",
                err=True,
            )
            outputs.append([])
            continue
        typer.echo(f"{stage_label} [{index}/{total}] finished {uid}")
        outputs.append(list(result))
    return outputs


def _download_products_with_progress(
    products: list[Any],
    destination: Path,
    **kwargs: Any,
):
    total_known_bytes = sum(max(int(product.content_length), 0) for product in products)
    total_files = len(products)
    progress_total = total_known_bytes if total_known_bytes > 0 else max(total_files, 1)
    progress_unit = "B" if total_known_bytes > 0 else "file"
    progress_kwargs = {"unit_scale": True, "unit_divisor": 1000} if total_known_bytes > 0 else {}
    progress_lock = Lock()
    previous_bytes: dict[str, int] = {}
    previous_states: dict[str, str] = {}
    completed_files = 0

    with tqdm(
        total=progress_total,
        desc="Downloading products",
        unit=progress_unit,
        leave=True,
        **progress_kwargs,
    ) as progress_bar:
        progress_bar.set_postfix_str(f"files 0/{total_files}")

        def update_callback(file_status: Any) -> None:
            nonlocal completed_files

            key = f"{file_status.access_url}|{file_status.filename}"
            with progress_lock:
                current_bytes = max(int(file_status.bytes_downloaded), 0)
                previous = previous_bytes.get(key, 0)
                if total_known_bytes > 0 and current_bytes > previous:
                    progress_bar.update(current_bytes - previous)
                previous_bytes[key] = max(previous, current_bytes)

                status = str(file_status.status)
                previous_status = previous_states.get(key)
                if status in {"completed", "failed", "cancelled"} and previous_status not in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    completed_files += 1
                    if total_known_bytes <= 0:
                        progress_bar.update(1)
                previous_states[key] = status
                progress_bar.set_postfix_str(f"files {completed_files}/{total_files}")

        return download_products(
            products,
            destination,
            update_callback=update_callback,
            **kwargs,
        )


def _read_member_uids_from_metadata(
    metadata_csv: Path,
    member_limit: Optional[int],
) -> list[str]:
    import pandas as pd

    metadata = pd.read_csv(metadata_csv.expanduser().resolve())
    if "member_ous_uid" not in metadata.columns:
        typer.echo(
            f"Metadata CSV does not contain member_ous_uid: {metadata_csv}",
            err=True,
        )
        raise typer.Exit(code=2)
    series = metadata["member_ous_uid"].dropna().astype(str)
    if member_limit is not None:
        series = series.head(member_limit)
    return dedupe_keep_order(series.tolist())


def _parse_member_uid_options(member_ous_uid: Optional[List[str]]) -> list[str]:
    parsed = split_csv_values(member_ous_uid)
    if not parsed:
        return []
    return dedupe_keep_order(parsed)


def _parse_asdm_uid_options(asdm_uid: Optional[List[str]]) -> list[str]:
    parsed = split_csv_values(asdm_uid)
    if not parsed:
        return []
    return dedupe_keep_order(parsed)


def _resolve_products_from_inputs(
    *,
    products_csv: Optional[Path],
    metadata_csv: Optional[Path],
    member_ous_uid: Optional[List[str]],
    member_limit: Optional[int],
    save_products_csv_path: Optional[Path],
) -> list[Any]:
    if products_csv is not None:
        loaded = load_products_csv(products_csv)
        typer.echo(f"Loaded products CSV: {products_csv.expanduser().resolve()}")
        return loaded

    member_uids = _parse_member_uid_options(member_ous_uid)
    if metadata_csv is not None:
        member_uids.extend(_read_member_uids_from_metadata(metadata_csv, member_limit))
    member_uids = dedupe_keep_order([uid for uid in member_uids if uid])
    if not member_uids:
        typer.echo(
            "Provide --products-csv, --metadata-csv, or at least one --member-ous-uid.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(f"Resolving DataLink products for {len(member_uids)} member OUS UID(s)...")
    typer.echo(
        "Using ALMA DataLink services: "
        "ESO (almascience.eso.org), NRAO (almascience.nrao.edu), "
        "NAOJ (almascience.nao.ac.jp)"
    )

    resolved = []
    with typer.progressbar(member_uids, label="Resolving member OUS UIDs") as progress:
        for uid in progress:
            resolved.extend(resolve_products([uid]))

    typer.echo(f"Resolved DataLink rows: {len(resolved)}")
    if not resolved:
        typer.echo("No products were resolved for the requested member_ous_uid values.", err=True)
        raise typer.Exit(code=1)

    if save_products_csv_path is not None:
        saved = save_products_csv(resolved, save_products_csv_path)
        typer.echo(f"Saved resolved products CSV: {saved}")
    return resolved


def _extract_asdm_uids_from_download_root(download_root: Path) -> list[str]:
    from .services.archive import find_asdm_directories

    asdm_dirs = find_asdm_directories(download_root)
    return [path.name.removesuffix(".asdm.sdm") for path in asdm_dirs]


def _extract_uids_from_raw_ms_root(raw_ms_root: Path) -> list[str]:
    from .services.archive.calibrate_ms import find_raw_ms_directories

    return [path.name.removesuffix(".ms") for path in find_raw_ms_directories(raw_ms_root)]


def _unpack_single_uid(
    *,
    input_root: str,
    raw_output_root: str,
    asdm_uid: str,
    casa_data_root: Optional[str],
    skip_casa_data_update: bool,
    overwrite: bool,
) -> list[str]:
    """Run one UID's ASDM import in a fresh subprocess.

    CASA's importasdm maintains process-level global state that is not reset
    between calls in the same Python process. Running as a subprocess guarantees
    a clean CASA environment for every UID, matching the approach used by
    _calibrate_single_uid.
    """
    import os
    import subprocess
    import sys
    from collections import deque
    from pathlib import Path as _Path

    # Point workers at the pre-downloaded CASA runtime data so they never try
    # to download it themselves (compute nodes typically have no internet).
    effective_casa_data = casa_data_root or str(_Path(raw_output_root) / ".casa-data")

    cmd = [
        sys.executable,
        "-m",
        "almasim.cli",
        "products",
        "unpack",
        "--input-root",
        input_root,
        "--output-root",
        raw_output_root,
        "--asdm-uid",
        asdm_uid,
        "--postprocess-backend",
        "sync",
        "--casa-data-root",
        effective_casa_data,
        "--skip-casa-data-update",
    ]
    if overwrite:
        cmd.append("--overwrite-outputs")

    project_root = _Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}:{existing_pythonpath}" if existing_pythonpath else str(src_root)
    )

    # Prioritize system libraries to avoid GLIBC version conflicts with spack binaries.
    ld_library_path = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_library_path = f"{ld_library_path}:{existing_ld}"
    env["LD_LIBRARY_PATH"] = ld_library_path
    # Make the child's Python output line-buffered through the pipe so progress is live.
    env["PYTHONUNBUFFERED"] = "1"

    # Mirror everything the child prints into a per-UID log on the (shared) output
    # filesystem so the run can be followed with ``tail -f`` even on Slurm.
    log_path = _stage_log_path(raw_output_root, asdm_uid, "unpack")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        cwd=str(project_root),
        env=env,
    )

    # Stream child output as it arrives so long CASA logs never accumulate in memory.
    tail_lines: deque[str] = deque(maxlen=200)
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("# " + " ".join(cmd) + "\n")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            tail_lines.append(line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        tail_text = "\n".join(tail_lines)
        raise RuntimeError(
            f"Unpack failed for {asdm_uid}.\n"
            f"Return code: {return_code}\n"
            f"Full log: {log_path}\n"
            f"Last {len(tail_lines)} log lines:\n{tail_text}"
        )

    # Discover output paths produced by the subprocess.
    working_dir = _Path(raw_output_root) / "working"
    expected = working_dir / f"{asdm_uid}.ms"
    if expected.is_dir():
        return [str(expected)]
    # Fallback: scan for any MS produced for this UID.
    return [str(p) for p in working_dir.glob(f"{asdm_uid}*.ms") if p.is_dir()]


def _calibrate_single_uid(
    *,
    input_root: str,
    raw_ms_root: str,
    calibrated_output_root: str,
    asdm_uid: str,
    casa_data_root: Optional[str],
    skip_casa_data_update: bool,
    overwrite: bool,
    clean_intermediate: bool,
    keep_working_copies: bool = False,
) -> list[str]:
    """Run one UID's calibration in a fresh subprocess.

    CASA's calibrater tool maintains process-level global state that is not
    reset between calls in the same Python process. Running as a subprocess
    guarantees a clean CASA environment for every UID.
    """
    import os
    import subprocess
    import sys
    from collections import deque
    from pathlib import Path as _Path

    # Point workers at the pre-downloaded CASA runtime data so they never try
    # to download it themselves (compute nodes typically have no internet).
    effective_casa_data = casa_data_root or str(_Path(calibrated_output_root) / ".casa-data")

    cmd = [
        sys.executable,
        "-m",
        "almasim.cli",
        "products",
        "calibrate",
        "--input-root",
        input_root,
        "--raw-ms-root",
        raw_ms_root,
        "--output-root",
        calibrated_output_root,
        "--asdm-uid",
        asdm_uid,
        "--postprocess-backend",
        "sync",
        "--casa-data-root",
        effective_casa_data,
        "--skip-casa-data-update",
    ]
    if overwrite:
        cmd.append("--overwrite-outputs")
    if keep_working_copies:
        cmd.append("--keep-working-copies")

    project_root = _Path(__file__).resolve().parents[2]
    src_root = project_root / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}:{existing_pythonpath}" if existing_pythonpath else str(src_root)
    )

    # Prioritize system libraries to avoid GLIBC version conflicts with spack binaries.
    ld_library_path = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if existing_ld:
        ld_library_path = f"{ld_library_path}:{existing_ld}"
    env["LD_LIBRARY_PATH"] = ld_library_path
    # Make the child's Python output line-buffered through the pipe so progress is live.
    env["PYTHONUNBUFFERED"] = "1"

    # Mirror everything the child prints into a per-UID log on the (shared) output
    # filesystem so the run can be followed with ``tail -f`` even on Slurm.
    log_path = _stage_log_path(calibrated_output_root, asdm_uid, "calibrate")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        cwd=str(project_root),
        env=env,
    )

    # Stream child output as it arrives so long CASA logs never accumulate in memory.
    tail_lines: deque[str] = deque(maxlen=200)
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("# " + " ".join(cmd) + "\n")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
            tail_lines.append(line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        tail_text = "\n".join(tail_lines)
        raise RuntimeError(
            f"Calibration failed for {asdm_uid}.\n"
            f"Return code: {return_code}\n"
            f"Full log: {log_path}\n"
            f"Last {len(tail_lines)} log lines:\n{tail_text}"
        )

    # Discover output paths produced by the subprocess.
    output_path = _Path(calibrated_output_root)
    expected = output_path / f"{asdm_uid}.ms.split.cal"
    if expected.is_dir():
        return [str(expected)]
    # Fallback: scan for any split.cal produced for this UID.
    return [str(p) for p in output_path.glob(f"{asdm_uid}*.split.cal") if p.is_dir()]


def _preflight_casa_data(
    output_root: Path,
    casa_data_root: Optional[Path],
    skip_casa_data_update: bool,
) -> Path:
    """Ensure CASA runtime data is populated on the master node before workers start.

    Slurm compute nodes typically have no internet access, so the data must be
    downloaded once from the submit node and written to a shared filesystem path
    that all workers can read. Returns the directory the workers should use.
    """
    from .services.archive.unpack_ms import (
        ensure_casa_runtime_data,
        find_existing_casa_data,
    )

    casa_data = find_existing_casa_data(output_root, output_root, casa_data_root)
    typer.echo(f"Preflight: ensuring CASA runtime data at {casa_data} …")
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update)
    return Path(casa_data)


def _partition_completed_calibrations(
    output_root: Path, uids: list[str], overwrite: bool
) -> tuple[list[str], list[str]]:
    """Split ``uids`` into those still to calibrate and the outputs of finished ones.

    A UID counts as finished when ``<uid>.ms.split.cal`` exists together with
    its ``.done`` marker; with ``overwrite`` everything is redone.
    """
    from .services.archive import calibrated_output_path, is_calibration_complete

    if overwrite:
        return list(uids), []
    todo: list[str] = []
    done_outputs: list[str] = []
    for uid in uids:
        if is_calibration_complete(output_root, uid):
            done_outputs.append(str(calibrated_output_path(output_root, uid)))
        else:
            todo.append(uid)
    if done_outputs:
        typer.echo(
            f"Skipping {len(done_outputs)} already calibrated UID(s) with completion markers "
            "(pass --overwrite-outputs to redo them)."
        )
    return todo, done_outputs


def _run_unpack_jobs(
    *,
    input_root: Path,
    output_root: Path,
    asdm_uids: list[str],
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    casa_data_root: Optional[Path],
    skip_casa_data_update: bool,
    overwrite_outputs: bool,
    continue_on_error: bool = True,
    failures: Optional[list[StageFailure]] = None,
) -> list[str]:
    """Import ASDMs into raw MeasurementSets, one isolated subprocess per UID.

    A single UID (which is also how each worker subprocess re-enters this
    function) runs in-process. Several UIDs always run one subprocess each so a
    CASA crash on one execution block cannot take the rest of the batch down;
    failed UIDs are appended to ``failures`` when ``continue_on_error`` is set.
    """
    from .services.archive import create_measurement_sets

    if failures is None:
        failures = []

    effective_uids = asdm_uids or dedupe_keep_order(
        _extract_asdm_uids_from_download_root(input_root)
    )
    if not effective_uids:
        typer.echo("No ASDM directories found to unpack.", err=True)
        raise typer.Exit(code=1)

    if postprocess_backend == "sync" and len(effective_uids) == 1:
        return [
            str(path)
            for path in create_measurement_sets(
                input_root=input_root,
                output_root=output_root,
                asdm_uid=effective_uids[0],
                casa_data_root=casa_data_root,
                skip_casa_data_update=skip_casa_data_update,
                overwrite=overwrite_outputs,
                logger_fn=typer.echo,
            )
        ]

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(effective_uids),
        }

    worker_casa_data = _preflight_casa_data(output_root, casa_data_root, skip_casa_data_update)
    skip_casa_data_update = True  # workers reuse what master just populated

    log_paths = [_stage_log_path(output_root, uid, "unpack") for uid in effective_uids]
    typer.echo(f"Per-UID unpack logs: {log_paths[0].parent}/<uid>.unpack.log")
    job_kwargs = [
        {
            "input_root": str(input_root),
            "raw_output_root": str(output_root),
            "asdm_uid": uid,
            "casa_data_root": str(worker_casa_data),
            "skip_casa_data_update": skip_casa_data_update,
            "overwrite": overwrite_outputs,
        }
        for uid in effective_uids
    ]

    stage_label = "Slurm unpack" if postprocess_backend == "slurm" else "Unpack"
    if postprocess_backend == "slurm":
        with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
            unpack_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_unpack_single_uid,
                job_kwargs=job_kwargs,
                job_uids=effective_uids,
                stage_label=stage_label,
                log_paths=log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
            )
    else:
        unpack_results = _run_uid_stage(
            backend=None,
            postprocess_backend=postprocess_backend,
            task_fn=_unpack_single_uid,
            job_kwargs=job_kwargs,
            job_uids=effective_uids,
            stage_label=stage_label,
            log_paths=log_paths,
            continue_on_error=continue_on_error,
            failures=failures,
        )
    outputs: list[str] = []
    for result in unpack_results:
        outputs.extend(result)
    return outputs


def _run_calibrate_jobs(
    *,
    input_root: Path,
    raw_ms_root: Path,
    output_root: Path,
    asdm_uids: list[str],
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    casa_data_root: Optional[Path],
    skip_casa_data_update: bool,
    overwrite_outputs: bool,
    clean_intermediate: bool,
    continue_on_error: bool = True,
    failures: Optional[list[StageFailure]] = None,
    keep_working_copies: bool = False,
) -> list[str]:
    """Calibrate raw MeasurementSets, one isolated subprocess per UID.

    UIDs whose calibrated output already carries a completion marker are skipped
    and reported as outputs, so an interrupted batch can simply be re-run.

    A single UID (which is also how each worker subprocess re-enters this
    function) runs in-process. Several UIDs always run one subprocess each so a
    CASA crash (for example an uncaught ``casacore::ArrayError``) on one
    execution block cannot take the rest of the batch down; failed UIDs are
    appended to ``failures`` when ``continue_on_error`` is set.
    """
    from .services.archive import (
        cleanup_intermediate_calibration_data,
        create_calibrated_measurement_sets,
    )

    if failures is None:
        failures = []

    if clean_intermediate and postprocess_backend == "slurm":
        typer.echo(
            "--clean-intermediate-files is not supported with --postprocess-backend=slurm.",
            err=True,
        )
        raise typer.Exit(code=2)

    effective_uids = asdm_uids or dedupe_keep_order(_extract_uids_from_raw_ms_root(raw_ms_root))
    if not effective_uids:
        typer.echo("No raw MeasurementSets found to calibrate.", err=True)
        raise typer.Exit(code=1)

    if postprocess_backend == "sync" and len(effective_uids) == 1:
        return [
            str(path)
            for path in create_calibrated_measurement_sets(
                input_root=input_root,
                raw_ms_root=raw_ms_root,
                output_root=output_root,
                asdm_uid=effective_uids[0],
                casa_data_root=casa_data_root,
                skip_casa_data_update=skip_casa_data_update,
                overwrite=overwrite_outputs,
                clean_intermediate=clean_intermediate,
                logger_fn=typer.echo,
                remove_working_copy=not keep_working_copies,
            )
        ]

    effective_uids, done_outputs = _partition_completed_calibrations(
        output_root, effective_uids, overwrite_outputs
    )
    if not effective_uids:
        typer.echo("All requested UIDs are already calibrated; nothing to do.")
        return done_outputs

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(effective_uids),
        }

    worker_casa_data = _preflight_casa_data(output_root, casa_data_root, skip_casa_data_update)
    skip_casa_data_update = True  # workers reuse what master just populated

    log_paths = [_stage_log_path(output_root, uid, "calibrate") for uid in effective_uids]
    typer.echo(f"Per-UID calibration logs: {log_paths[0].parent}/<uid>.calibrate.log")
    typer.echo(f"Calibrating {len(effective_uids)} UID(s)")
    job_kwargs = [
        {
            "input_root": str(input_root),
            "raw_ms_root": str(raw_ms_root),
            "calibrated_output_root": str(output_root),
            "asdm_uid": uid,
            "casa_data_root": str(worker_casa_data),
            "skip_casa_data_update": skip_casa_data_update,
            "overwrite": overwrite_outputs,
            "clean_intermediate": False,
            "keep_working_copies": keep_working_copies,
        }
        for uid in effective_uids
    ]

    stage_label = "Slurm calibrate" if postprocess_backend == "slurm" else "Calibrate"
    if postprocess_backend == "slurm":
        with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
            calibrate_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_calibrate_single_uid,
                job_kwargs=job_kwargs,
                job_uids=effective_uids,
                stage_label=stage_label,
                log_paths=log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
            )
    else:
        calibrate_results = _run_uid_stage(
            backend=None,
            postprocess_backend=postprocess_backend,
            task_fn=_calibrate_single_uid,
            job_kwargs=job_kwargs,
            job_uids=effective_uids,
            stage_label=stage_label,
            log_paths=log_paths,
            continue_on_error=continue_on_error,
            failures=failures,
        )
    outputs: list[str] = list(done_outputs)
    for result in calibrate_results:
        outputs.extend(result)

    if clean_intermediate:
        if failures:
            typer.echo(
                "Skipping --clean-intermediate-files because "
                f"{len(failures)} UID(s) failed; raw data is kept for a re-run.",
                err=True,
            )
        else:
            cleanup_intermediate_calibration_data(
                raw_ms_root,
                output_root,
                [Path(path) for path in outputs],
                logger_fn=typer.echo,
            )
    return outputs


def _archive_stem(archive_path: Path) -> str:
    """Return the archive name with all recognised compression suffixes stripped."""
    name = archive_path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.lower().endswith(suffix):
            return name[: len(name) - len(suffix)]
    return archive_path.stem


def _has_single_top_level_dir(archive: tarfile.TarFile) -> bool:
    """Return True when every member sits under a single top-level directory."""
    roots: set[str] = set()
    for member in archive.getmembers():
        top = Path(member.name).parts[0] if Path(member.name).parts else ""
        roots.add(top)
    return len(roots) == 1


def _safe_extract_tar_archive(archive_path: Path, destination: Path) -> list[Path]:
    """Extract a tarball while refusing absolute or escaping paths.

    If the archive is flat (no common top-level directory), all contents are
    placed inside a subdirectory named after the archive stem so they never
    spill into the destination root.
    """
    extracted: list[Path] = []
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        if _has_single_top_level_dir(archive):
            extract_root = destination
        else:
            extract_root = destination / _archive_stem(archive_path)
            extract_root.mkdir(parents=True, exist_ok=True)
        extract_root_resolved = extract_root.resolve()

        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                typer.echo(f"Skipping unsafe archive member: {member.name}", err=True)
                continue
            resolved = (extract_root / member.name).resolve()
            if not str(resolved).startswith(str(extract_root_resolved)):
                typer.echo(f"Skipping escaping archive member: {member.name}", err=True)
                continue
            # Also ensure nothing escapes the original destination root.
            if not str(resolved).startswith(str(destination_resolved)):
                typer.echo(f"Skipping escaping archive member: {member.name}", err=True)
                continue
            archive.extract(member, extract_root, filter="data")
            if not member.isdir():
                extracted.append(resolved)
    return extracted


def _find_archives(root: Path, recursive: bool) -> list[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    archives = []
    for path in candidates:
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".tar") or name.endswith(".tgz") or name.endswith(".tar.gz"):
            archives.append(path)
    return sorted(archives)


def _archive_done_marker(archive_path: Path) -> Path:
    return archive_path.parent / (archive_path.name + ".done")


def _extract_single_archive(
    *,
    archive_path: str,
    destination: str,
    delete_archive: bool,
) -> list[str]:
    """Extract one tarball on a Slurm worker; optionally delete it afterwards.

    Fully self-contained so cloudpickle can serialize it without pulling in
    cli_products module globals (tarfile, typer, …).
    """
    import tarfile as _tarfile
    from pathlib import Path as _Path

    def _stem(p: "_Path") -> str:
        name = p.name
        for sfx in (".tar.gz", ".tgz", ".tar"):
            if name.lower().endswith(sfx):
                return name[: len(name) - len(sfx)]
        return p.stem

    def _single_top_level(tf: "_tarfile.TarFile") -> bool:
        roots: set[str] = set()
        for m in tf.getmembers():
            parts = _Path(m.name).parts
            roots.add(parts[0] if parts else "")
        return len(roots) == 1

    archive = _Path(archive_path)
    dest = _Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    destination_resolved = dest.resolve()
    with _tarfile.open(archive, "r:*") as tf:
        if _single_top_level(tf):
            extract_root = dest
        else:
            extract_root = dest / _stem(archive)
            extract_root.mkdir(parents=True, exist_ok=True)
        extract_root_resolved = extract_root.resolve()

        for member in tf.getmembers():
            member_path = _Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            resolved = (extract_root / member.name).resolve()
            if not str(resolved).startswith(str(extract_root_resolved)):
                continue
            if not str(resolved).startswith(str(destination_resolved)):
                continue
            tf.extract(member, extract_root, filter="data")
            if not member.isdir():
                extracted.append(str(resolved))

    # Write marker so re-runs with --skip-existing can detect completion.
    (_Path(archive_path + ".done")).write_text("")

    if delete_archive:
        archive.unlink(missing_ok=True)
    return extracted


def _run_extract_jobs(
    *,
    source: Path,
    target: Path,
    archives: list[Path],
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    delete_archives: bool,
    skip_existing: bool = False,
) -> tuple[list[str], list[str]]:
    if skip_existing:
        pending = [a for a in archives if not _archive_done_marker(a).exists()]
        skipped = len(archives) - len(pending)
        if skipped:
            typer.echo(f"Skipped {skipped} already-extracted archive(s).")
        archives = pending

    if not archives:
        typer.echo("Nothing to extract.")
        return [], []

    if postprocess_backend == "sync":
        extracted_files: list[str] = []
        failed_archives: list[str] = []
        for archive_path in archives:
            try:
                files = _safe_extract_tar_archive(archive_path, target)
                extracted_files.extend(str(p) for p in files)
                typer.echo(f"Extracted {archive_path}")
                _archive_done_marker(archive_path).write_text("")
                if delete_archives:
                    archive_path.unlink(missing_ok=True)
            except (tarfile.TarError, OSError, ValueError) as exc:
                failed_archives.append(str(archive_path))
                typer.echo(f"Failed to extract {archive_path}: {exc}", err=True)
        return extracted_files, failed_archives

    archive_labels = [a.name for a in archives]
    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(archives),
        }

    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        extract_task = backend.delayed(_extract_single_archive)
        extract_jobs = [
            extract_task(
                archive_path=str(archive_path),
                destination=str(target),
                delete_archive=delete_archives,
            )
            for archive_path in archives
        ]
        extract_results = _compute_jobs_with_progress(
            backend=backend,
            jobs=extract_jobs,
            job_uids=archive_labels,
            stage_label="Slurm extract",
        )

    extracted_files = []
    for result in extract_results:
        extracted_files.extend(result)
    return extracted_files, []


def _run_parallel_archive_jobs(
    *,
    download_root: Path,
    archive_output_root: Path,
    unpack_ms: bool,
    generate_calibrated_visibilities: bool,
    postprocess_backend: str,
    postprocess_backend_kwargs: dict[str, Any],
    casa_data_root: Optional[Path],
    skip_casa_data_update: bool,
    overwrite_archive_outputs: bool,
    continue_on_error: bool = True,
    failures: Optional[list[StageFailure]] = None,
    keep_working_copies: bool = False,
) -> tuple[list[str], list[str]]:
    if failures is None:
        failures = []
    raw_ms_root = archive_output_root / "raw_ms"
    calibrated_ms_root = archive_output_root / "calibrated_ms"

    if unpack_ms:
        asdm_uids = _extract_asdm_uids_from_download_root(download_root)
    elif generate_calibrated_visibilities:
        asdm_uids = _extract_uids_from_raw_ms_root(raw_ms_root)
    else:
        return [], []

    asdm_uids = dedupe_keep_order(asdm_uids)
    if not asdm_uids:
        typer.echo("No ASDM/raw-MS inputs found for archive post-processing.", err=True)
        raise typer.Exit(code=1)

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(asdm_uids),
        }

    typer.echo(
        "Running archive post-processing with "
        f"backend={postprocess_backend} for {len(asdm_uids)} UID(s)..."
    )
    raw_outputs: list[str] = []
    calibrated_outputs: list[str] = []

    # One CASA runtime data directory for both stages, populated on the submit node.
    worker_casa_data = _preflight_casa_data(
        archive_output_root, casa_data_root, skip_casa_data_update
    )
    skip_casa_data_update = True

    stage_prefix = "Slurm " if postprocess_backend == "slurm" else ""
    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        calibrate_uids = list(asdm_uids)
        if unpack_ms:
            unpack_log_paths = [_stage_log_path(raw_ms_root, uid, "unpack") for uid in asdm_uids]
            typer.echo(f"Per-UID unpack logs: {unpack_log_paths[0].parent}/<uid>.unpack.log")
            unpack_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_unpack_single_uid,
                job_kwargs=[
                    {
                        "input_root": str(download_root),
                        "raw_output_root": str(raw_ms_root),
                        "asdm_uid": uid,
                        "casa_data_root": str(worker_casa_data),
                        "skip_casa_data_update": skip_casa_data_update,
                        "overwrite": overwrite_archive_outputs,
                    }
                    for uid in asdm_uids
                ],
                job_uids=asdm_uids,
                stage_label=f"{stage_prefix}unpack",
                log_paths=unpack_log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
            )
            calibrate_uids = []
            for uid, result in zip(asdm_uids, unpack_results):
                raw_outputs.extend(result)
                if result:
                    calibrate_uids.append(uid)
            skipped = [uid for uid in asdm_uids if uid not in calibrate_uids]
            if skipped and generate_calibrated_visibilities:
                typer.echo(
                    f"Skipping calibration for {len(skipped)} UID(s) whose unpack failed: "
                    + ", ".join(skipped),
                    err=True,
                )

        if generate_calibrated_visibilities and calibrate_uids:
            calibrate_uids, done_outputs = _partition_completed_calibrations(
                calibrated_ms_root, calibrate_uids, overwrite_archive_outputs
            )
            calibrated_outputs.extend(done_outputs)

        if generate_calibrated_visibilities and calibrate_uids:
            calibrate_log_paths = [
                _stage_log_path(calibrated_ms_root, uid, "calibrate") for uid in calibrate_uids
            ]
            typer.echo(
                f"Per-UID calibration logs: {calibrate_log_paths[0].parent}/<uid>.calibrate.log"
            )
            calibrate_results = _run_uid_stage(
                backend=backend,
                postprocess_backend=postprocess_backend,
                task_fn=_calibrate_single_uid,
                job_kwargs=[
                    {
                        "input_root": str(download_root),
                        "raw_ms_root": str(raw_ms_root),
                        "calibrated_output_root": str(calibrated_ms_root),
                        "asdm_uid": uid,
                        "casa_data_root": str(worker_casa_data),
                        "skip_casa_data_update": skip_casa_data_update,
                        "overwrite": overwrite_archive_outputs,
                        "clean_intermediate": False,
                        "keep_working_copies": keep_working_copies,
                    }
                    for uid in calibrate_uids
                ],
                job_uids=calibrate_uids,
                stage_label=f"{stage_prefix}calibrate",
                log_paths=calibrate_log_paths,
                continue_on_error=continue_on_error,
                failures=failures,
            )
            for result in calibrate_results:
                calibrated_outputs.extend(result)

    return raw_outputs, calibrated_outputs


@products_app.command("resolve")
def products_resolve(
    metadata_csv: Optional[Path] = typer.Option(
        None,
        "--metadata-csv",
        help="Metadata CSV containing member_ous_uid rows.",
    ),
    member_ous_uid: Optional[List[str]] = typer.Option(
        None,
        "--member-ous-uid",
        help="Direct member_ous_uid values. Repeat or pass comma-separated values.",
    ),
    member_limit: Optional[int] = typer.Option(
        None,
        "--member-limit",
        min=1,
        help="Max metadata member_ous_uid rows to read (default: unlimited).",
    ),
    save_member_ous_uid_list: Optional[Path] = typer.Option(
        None,
        "--save-member-ous-uid-list",
        help="Optional text file to write extracted member_ous_uid list (one per line).",
    ),
    save_products_csv_path: Path = typer.Option(
        default_output_path("resolved_products.csv"),
        "--save-products-csv",
        help="Destination CSV for resolved DataLink products.",
    ),
) -> None:
    """Extract member_ous_uid values and resolve ALMA DataLink products."""
    member_uids = _parse_member_uid_options(member_ous_uid)
    if metadata_csv is not None:
        member_uids.extend(_read_member_uids_from_metadata(metadata_csv, member_limit))
    member_uids = dedupe_keep_order([uid for uid in member_uids if uid])
    if not member_uids:
        typer.echo(
            "Provide --metadata-csv and/or --member-ous-uid to extract member_ous_uid values.",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(f"Extracted member_ous_uid values: {len(member_uids)}")
    if save_member_ous_uid_list is not None:
        uid_path = save_member_ous_uid_list.expanduser().resolve()
        uid_path.parent.mkdir(parents=True, exist_ok=True)
        uid_path.write_text("\n".join(member_uids) + "\n", encoding="utf-8")
        typer.echo(f"Saved member_ous_uid list: {uid_path}")

    products = _resolve_products_from_inputs(
        products_csv=None,
        metadata_csv=None,
        member_ous_uid=member_uids,
        member_limit=None,
        save_products_csv_path=save_products_csv_path,
    )
    typer.echo(f"Resolved products: {len(products)}")


def _update_manifest_measurement_sets(
    manifest_path: Optional[str], raw_mss: list[str], calibrated_mss: list[str]
) -> None:
    """Record post-processing outputs in the download manifest written earlier."""
    if not manifest_path:
        return
    import json

    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(manifest, dict):
        return
    manifest["raw_measurement_sets"] = list(raw_mss)
    manifest["calibrated_measurement_sets"] = list(calibrated_mss)
    try:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        typer.echo(f"Could not update manifest {path}", err=True)


def _cleanup_after_download_postprocess(
    *,
    summary: Any,
    destination: Path,
    archive_root: Path,
    calibrated_mss: list[str],
) -> None:
    """Apply --clean-intermediate-files once every UID has been calibrated successfully."""
    from .services.archive import cleanup_intermediate_calibration_data
    from .services.download import _cleanup_download_inputs

    calibrated_paths = [Path(path) for path in calibrated_mss]
    cleanup_intermediate_calibration_data(
        archive_root / "raw_ms",
        archive_root / "calibrated_ms",
        calibrated_paths,
        logger_fn=typer.echo,
    )
    _cleanup_download_inputs(
        destination,
        getattr(summary, "files", []) or [],
        getattr(summary, "extracted_files", []) or [],
        protected_roots=[archive_root, *calibrated_paths],
    )


@products_app.command("download")
def products_download(
    products_csv: Optional[Path] = typer.Option(
        None,
        "--products-csv",
        help="Previously resolved products CSV.",
    ),
    metadata_csv: Optional[Path] = typer.Option(
        None,
        "--metadata-csv",
        help="Metadata CSV containing member_ous_uid rows to resolve first.",
    ),
    member_ous_uid: Optional[List[str]] = typer.Option(
        None,
        "--member-ous-uid",
        help="Direct member_ous_uid values. Repeat or pass comma-separated values.",
    ),
    member_limit: Optional[int] = typer.Option(
        None,
        "--member-limit",
        min=1,
        help="Max metadata member_ous_uid rows to read (default: unlimited).",
    ),
    product_filter: str = typer.Option(
        "all",
        "--product-filter",
        help=(
            "Subset of resolved products to download. "
            "Choices: all, " + ", ".join(PRODUCT_TYPES) + "."
        ),
        case_sensitive=False,
    ),
    save_products_csv_path: Optional[Path] = typer.Option(
        default_output_path("resolved_products.csv"),
        "--save-products-csv",
        help="Save resolved products CSV before download.",
    ),
    destination: Path = typer.Option(
        default_output_path("downloads"),
        "--destination",
        help="Directory for downloaded files.",
    ),
    max_parallel: int = typer.Option(
        3,
        "--max-parallel",
        min=1,
        help=(
            "Max concurrent downloads across ALMA mirrors; capped at "
            f"{MAX_PARALLEL_TOTAL} ({MAX_PARALLEL_PER_MIRROR} per mirror)."
        ),
    ),
    extract_tar: bool = typer.Option(
        False,
        "--extract-tar",
        help="Extract downloaded tar/tgz archives.",
    ),
    unpack_ms: bool = typer.Option(
        False,
        "--unpack-ms",
        help="Import extracted ASDMs into raw MeasurementSets.",
    ),
    generate_calibrated_visibilities: bool = typer.Option(
        False,
        "--generate-calibrated-visibilities",
        help="Apply delivered calibrations and write calibrated MeasurementSets.",
    ),
    archive_output_root: Optional[Path] = typer.Option(
        None,
        "--archive-output-root",
        help="Root directory for archive_ms raw_ms/calibrated_ms products.",
    ),
    casa_data_root: Optional[Path] = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
    ),
    clean_intermediate_files: bool = typer.Option(
        False,
        "--clean-intermediate-files",
        help="Clean downloaded/intermediate raw files after calibrated outputs are created.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for unpack/calibration stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "02:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
    ),
    overwrite_archive_outputs: bool = typer.Option(
        False,
        "--overwrite-archive-outputs",
        help="Overwrite existing raw/calibrated MS outputs.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
    ),
    keep_working_copies: bool = typer.Option(
        False,
        "--keep-working-copies",
        help=(
            "Keep each UID's working directory (raw MS copy plus caltables, about 1.8x the "
            "raw MS) after its calibrated output is written. By default it is removed."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt and start download immediately.",
    ),
) -> None:
    """Download ALMA products and optionally unpack/calibrate archive data."""
    product_filter_normalized = product_filter.lower()
    if product_filter_normalized not in PRODUCT_TYPES:
        typer.echo(
            "Invalid --product-filter. Allowed values: " + ", ".join(sorted(PRODUCT_TYPES)),
            err=True,
        )
        raise typer.Exit(code=2)

    backend_normalized = postprocess_backend.lower()
    if backend_normalized not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    if max_parallel > MAX_PARALLEL_TOTAL:
        typer.echo(
            "Requested --max-parallel="
            f"{max_parallel} exceeds cap; clamping to {MAX_PARALLEL_TOTAL}."
        )
        max_parallel = MAX_PARALLEL_TOTAL

    products = _resolve_products_from_inputs(
        products_csv=products_csv,
        metadata_csv=metadata_csv,
        member_ous_uid=member_ous_uid,
        member_limit=member_limit,
        save_products_csv_path=save_products_csv_path,
    )
    filtered = filter_products(products, product_filter_normalized)
    if not filtered:
        typer.echo(f"No products matched --product-filter={product_filter_normalized}", err=True)
        raise typer.Exit(code=1)

    total_bytes = sum(product.content_length for product in filtered)
    typer.echo(f"Resolved products: {len(products)}")
    typer.echo(f"Selected for download: {len(filtered)} ({format_bytes(total_bytes)})")

    if not yes:
        unknown_sizes = sum(1 for product in filtered if product.content_length <= 0)
        message = (
            f"About to download {len(filtered)} product(s), total size {format_bytes(total_bytes)}"
        )
        if unknown_sizes:
            message += f" ({unknown_sizes} item(s) with unknown size)"
        message += ". Continue?"

        if not typer.confirm(message, default=True):
            typer.echo("Download cancelled.")
            raise typer.Exit(code=0)

    needs_archive_postprocess = unpack_ms or generate_calibrated_visibilities
    if generate_calibrated_visibilities and not unpack_ms:
        typer.echo(
            "--generate-calibrated-visibilities without --unpack-ms "
            "expects existing raw_ms outputs.",
            err=True,
        )

    archive_root = (
        archive_output_root.expanduser().resolve()
        if archive_output_root is not None
        else destination.expanduser().resolve() / "archive_ms"
    )

    if needs_archive_postprocess:
        # Download first, then unpack/calibrate one UID per subprocess (sync) or per
        # Slurm task, so a CASA crash on one execution block never aborts the run.
        summary = _download_products_with_progress(
            filtered,
            destination,
            max_parallel=max_parallel,
            extract_tar=extract_tar,
            unpack_ms=False,
            generate_calibrated_visibilities=False,
            clean_intermediate_files=False,
            archive_output_root=archive_root,
            casa_data_root=casa_data_root,
            skip_casa_data_update=skip_casa_data_update,
            logger_fn=typer.echo,
        )
        stage_failures: list[StageFailure] = []
        raw_mss, calibrated_mss = _run_parallel_archive_jobs(
            download_root=Path(summary.destination),
            archive_output_root=archive_root,
            unpack_ms=unpack_ms,
            generate_calibrated_visibilities=generate_calibrated_visibilities,
            postprocess_backend=backend_normalized,
            postprocess_backend_kwargs=(
                {
                    "queue": slurm_queue,
                    "project": slurm_project,
                    "walltime": slurm_walltime,
                    "cores": slurm_cores,
                    "memory": slurm_memory,
                    "n_workers": slurm_workers,
                }
                if backend_normalized == "slurm"
                else {}
            ),
            casa_data_root=casa_data_root,
            skip_casa_data_update=skip_casa_data_update,
            overwrite_archive_outputs=overwrite_archive_outputs,
            continue_on_error=continue_on_error,
            failures=stage_failures,
            keep_working_copies=keep_working_copies,
        )
        _update_manifest_measurement_sets(summary.manifest_path, raw_mss, calibrated_mss)
        typer.echo(f"Destination: {summary.destination}")
        typer.echo(f"Completed: {summary.files_completed}")
        typer.echo(f"Failed: {summary.files_failed}")
        if summary.manifest_path:
            typer.echo(f"Manifest: {summary.manifest_path}")
        if raw_mss:
            typer.echo("Raw MS products:")
            for raw_ms in raw_mss:
                typer.echo(f"  {raw_ms}")
        if calibrated_mss:
            typer.echo("Calibrated MS products:")
            for calibrated_ms in calibrated_mss:
                typer.echo(f"  {calibrated_ms}")
        if clean_intermediate_files and generate_calibrated_visibilities:
            if stage_failures:
                typer.echo(
                    "Skipping --clean-intermediate-files because "
                    f"{len(stage_failures)} UID(s) failed; inputs are kept for a re-run.",
                    err=True,
                )
            else:
                _cleanup_after_download_postprocess(
                    summary=summary,
                    destination=Path(summary.destination),
                    archive_root=archive_root,
                    calibrated_mss=calibrated_mss,
                )
        _exit_if_failures("Archive post-processing", stage_failures)
        return

    summary = _download_products_with_progress(
        filtered,
        destination,
        max_parallel=max_parallel,
        extract_tar=extract_tar,
        unpack_ms=unpack_ms,
        generate_calibrated_visibilities=generate_calibrated_visibilities,
        clean_intermediate_files=clean_intermediate_files,
        archive_output_root=archive_root,
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        logger_fn=typer.echo,
    )

    typer.echo(f"Destination: {summary.destination}")
    typer.echo(f"Completed: {summary.files_completed}")
    typer.echo(f"Failed: {summary.files_failed}")
    if summary.manifest_path:
        typer.echo(f"Manifest: {summary.manifest_path}")
    if summary.raw_measurement_sets:
        typer.echo("Raw MS products:")
        for raw_ms in summary.raw_measurement_sets:
            typer.echo(f"  {raw_ms}")
    if summary.calibrated_measurement_sets:
        typer.echo("Calibrated MS products:")
        for calibrated_ms in summary.calibrated_measurement_sets:
            typer.echo(f"  {calibrated_ms}")


@products_app.command("extract", hidden=True)
def products_extract(
    source_root: Path = typer.Option(
        default_output_path("downloads"),
        "--source-root",
        help="Directory containing downloaded archive files.",
    ),
    destination: Optional[Path] = typer.Option(
        None,
        "--destination",
        help="Extraction destination (defaults to --source-root).",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Recursively search --source-root for .tar/.tgz archives.",
    ),
    delete_archives: bool = typer.Option(
        False,
        "--delete-archives",
        help="Delete each archive after successful extraction.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for extraction stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "01:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers. Pass 0 to spawn one worker per archive.",
    ),
    slurm_scheduler_host: Optional[str] = typer.Option(
        None,
        "--slurm-scheduler-host",
        help=(
            "IP or hostname that Slurm workers use to reach the Dask scheduler. "
            "Set this to an internal/HPC network address when the public hostname "
            "is not reachable from compute nodes (e.g. 10.20.25.44)."
        ),
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip archives that have already been extracted (detected via a .done marker file).",
    ),
) -> None:
    """Extract ALMA archive tarballs as a standalone step."""
    backend_normalized = postprocess_backend.lower()
    if backend_normalized not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    source = source_root.expanduser().resolve()
    if not source.exists() or not source.is_dir():
        typer.echo(f"--source-root is not a directory: {source}", err=True)
        raise typer.Exit(code=2)

    target = destination.expanduser().resolve() if destination is not None else source
    target.mkdir(parents=True, exist_ok=True)

    archives = _find_archives(source, recursive)
    if not archives:
        typer.echo(f"No .tar/.tgz archives found under {source}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(archives)} archive(s) to extract.")
    extracted_files, failed_archives = _run_extract_jobs(
        source=source,
        target=target,
        archives=archives,
        postprocess_backend=backend_normalized,
        postprocess_backend_kwargs={
            "queue": slurm_queue,
            "project": slurm_project,
            "walltime": slurm_walltime,
            "cores": slurm_cores,
            "memory": slurm_memory,
            "n_workers": slurm_workers,
            **({"scheduler_host": slurm_scheduler_host} if slurm_scheduler_host else {}),
        },
        delete_archives=delete_archives,
        skip_existing=skip_existing,
    )

    typer.echo(f"Extracted files: {len(extracted_files)}")
    typer.echo(f"Failed archives: {len(failed_archives)}")
    if failed_archives:
        raise typer.Exit(code=1)


@products_app.command("unpack", hidden=True)
def products_unpack(
    input_root: Path = typer.Option(
        default_output_path("downloads"),
        "--input-root",
        help="Directory containing extracted ASDM directories.",
    ),
    output_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "raw_ms",
        "--output-root",
        help="Directory where raw MeasurementSets are written.",
    ),
    asdm_uid: Optional[List[str]] = typer.Option(
        None,
        "--asdm-uid",
        help="Optional ASDM UID(s) to process. Repeat or pass comma-separated values.",
    ),
    casa_data_root: Optional[Path] = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for unpack stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "02:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
    ),
    slurm_scheduler_host: Optional[str] = typer.Option(
        None,
        "--slurm-scheduler-host",
        help=(
            "IP or hostname that Slurm workers use to reach the Dask scheduler. "
            "Set this to an internal/HPC network address when the public hostname "
            "is not reachable from compute nodes (e.g. 10.20.25.44)."
        ),
    ),
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Overwrite existing raw MS outputs.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
    ),
) -> None:
    """Import ASDM directories into raw MeasurementSets as a standalone step."""
    backend_normalized = postprocess_backend.lower()
    if backend_normalized not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    parsed_uids = _parse_asdm_uid_options(asdm_uid)
    stage_failures: list[StageFailure] = []
    raw_outputs = _run_unpack_jobs(
        input_root=input_root.expanduser().resolve(),
        output_root=output_root.expanduser().resolve(),
        asdm_uids=parsed_uids,
        postprocess_backend=backend_normalized,
        postprocess_backend_kwargs={
            "queue": slurm_queue,
            "project": slurm_project,
            "walltime": slurm_walltime,
            "cores": slurm_cores,
            "memory": slurm_memory,
            "n_workers": slurm_workers,
            **({"scheduler_host": slurm_scheduler_host} if slurm_scheduler_host else {}),
        },
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        overwrite_outputs=overwrite_outputs,
        continue_on_error=continue_on_error,
        failures=stage_failures,
    )

    typer.echo(f"Raw MS products: {len(raw_outputs)}")
    for raw_ms in raw_outputs:
        typer.echo(f"  {raw_ms}")
    _exit_if_failures("Unpack", stage_failures)


@products_app.command("calibrate", hidden=True)
def products_calibrate(
    input_root: Path = typer.Option(
        default_output_path("downloads"),
        "--input-root",
        help="ALMA delivery root containing calibration products.",
    ),
    raw_ms_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "raw_ms",
        "--raw-ms-root",
        help="Directory containing raw MeasurementSets.",
    ),
    output_root: Path = typer.Option(
        default_output_path("downloads") / "archive_ms" / "calibrated_ms",
        "--output-root",
        help="Directory where calibrated MeasurementSets are written.",
    ),
    asdm_uid: Optional[List[str]] = typer.Option(
        None,
        "--asdm-uid",
        help="Optional UID(s) to calibrate. Repeat or pass comma-separated values.",
    ),
    casa_data_root: Optional[Path] = typer.Option(
        None,
        "--casa-data-root",
        help="Optional CASA runtime data directory.",
    ),
    skip_casa_data_update: bool = typer.Option(
        False,
        "--skip-casa-data-update",
        help="Do not auto-download CASA runtime data if missing.",
    ),
    postprocess_backend: str = typer.Option(
        "sync",
        "--postprocess-backend",
        help="Backend for calibration stage. Choices: sync, slurm.",
        case_sensitive=False,
    ),
    slurm_queue: str = typer.Option("normal", "--slurm-queue", help="Slurm queue/partition."),
    slurm_project: Optional[str] = typer.Option(
        None,
        "--slurm-project",
        help="Optional Slurm project/account.",
    ),
    slurm_walltime: str = typer.Option(
        "02:00:00",
        "--slurm-walltime",
        help="Slurm walltime per worker job (HH:MM:SS).",
    ),
    slurm_cores: int = typer.Option(
        1,
        "--slurm-cores",
        min=1,
        help="Cores per Slurm worker.",
    ),
    slurm_memory: str = typer.Option("4GB", "--slurm-memory", help="Memory per Slurm worker."),
    slurm_workers: int = typer.Option(
        4,
        "--slurm-workers",
        min=0,
        help="Number of Slurm workers for post-processing. Pass 0 to spawn one worker per UID.",
    ),
    slurm_scheduler_host: Optional[str] = typer.Option(
        None,
        "--slurm-scheduler-host",
        help=(
            "IP or hostname that Slurm workers use to reach the Dask scheduler. "
            "Set this to an internal/HPC network address when the public hostname "
            "is not reachable from compute nodes (e.g. 10.20.25.44)."
        ),
    ),
    overwrite_outputs: bool = typer.Option(
        False,
        "--overwrite-outputs",
        help="Overwrite existing calibrated MS outputs.",
    ),
    clean_intermediate_files: bool = typer.Option(
        False,
        "--clean-intermediate-files",
        help="Remove intermediate raw and working files after successful calibration.",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--fail-fast",
        help=(
            "Skip UIDs whose processing fails and carry on with the rest, then exit "
            "non-zero with a summary (default). --fail-fast aborts on the first failure."
        ),
    ),
    keep_working_copies: bool = typer.Option(
        False,
        "--keep-working-copies",
        help=(
            "Keep each UID's working directory (raw MS copy plus caltables, about 1.8x the "
            "raw MS) after its calibrated output is written. By default it is removed."
        ),
    ),
) -> None:
    """Create calibrated MeasurementSets as a standalone step."""
    backend_normalized = postprocess_backend.lower()
    if backend_normalized not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    parsed_uids = _parse_asdm_uid_options(asdm_uid)
    stage_failures: list[StageFailure] = []
    calibrated_outputs = _run_calibrate_jobs(
        input_root=input_root.expanduser().resolve(),
        raw_ms_root=raw_ms_root.expanduser().resolve(),
        output_root=output_root.expanduser().resolve(),
        asdm_uids=parsed_uids,
        postprocess_backend=backend_normalized,
        postprocess_backend_kwargs={
            "queue": slurm_queue,
            "project": slurm_project,
            "walltime": slurm_walltime,
            "cores": slurm_cores,
            "memory": slurm_memory,
            "n_workers": slurm_workers,
            **({"scheduler_host": slurm_scheduler_host} if slurm_scheduler_host else {}),
        },
        casa_data_root=casa_data_root,
        skip_casa_data_update=skip_casa_data_update,
        overwrite_outputs=overwrite_outputs,
        clean_intermediate=clean_intermediate_files,
        continue_on_error=continue_on_error,
        failures=stage_failures,
        keep_working_copies=keep_working_copies,
    )

    typer.echo(f"Calibrated MS products: {len(calibrated_outputs)}")
    for calibrated_ms in calibrated_outputs:
        typer.echo(f"  {calibrated_ms}")
    _exit_if_failures("Calibrate", stage_failures)
