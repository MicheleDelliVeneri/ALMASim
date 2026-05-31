"""Product resolution and download commands for the ALMASim CLI."""

from __future__ import annotations

import tarfile
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import sleep
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


def _compute_jobs_with_progress(
    *,
    backend: Any,
    jobs: list[Any],
    job_uids: list[str],
    stage_label: str,
) -> list[Any]:
    """Run a stage and show per-UID progress for asynchronous backends."""
    if not jobs:
        return []

    futures = backend.compute(jobs, sync=False)
    if not isinstance(futures, list):
        futures = [futures]

    completed: set[int] = set()
    failed: set[int] = set()
    with tqdm(total=len(futures), desc=stage_label, unit="uid", leave=True) as progress_bar:
        progress_bar.set_postfix_str(f"completed 0/{len(futures)}")
        while len(completed) < len(futures):
            for index, future in enumerate(futures):
                if index in completed:
                    continue

                state = _future_status(future)
                if state in {"error", "failed", "cancelled"}:
                    failed.add(index)

                if not _future_done(future):
                    continue

                completed.add(index)
                progress_bar.update(1)
                uid = job_uids[index] if index < len(job_uids) else f"job-{index + 1}"
                status_text = state if state else "finished"
                progress_bar.write(f"{stage_label} completed for {uid} [{status_text}]")
                if state in {"error", "failed"}:
                    try:
                        exc = future.exception()
                        if exc is not None:
                            progress_bar.write(f"  {uid} error: {exc}")
                    except Exception:
                        pass

            progress_bar.set_postfix_str(
                f"completed {len(completed)}/{len(futures)} failed {len(failed)}"
            )
            if len(completed) < len(futures):
                sleep(0.5)

    return backend.gather(futures)


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
    for line in process.stdout:
        print(line, end="", flush=True)
        tail_lines.append(line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        tail_text = "\n".join(tail_lines)
        raise RuntimeError(
            f"Unpack failed for {asdm_uid}.\n"
            f"Return code: {return_code}\n"
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
    for line in process.stdout:
        print(line, end="", flush=True)
        tail_lines.append(line.rstrip("\n"))

    return_code = process.wait()
    if return_code != 0:
        tail_text = "\n".join(tail_lines)
        raise RuntimeError(
            f"Calibration failed for {asdm_uid}.\n"
            f"Return code: {return_code}\n"
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
) -> None:
    """Ensure CASA runtime data is populated on the master node before Slurm workers start.

    Slurm compute nodes typically have no internet access, so the data must be
    downloaded once from the submit node and written to a shared filesystem path
    that all workers can read.
    """
    from .services.archive.unpack_ms import (
        ensure_casa_runtime_data,
        find_existing_casa_data,
    )

    casa_data = find_existing_casa_data(output_root, output_root, casa_data_root)
    typer.echo(f"Preflight: ensuring CASA runtime data at {casa_data} …")
    ensure_casa_runtime_data(casa_data, skip_update=skip_casa_data_update)


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
) -> list[str]:
    from .services.archive import create_measurement_sets

    if postprocess_backend == "sync":
        if not asdm_uids:
            return [
                str(path)
                for path in create_measurement_sets(
                    input_root=input_root,
                    output_root=output_root,
                    casa_data_root=casa_data_root,
                    skip_casa_data_update=skip_casa_data_update,
                    overwrite=overwrite_outputs,
                )
            ]

        outputs: list[str] = []
        for uid in asdm_uids:
            outputs.extend(
                str(path)
                for path in create_measurement_sets(
                    input_root=input_root,
                    output_root=output_root,
                    asdm_uid=uid,
                    casa_data_root=casa_data_root,
                    skip_casa_data_update=skip_casa_data_update,
                    overwrite=overwrite_outputs,
                )
            )
        return outputs

    effective_uids = asdm_uids or dedupe_keep_order(
        _extract_asdm_uids_from_download_root(input_root)
    )
    if not effective_uids:
        typer.echo("No ASDM directories found to unpack.", err=True)
        raise typer.Exit(code=1)

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(effective_uids),
        }

    _preflight_casa_data(output_root, casa_data_root, skip_casa_data_update)
    skip_casa_data_update = True  # workers reuse what master just populated

    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        unpack_task = backend.delayed(_unpack_single_uid)
        unpack_jobs = [
            unpack_task(
                input_root=str(input_root),
                raw_output_root=str(output_root),
                asdm_uid=uid,
                casa_data_root=str(casa_data_root) if casa_data_root else None,
                skip_casa_data_update=skip_casa_data_update,
                overwrite=overwrite_outputs,
            )
            for uid in effective_uids
        ]
        unpack_results = _compute_jobs_with_progress(
            backend=backend,
            jobs=unpack_jobs,
            job_uids=effective_uids,
            stage_label="Slurm unpack",
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
) -> list[str]:
    from .services.archive import create_calibrated_measurement_sets

    if postprocess_backend == "sync":
        if not asdm_uids:
            return [
                str(path)
                for path in create_calibrated_measurement_sets(
                    input_root=input_root,
                    raw_ms_root=raw_ms_root,
                    output_root=output_root,
                    casa_data_root=casa_data_root,
                    skip_casa_data_update=skip_casa_data_update,
                    overwrite=overwrite_outputs,
                    clean_intermediate=clean_intermediate,
                )
            ]

        outputs: list[str] = []
        for uid in asdm_uids:
            outputs.extend(
                str(path)
                for path in create_calibrated_measurement_sets(
                    input_root=input_root,
                    raw_ms_root=raw_ms_root,
                    output_root=output_root,
                    asdm_uid=uid,
                    casa_data_root=casa_data_root,
                    skip_casa_data_update=skip_casa_data_update,
                    overwrite=overwrite_outputs,
                    clean_intermediate=clean_intermediate,
                )
            )
        return outputs

    if clean_intermediate:
        typer.echo(
            "--clean-intermediate-files is not supported with --postprocess-backend=slurm.",
            err=True,
        )
        raise typer.Exit(code=2)

    effective_uids = asdm_uids or dedupe_keep_order(_extract_uids_from_raw_ms_root(raw_ms_root))
    if not effective_uids:
        typer.echo("No raw MeasurementSets found to calibrate.", err=True)
        raise typer.Exit(code=1)

    if postprocess_backend_kwargs.get("n_workers") == 0:
        postprocess_backend_kwargs = {
            **postprocess_backend_kwargs,
            "n_workers": len(effective_uids),
        }

    _preflight_casa_data(output_root, casa_data_root, skip_casa_data_update)
    skip_casa_data_update = True  # workers reuse what master just populated

    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        calibrate_task = backend.delayed(_calibrate_single_uid)
        calibrate_jobs = [
            calibrate_task(
                input_root=str(input_root),
                raw_ms_root=str(raw_ms_root),
                calibrated_output_root=str(output_root),
                asdm_uid=uid,
                casa_data_root=str(casa_data_root) if casa_data_root else None,
                skip_casa_data_update=skip_casa_data_update,
                overwrite=overwrite_outputs,
                clean_intermediate=False,
            )
            for uid in effective_uids
        ]
        calibrate_results = _compute_jobs_with_progress(
            backend=backend,
            jobs=calibrate_jobs,
            job_uids=effective_uids,
            stage_label="Slurm calibrate",
        )
    outputs: list[str] = []
    for result in calibrate_results:
        outputs.extend(result)
    return outputs


def _safe_extract_tar_archive(archive_path: Path, destination: Path) -> list[Path]:
    """Extract a tarball while refusing absolute or escaping paths."""
    extracted: list[Path] = []
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                typer.echo(f"Skipping unsafe archive member: {member.name}", err=True)
                continue
            resolved = (destination / member.name).resolve()
            if not str(resolved).startswith(str(destination_resolved)):
                typer.echo(f"Skipping escaping archive member: {member.name}", err=True)
                continue
            archive.extract(member, destination, filter="data")
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

    archive = _Path(archive_path)
    dest = _Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    destination_resolved = dest.resolve()
    with _tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            member_path = _Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            resolved = (dest / member.name).resolve()
            if not str(resolved).startswith(str(destination_resolved)):
                continue
            tf.extract(member, dest, filter="data")
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
) -> tuple[list[str], list[str]]:
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

    with create_backend(postprocess_backend, **postprocess_backend_kwargs) as backend:
        if unpack_ms:
            unpack_task = backend.delayed(_unpack_single_uid)
            unpack_jobs = [
                unpack_task(
                    input_root=str(download_root),
                    raw_output_root=str(raw_ms_root),
                    asdm_uid=uid,
                    casa_data_root=str(casa_data_root) if casa_data_root else None,
                    skip_casa_data_update=skip_casa_data_update,
                    overwrite=overwrite_archive_outputs,
                )
                for uid in asdm_uids
            ]
            if postprocess_backend == "slurm":
                unpack_results = _compute_jobs_with_progress(
                    backend=backend,
                    jobs=unpack_jobs,
                    job_uids=asdm_uids,
                    stage_label="Slurm unpack",
                )
            else:
                unpack_results = backend.compute(unpack_jobs, sync=True)
            for result in unpack_results:
                raw_outputs.extend(result)

        if generate_calibrated_visibilities:
            calibrate_task = backend.delayed(_calibrate_single_uid)
            calibrate_jobs = [
                calibrate_task(
                    input_root=str(download_root),
                    raw_ms_root=str(raw_ms_root),
                    calibrated_output_root=str(calibrated_ms_root),
                    asdm_uid=uid,
                    casa_data_root=str(casa_data_root) if casa_data_root else None,
                    skip_casa_data_update=skip_casa_data_update,
                    overwrite=overwrite_archive_outputs,
                    clean_intermediate=False,
                )
                for uid in asdm_uids
            ]
            if postprocess_backend == "slurm":
                calibrate_results = _compute_jobs_with_progress(
                    backend=backend,
                    jobs=calibrate_jobs,
                    job_uids=asdm_uids,
                    stage_label="Slurm calibrate",
                )
            else:
                calibrate_results = backend.compute(calibrate_jobs, sync=True)
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

    if backend_normalized == "slurm" and needs_archive_postprocess:
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
        raw_mss, calibrated_mss = _run_parallel_archive_jobs(
            download_root=Path(summary.destination),
            archive_output_root=archive_root,
            unpack_ms=unpack_ms,
            generate_calibrated_visibilities=generate_calibrated_visibilities,
            postprocess_backend=backend_normalized,
            postprocess_backend_kwargs={
                "queue": slurm_queue,
                "project": slurm_project,
                "walltime": slurm_walltime,
                "cores": slurm_cores,
                "memory": slurm_memory,
                "n_workers": slurm_workers,
            },
            casa_data_root=casa_data_root,
            skip_casa_data_update=skip_casa_data_update,
            overwrite_archive_outputs=overwrite_archive_outputs,
        )
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
        if clean_intermediate_files:
            typer.echo(
                "--clean-intermediate-files is not yet applied in slurm post-processing mode.",
                err=True,
            )
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
) -> None:
    """Import ASDM directories into raw MeasurementSets as a standalone step."""
    backend_normalized = postprocess_backend.lower()
    if backend_normalized not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    parsed_uids = _parse_asdm_uid_options(asdm_uid)
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
    )

    typer.echo(f"Raw MS products: {len(raw_outputs)}")
    for raw_ms in raw_outputs:
        typer.echo(f"  {raw_ms}")


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
) -> None:
    """Create calibrated MeasurementSets as a standalone step."""
    backend_normalized = postprocess_backend.lower()
    if backend_normalized not in {"sync", "slurm"}:
        typer.echo("--postprocess-backend must be one of: sync, slurm.", err=True)
        raise typer.Exit(code=2)

    parsed_uids = _parse_asdm_uid_options(asdm_uid)
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
    )

    typer.echo(f"Calibrated MS products: {len(calibrated_outputs)}")
    for calibrated_ms in calibrated_outputs:
        typer.echo(f"  {calibrated_ms}")
