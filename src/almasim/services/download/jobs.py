"""Re-running and removing download jobs from their stored records (MADIV-28 T6.7).

A job record (the backend's database row, or anything with the same
attributes) keeps the files it downloaded, the member UIDs it resolved and
the post-processing options it ran with. ``stored_products`` rebuilds the
product list for a re-download, ``redownload_options`` merges a request's
overrides with the stored options, and ``delete_job_files`` removes what a
job wrote, but only inside its destination.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import UnsafePathError, safe_join
from .models import DataProduct


def json_list(value: Any) -> list:
    """A list stored as JSON text ([] when empty or not a list)."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def stored_products(record: Any) -> list[DataProduct]:
    """The products a job downloaded, from its file records (original access URLs)."""
    return [
        DataProduct(
            access_url=f.access_url,
            uid="",
            filename=f.filename,
            content_length=int(f.content_length),
            content_type="",
            product_type="other",
        )
        for f in record.files
    ]


@dataclass(frozen=True)
class RedownloadOptions:
    """How a re-download runs: the request's values where given, else the job's."""

    max_parallel: int
    extract_tar: bool
    unpack_ms: bool
    generate_calibrated_visibilities: bool
    clean_intermediate_files: bool
    archive_output_root: str | None
    casa_data_root: str | None
    skip_casa_data_update: bool


def redownload_options(
    record: Any, request: Any, *, default_max_parallel: int = 3
) -> RedownloadOptions:
    """Merge a re-download request (fields may be None) with the job's stored options.

    Calibrated visibilities need the raw MS, and both need the tar extracted.
    """

    def pick(name: str, stored: Any) -> Any:
        value = getattr(request, name, None)
        return value if value is not None else stored

    unpack_ms = bool(pick("unpack_ms", bool(record.unpack_ms)))
    calibrated = bool(
        pick("generate_calibrated_visibilities", bool(record.generate_calibrated_visibilities))
    )
    stored_extract = bool(record.unpack_ms) or bool(record.generate_calibrated_visibilities)
    return RedownloadOptions(
        max_parallel=pick("max_parallel", default_max_parallel),
        extract_tar=bool(pick("extract_tar", stored_extract)) or unpack_ms or calibrated,
        unpack_ms=unpack_ms or calibrated,
        generate_calibrated_visibilities=calibrated,
        clean_intermediate_files=bool(
            pick("clean_intermediate_files", bool(record.clean_intermediate_files))
        ),
        archive_output_root=pick("archive_output_root", record.archive_output_root),
        casa_data_root=pick("casa_data_root", record.casa_data_root),
        skip_casa_data_update=bool(
            pick("skip_casa_data_update", bool(record.skip_casa_data_update))
        ),
    )


def delete_job_files(record: Any) -> tuple[int, int]:  # noqa: C901
    """Delete a job's downloads (and leftover ``.part`` files) and the MSs it made.

    Only paths inside the job's destination are removed; a stored name that
    points elsewhere is skipped (MADIV-28 T1.9). Removal is best effort.
    Returns (entries deleted, entries skipped as outside the destination).
    """
    dest = Path(record.destination)
    deleted = skipped = 0
    for file_rec in record.files:
        for name in (file_rec.filename, file_rec.filename + ".part"):
            try:
                path = safe_join(dest, name)
            except UnsafePathError:
                skipped += 1
                continue
            try:
                if path.is_file():
                    path.unlink()
                    deleted += 1
            except OSError:
                pass  # best-effort; don't fail the delete if a file is gone

    generated = [
        *json_list(record.raw_measurement_sets),
        *json_list(record.calibrated_measurement_sets),
    ]
    if record.manifest_path:
        generated.append(record.manifest_path)
    for name in generated:
        try:
            path = safe_join(dest, name)
        except UnsafePathError:
            skipped += 1
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
                deleted += 1
            elif path.is_file():
                path.unlink()
                deleted += 1
        except OSError:
            pass
    return deleted, skipped
