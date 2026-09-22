"""Atomic tar extraction.

Every stage that extracts an archive from the ALMA archive goes through
:func:`extract_tar_atomically`. The point is a single invariant that the unpack
stage relies on: **a ``*.asdm.sdm`` directory is either absent or complete**.

Extracting straight into the final location breaks that. An ASDM tar holds
thousands of members and takes minutes to unpack over NFS, and the
``<uid>.asdm.sdm`` directory becomes visible after the first member is written.
An unpack job that lists ASDMs during that window finds a directory that passes
every existence check and hands it to ``importasdm``, which then fails on the
first table that is not there yet (``ASDMUtilsException: File not found
.../ExecBlock.xml``) — or worse, if the archive order is unlucky, imports an
ASDM whose binary data is still arriving. A download job killed mid-extraction
leaves the same half-directory behind permanently.

So the archive is extracted into a hidden staging directory *next to* the
destination (same filesystem, so ``rename`` is atomic) and only then published:
each directory that does not yet exist at the destination is moved into place
with one ``rename``; directories that already exist (the shared
``<project>/science_goal…/member…/raw`` tree) are merged one level down; an
existing ``*.asdm.sdm`` is replaced wholesale, because the archive is the source
of truth and whatever is on disk is by definition suspect. On any failure the
staging directory is removed and the tar is left in place, so nothing partial
is ever visible under its final name.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Callable, Iterable, Optional

EXTRACTION_TMP_PREFIX = ".extracting-"
"""Prefix of the hidden staging directories used while an archive is extracted.

Directory walkers that look for ASDMs must prune these: they hold half-written
trees that look like ASDMs but are not yet published.
"""

ASDM_SUFFIX = ".asdm.sdm"

_ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")


def is_extraction_tmp_dir(name: str) -> bool:
    """Return True for the basename of an in-progress extraction staging dir."""
    return name.startswith(EXTRACTION_TMP_PREFIX)


def archive_stem(archive_path: Path) -> str:
    """``foo.tar`` / ``foo.tgz`` / ``foo.tar.gz`` -> ``foo``."""
    name = archive_path.name
    lower = name.lower()
    for suffix in _ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: len(name) - len(suffix)]
    return archive_path.stem


def _has_single_top_level_dir(members: Iterable[tarfile.TarInfo]) -> bool:
    roots: set[str] = set()
    for member in members:
        parts = Path(member.name).parts
        roots.add(parts[0] if parts else "")
    return len(roots) == 1


def _member_name_is_safe(member: tarfile.TarInfo) -> bool:
    member_path = Path(member.name)
    return not (member_path.is_absolute() or ".." in member_path.parts)


def _member_is_safe(member: tarfile.TarInfo, extract_root: Path) -> bool:
    if not _member_name_is_safe(member):
        return False
    resolved = (extract_root / member.name).resolve()
    try:
        resolved.relative_to(extract_root.resolve())
    except ValueError:
        return False
    return True


def _publish_tree(staged: Path, final: Path, *, replace_suffixes: tuple[str, ...]) -> None:
    """Move everything under ``staged`` into ``final`` with atomic renames.

    A directory that does not exist under ``final`` is moved with a single
    ``rename`` and therefore appears complete or not at all. A directory that
    already exists is merged recursively, except when its name carries one of
    ``replace_suffixes`` (an ASDM): then the staged copy replaces it wholesale.
    Files are moved with ``os.replace``.
    """
    final.mkdir(parents=True, exist_ok=True)
    for entry in sorted(staged.iterdir(), key=lambda p: p.name):
        target = final / entry.name
        if entry.is_dir() and not entry.is_symlink():
            if not target.exists():
                os.rename(entry, target)
                continue
            if target.is_dir() and not target.name.endswith(replace_suffixes):
                _publish_tree(entry, target, replace_suffixes=replace_suffixes)
                entry.rmdir()
                continue
            # An existing ASDM (or a file squatting on the name): replace it.
            # Move the old one aside first so the name is never a mix of both.
            retired = final / f"{EXTRACTION_TMP_PREFIX}retired-{entry.name}-{uuid.uuid4().hex}"
            os.rename(target, retired)
            os.rename(entry, target)
            if retired.is_dir() and not retired.is_symlink():
                shutil.rmtree(retired, ignore_errors=True)
            else:
                retired.unlink(missing_ok=True)
        else:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            os.replace(entry, target)


def extract_tar_atomically(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    flat_archive_subdir: bool = True,
    on_skipped_member: Optional[Callable[[str], None]] = None,
) -> list[Path]:
    """Extract ``archive_path`` under ``destination`` so that no partial tree is
    ever visible at its final path.

    Members that are absolute or escape the extraction root are skipped and
    reported through ``on_skipped_member``. When ``flat_archive_subdir`` is
    true and the archive has no single top-level directory, its contents are
    placed under ``destination/<archive stem>`` so they never spill into the
    destination root.

    Returns the final paths of the extracted regular files. Raises on any
    failure (unreadable or truncated tar, I/O error) after removing the staging
    directory; the archive itself is left untouched for a retry.
    """
    archive = Path(archive_path)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)

    staging = dest / f"{EXTRACTION_TMP_PREFIX}{archive_stem(archive)}-{uuid.uuid4().hex}"
    staging.mkdir()

    extracted: list[Path] = []
    try:
        with tarfile.open(archive, "r:*") as tar:
            members = tar.getmembers()
            # Unsafe members are skipped, so they must not decide the layout.
            safe_members = [m for m in members if _member_name_is_safe(m)]
            if flat_archive_subdir and not _has_single_top_level_dir(safe_members):
                extract_root = staging / archive_stem(archive)
                final_root = dest / archive_stem(archive)
                extract_root.mkdir()
            else:
                extract_root = staging
                final_root = dest

            for member in members:
                if not _member_is_safe(member, extract_root):
                    if on_skipped_member is not None:
                        on_skipped_member(member.name)
                    continue
                tar.extract(member, extract_root, filter="data")
                if not member.isdir():
                    extracted.append(final_root / member.name)

        _publish_tree(extract_root, final_root, replace_suffixes=(ASDM_SUFFIX,))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    shutil.rmtree(staging, ignore_errors=True)
    return extracted
