"""Tests for atomic tar extraction (almasim.services.extraction)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from almasim.services.extraction import (
    EXTRACTION_TMP_PREFIX,
    archive_stem,
    extract_tar_atomically,
    is_extraction_tmp_dir,
)

PROJECT = "2023.1.00001.S"
MEMBER = (
    f"{PROJECT}/science_goal.uid___A001_X1_X1/group.uid___A001_X1_X2/member.uid___A001_X1_X3/raw"
)
UID = "uid___A002_X1_X1"
ASDM = f"{MEMBER}/{UID}.asdm.sdm"


def _add(tar: tarfile.TarFile, name: str, data: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _asdm_tar(path: Path, uid: str = UID, n_bdf: int = 3) -> Path:
    """A tar laid out like a real ALMA ASDM delivery: project tree on top."""
    asdm = f"{MEMBER}/{uid}.asdm.sdm"
    with tarfile.open(path, "w") as tar:
        _add(tar, f"{asdm}/ASDM.xml", b"<ASDM/>")
        _add(tar, f"{asdm}/Main.xml", b"<Main/>")
        for i in range(n_bdf):
            _add(tar, f"{asdm}/ASDMBinary/{uid}_bdf{i}", b"0123456789")
        _add(tar, f"{asdm}/ExecBlock.xml", b"<ExecBlock/>")
    return path


def _no_staging_left(dest: Path) -> bool:
    return not any(is_extraction_tmp_dir(p.name) for p in dest.rglob("*") if p.is_dir())


@pytest.mark.unit
def test_archive_stem():
    assert archive_stem(Path("a.tar")) == "a"
    assert archive_stem(Path("a.tgz")) == "a"
    assert archive_stem(Path("a.TAR.GZ")) == "a"
    assert archive_stem(Path("a.asdm.sdm.tar")) == "a.asdm.sdm"
    assert is_extraction_tmp_dir(f"{EXTRACTION_TMP_PREFIX}x")
    assert not is_extraction_tmp_dir("uid___A002_X1_X1.asdm.sdm")


@pytest.mark.unit
def test_extracts_project_tree_into_place(tmp_path):
    dest = tmp_path / "dest"
    files = extract_tar_atomically(_asdm_tar(tmp_path / "a.tar"), dest)

    asdm = dest / ASDM
    assert (asdm / "ASDM.xml").read_bytes() == b"<ASDM/>"
    assert (asdm / "ExecBlock.xml").is_file()
    assert len(list((asdm / "ASDMBinary").iterdir())) == 3
    assert sorted(files) == sorted(p for p in asdm.rglob("*") if p.is_file())
    assert all(f.is_file() for f in files)
    assert _no_staging_left(dest)
    assert (tmp_path / "a.tar").is_file(), "the caller decides whether to delete the tar"


@pytest.mark.unit
def test_merges_into_existing_project_tree_without_touching_siblings(tmp_path):
    """The project/science_goal/member dirs already exist for other ASDMs."""
    dest = tmp_path / "dest"
    sibling = dest / MEMBER / "uid___A002_X1_Xsib.asdm.sdm"
    sibling.mkdir(parents=True)
    (sibling / "ASDM.xml").write_text("sibling")
    (dest / PROJECT / "README").write_text("keep me")

    extract_tar_atomically(_asdm_tar(tmp_path / "a.tar"), dest)

    assert (sibling / "ASDM.xml").read_text() == "sibling"
    assert (dest / PROJECT / "README").read_text() == "keep me"
    assert (dest / ASDM / "ExecBlock.xml").is_file()
    assert _no_staging_left(dest)


@pytest.mark.unit
def test_replaces_existing_asdm_wholesale(tmp_path):
    """A stale half-ASDM at the final path is replaced, not merged over."""
    dest = tmp_path / "dest"
    stale = dest / ASDM
    stale.mkdir(parents=True)
    (stale / "ASDM.xml").write_text("old")
    (stale / "LeftoverFromOldExtraction.xml").write_text("stale")

    extract_tar_atomically(_asdm_tar(tmp_path / "a.tar"), dest)

    assert (stale / "ASDM.xml").read_bytes() == b"<ASDM/>"
    assert not (stale / "LeftoverFromOldExtraction.xml").exists()
    assert _no_staging_left(dest)


@pytest.mark.unit
def test_truncated_tar_publishes_nothing_and_keeps_archive(tmp_path):
    good = _asdm_tar(tmp_path / "good.tar", n_bdf=1)
    with tarfile.open(good) as tar:
        last = tar.getmembers()[-1]
    data = good.read_bytes()
    bad = tmp_path / "bad.tar"
    # Cut inside the last member's data. A cut on a 512-byte block boundary
    # would look like a clean end-of-archive to tarfile; that case is the
    # download stage's job (it checks the byte count), not the extractor's.
    bad.write_bytes(data[: last.offset_data + last.size // 2])
    dest = tmp_path / "dest"
    (dest / PROJECT).mkdir(parents=True)

    with pytest.raises(tarfile.TarError):
        extract_tar_atomically(bad, dest)

    assert not (dest / ASDM).exists()
    assert not (dest / PROJECT / ASDM.split("/", 1)[1].split("/")[0]).exists()
    assert _no_staging_left(dest)
    assert bad.is_file()


@pytest.mark.unit
def test_failure_midway_never_exposes_partial_asdm(tmp_path, monkeypatch):
    """The exact race behind the production failure: fail after some members."""
    dest = tmp_path / "dest"
    calls = {"n": 0}
    real_extract = tarfile.TarFile.extract

    def flaky_extract(self, member, path="", *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            # Before failing, prove the partial tree exists only under staging.
            visible = list(Path(dest).rglob("*.asdm.sdm"))
            assert visible, "partial ASDM should exist in staging"
            assert all(
                any(is_extraction_tmp_dir(part) for part in p.relative_to(dest).parts)
                for p in visible
            ), "partial ASDM leaked outside the staging directory"
            raise OSError("disk full")
        return real_extract(self, member, path, *args, **kwargs)

    monkeypatch.setattr(tarfile.TarFile, "extract", flaky_extract)

    with pytest.raises(OSError, match="disk full"):
        extract_tar_atomically(_asdm_tar(tmp_path / "a.tar"), dest)

    assert not list(dest.rglob("*.asdm.sdm"))
    assert _no_staging_left(dest)


@pytest.mark.unit
def test_flat_archive_goes_under_stem_when_requested(tmp_path):
    flat = tmp_path / "bundle.tar"
    with tarfile.open(flat, "w") as tar:
        _add(tar, "a.txt", b"a")
        _add(tar, "b/c.txt", b"c")

    dest = tmp_path / "dest"
    files = extract_tar_atomically(flat, dest, flat_archive_subdir=True)
    assert (dest / "bundle" / "a.txt").is_file()
    assert (dest / "bundle" / "b" / "c.txt").is_file()
    assert sorted(files) == [dest / "bundle" / "a.txt", dest / "bundle" / "b" / "c.txt"]

    dest2 = tmp_path / "dest2"
    extract_tar_atomically(flat, dest2, flat_archive_subdir=False)
    assert (dest2 / "a.txt").is_file()
    assert (dest2 / "b" / "c.txt").is_file()
    assert _no_staging_left(dest) and _no_staging_left(dest2)


@pytest.mark.unit
def test_single_top_level_archive_is_not_nested_again(tmp_path):
    nested = tmp_path / "nested.tar"
    with tarfile.open(nested, "w") as tar:
        _add(tar, "top/a.txt", b"a")
    dest = tmp_path / "dest"
    extract_tar_atomically(nested, dest, flat_archive_subdir=True)
    assert (dest / "top" / "a.txt").is_file()
    assert not (dest / "nested").exists()


@pytest.mark.unit
def test_unsafe_members_are_skipped_and_reported(tmp_path):
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tar:
        _add(tar, "top/ok.txt", b"ok")
        _add(tar, "top/../escape.txt", b"no")
        _add(tar, "/abs.txt", b"no")

    skipped: list[str] = []
    dest = tmp_path / "dest"
    files = extract_tar_atomically(evil, dest, on_skipped_member=skipped.append)

    assert files == [dest / "top" / "ok.txt"]
    assert sorted(skipped) == sorted(["top/../escape.txt", "/abs.txt"])
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / "escape.txt").exists()
    assert not (dest / "abs.txt").exists()


@pytest.mark.unit
def test_two_archives_into_the_same_member_directory(tmp_path):
    """Concurrent download jobs land ASDMs of the same member; both survive."""
    dest = tmp_path / "dest"
    extract_tar_atomically(_asdm_tar(tmp_path / "a.tar", uid="uid___A002_X1_Xa"), dest)
    extract_tar_atomically(_asdm_tar(tmp_path / "b.tar", uid="uid___A002_X1_Xb"), dest)
    raw = dest / MEMBER
    assert sorted(p.name for p in raw.iterdir()) == [
        "uid___A002_X1_Xa.asdm.sdm",
        "uid___A002_X1_Xb.asdm.sdm",
    ]
    assert _no_staging_left(dest)
