"""Unit tests for the library-first download service."""

from almasim.services.download import (
    DataProduct,
    download_products,
    filter_products,
    load_products_csv,
    save_products_csv,
)


def test_filter_products():
    """Filtering should preserve only matching product types."""
    products = [
        DataProduct(
            access_url="https://example.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=10,
            content_type="application/fits",
            product_type="fits",
        ),
        DataProduct(
            access_url="https://example.org/b.tar",
            uid="uid://A001/X1",
            filename="b.tar",
            content_length=20,
            content_type="application/x-tar",
            product_type="raw",
        ),
    ]

    filtered = filter_products(products, "fits")

    assert len(filtered) == 1
    assert filtered[0].filename == "a.fits"


def test_products_csv_roundtrip(tmp_path):
    """Resolved products should round-trip through CSV."""
    products = [
        DataProduct(
            access_url="https://example.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=10,
            content_type="application/fits",
            product_type="fits",
            semantics="alma#imagecube",
        )
    ]
    csv_path = tmp_path / "products.csv"

    save_products_csv(products, csv_path)
    loaded = load_products_csv(csv_path)

    assert len(loaded) == 1
    assert loaded[0].access_url == products[0].access_url
    assert loaded[0].uid == products[0].uid
    assert loaded[0].filename == products[0].filename
    assert loaded[0].product_type == products[0].product_type
    assert loaded[0].semantics == products[0].semantics


def test_download_products_skips_existing_file(tmp_path):
    """Existing files should be treated as completed without network access."""
    destination = tmp_path / "downloads"
    destination.mkdir()
    file_path = destination / "existing.txt"
    file_path.write_bytes(b"already here")

    products = [
        DataProduct(
            access_url="https://example.org/existing.txt",
            uid="uid://A001/X1",
            filename="existing.txt",
            content_length=file_path.stat().st_size,
            content_type="text/plain",
            product_type="auxiliary",
        )
    ]

    summary = download_products(products, destination)

    assert summary.total_files == 1
    assert summary.files_completed == 1
    assert summary.files_failed == 0
    assert summary.files[0].status == "completed"
    assert summary.files[0].bytes_downloaded == file_path.stat().st_size


# ---------------------------------------------------------------------------
# Truncated / short downloads must never be reported as completed
# ---------------------------------------------------------------------------

import io  # noqa: E402
import tarfile  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import pytest  # noqa: E402

from almasim.services import download as download_module  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, body: bytes, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size: int):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class _FakeClient:
    """Plays back a script of ``headers -> (status, body)`` per GET."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @contextmanager
    def stream(self, method, url, headers=None):
        self.requests.append(dict(headers or {}))
        status, body, *response_headers = self.script.pop(0)(headers or {})
        yield _FakeResponse(status, body, *response_headers)


def _install_fake_client(monkeypatch, script) -> _FakeClient:
    fake = _FakeClient(script)
    monkeypatch.setattr(download_module.httpx, "Client", lambda **kwargs: fake)
    monkeypatch.setattr(download_module.time, "sleep", lambda *_: None)
    return fake


def _asdm_tar_bytes(uid: str = "uid___A002_X1_X1") -> bytes:
    asdm = (
        f"2023.1.00001.S/science_goal.uid___A001_X1_X1/member.uid___A001_X1_X3/raw/{uid}.asdm.sdm"  # noqa: E501
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in (("ASDM.xml", b"<ASDM/>"), ("Main.xml", b"<Main/>")):
            info = tarfile.TarInfo(f"{asdm}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _product(filename: str, body: bytes) -> DataProduct:
    return DataProduct(
        access_url=f"https://example.org/{filename}",
        uid="uid://A001/X1",
        filename=filename,
        content_length=len(body),
        content_type="application/x-tar",
        product_type="raw",
    )


@pytest.mark.unit
def test_short_download_is_failed_not_completed(tmp_path, monkeypatch):
    """The server closes early; without a size check this became a truncated tar."""
    body = _asdm_tar_bytes()
    half = body[: len(body) // 2]
    monkeypatch.setattr(download_module, "_MAX_FILE_ATTEMPTS", 1)
    _install_fake_client(monkeypatch, [lambda h: (200, half)])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert summary.files_failed == 1
    assert summary.files_completed == 0
    assert "Short download" in (summary.files[0].error or "")
    assert not (tmp_path / "a.asdm.sdm.tar").exists()
    assert not list(tmp_path.rglob("*.asdm.sdm"))
    assert summary.extraction_failed == []


@pytest.mark.unit
def test_maintenance_page_is_reported_and_logged(tmp_path, monkeypatch):
    """The EA portal answered 200 with an HTML notice during its 2026-09-29 maintenance."""
    body = _asdm_tar_bytes()
    page = b"<H1>ALMA Science Portal Maintenance</H1>\nPortal is temporarily unavailable."
    monkeypatch.setattr(download_module, "_MAX_FILE_ATTEMPTS", 1)
    _install_fake_client(
        monkeypatch, [lambda h: (200, page, {"content-type": "text/html; charset=utf-8"})]
    )
    messages: list[str] = []

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)],
        tmp_path,
        rate_limit_sec=0,
        extract_tar=True,
        logger_fn=messages.append,
    )

    assert summary.files_failed == 1
    error = summary.files[0].error or ""
    assert "HTML page" in error and "ALMA Science Portal Maintenance" in error
    assert not (tmp_path / "a.asdm.sdm.tar.part").exists(), "the page is not saved as data"
    failed = [m for m in messages if m.startswith("Download failed: a.asdm.sdm.tar")]
    assert len(failed) == 1
    assert " from almascience." in failed[0] and "Maintenance" in failed[0]


@pytest.mark.unit
def test_short_download_restarts_when_server_rejects_range(tmp_path, monkeypatch):
    """Attempt 1 is short; the resume gets a 416 (the ALMA portal); a fresh GET completes."""
    body = _asdm_tar_bytes()
    half = body[: len(body) // 2]

    def first(headers):
        assert "Range" not in headers
        return 200, half

    def resume(headers):
        assert headers.get("Range") == f"bytes={len(half)}-"
        return 416, b""

    def fresh(headers):
        assert "Range" not in headers
        return 200, body

    fake = _install_fake_client(monkeypatch, [first, resume, fresh])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert len(fake.requests) == 3
    assert summary.files_completed == 1
    assert summary.files[0].bytes_downloaded == len(body)
    asdms = list(tmp_path.rglob("*.asdm.sdm"))
    assert len(asdms) == 1 and (asdms[0] / "Main.xml").is_file()
    assert not (tmp_path / "a.asdm.sdm.tar").exists(), "tar is deleted after extraction"
    assert summary.extraction_failed == []


@pytest.mark.unit
def test_unextractable_tar_is_reported_and_kept(tmp_path, monkeypatch):
    """A full-size but corrupt tar: download completes, extraction failure is loud."""
    body = _asdm_tar_bytes()
    garbage = b"\xab" * len(body)  # right length, not a tar (all-zero *would* be one)
    _install_fake_client(monkeypatch, [lambda h: (200, garbage)])
    messages: list[str] = []

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)],
        tmp_path,
        rate_limit_sec=0,
        extract_tar=True,
        logger_fn=messages.append,
    )

    assert summary.files_completed == 1
    assert summary.extraction_failed == [str(tmp_path / "a.asdm.sdm.tar")]
    assert (tmp_path / "a.asdm.sdm.tar").is_file(), "kept for a manual retry"
    assert not list(tmp_path.rglob("*.asdm.sdm"))
    assert not any(p.name.startswith(".extracting-") for p in tmp_path.iterdir())
    assert any("Failed to extract a.asdm.sdm.tar" in m for m in messages)
    import json

    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["extraction_failed"] == summary.extraction_failed


_MAINTENANCE_PAGE = b"<H1>ALMA Science Portal Maintenance</H1>\nWe regret the inconvenience.\n"


@pytest.mark.unit
def test_html_part_is_discarded_not_resumed(tmp_path, monkeypatch):
    """A .part holding a maintenance page (job 4029, 2026-09-29) must not be resumed."""
    body = _asdm_tar_bytes()
    (tmp_path / "a.asdm.sdm.tar.part").write_bytes(_MAINTENANCE_PAGE)

    def fresh(headers):
        assert "Range" not in headers, "resuming would append the tar after the page"
        return 200, body

    fake = _install_fake_client(monkeypatch, [fresh])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert len(fake.requests) == 1
    assert summary.files_completed == 1
    assert summary.extraction_failed == []
    assert len(list(tmp_path.rglob("*.asdm.sdm"))) == 1


@pytest.mark.unit
def test_right_sized_tar_without_header_is_downloaded_again(tmp_path, monkeypatch):
    """The page overwrote the first bytes of a full-size tar: size alone passes it."""
    body = _asdm_tar_bytes()
    corrupt = _MAINTENANCE_PAGE + body[len(_MAINTENANCE_PAGE) :]
    (tmp_path / "a.asdm.sdm.tar").write_bytes(corrupt)
    fake = _install_fake_client(monkeypatch, [lambda h: (200, body)])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert len(fake.requests) == 1
    assert summary.files_completed == 1
    assert summary.extraction_failed == []
    assert len(list(tmp_path.rglob("*.asdm.sdm"))) == 1


@pytest.mark.unit
def test_valid_existing_tar_is_not_downloaded_again(tmp_path, monkeypatch):
    body = _asdm_tar_bytes()
    (tmp_path / "a.asdm.sdm.tar").write_bytes(body)
    fake = _install_fake_client(monkeypatch, [])

    summary = download_products(
        [_product("a.asdm.sdm.tar", body)], tmp_path, rate_limit_sec=0, extract_tar=True
    )

    assert fake.requests == []
    assert summary.files_completed == 1
    assert len(list(tmp_path.rglob("*.asdm.sdm"))) == 1
