"""Unit tests for the download API router."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Stub heavy database/service imports so the router can load without a live DB.
_db_stub = MagicMock()
for _mod in ("database", "database.config", "database.models", "database.service"):
    sys.modules.setdefault(_mod, _db_stub)

_ROUTER_PATH = REPO_ROOT / "backend" / "app" / "api" / "v1" / "routers" / "download.py"
_SPEC = importlib.util.spec_from_file_location("download_router_test_module", _ROUTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
download_router = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download_router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(job_id="job-1", status="completed", files=None, total_bytes=1000):
    from app.services.download_service import DownloadJob

    job = DownloadJob(
        job_id=job_id,
        destination="/tmp/downloads",
        member_ous_uids=["uid://A001/X1/X1"],
        total_files=1,
        total_bytes=total_bytes,
        status=status,
        files_completed=1,
        files_failed=0,
        bytes_downloaded=total_bytes,
    )
    if files:
        job.files = files
    return job


def _make_file_status(filename="a.fits", length=500, downloaded=500):
    from app.services.download_service import FileDownloadStatus

    return FileDownloadStatus(
        access_url="https://example.org/a.fits",
        filename=filename,
        content_length=length,
        bytes_downloaded=downloaded,
        status="completed",
    )


def _make_db_rec(job_id="job-1", status="completed"):
    rec = MagicMock()
    rec.job_id = job_id
    rec.status = status
    rec.destination = "/tmp/downloads"
    rec.total_files = 1
    rec.files_completed = 1
    rec.files_failed = 0
    rec.total_bytes = 1000
    rec.bytes_downloaded = 1000
    rec.created_at = datetime(2024, 1, 1)
    rec.member_ous_uids = json.dumps(["uid://A001/X1/X1"])
    rec.product_filter = "all"
    rec.error = None
    rec.manifest_path = None
    rec.raw_measurement_sets = json.dumps([])
    rec.calibrated_measurement_sets = json.dumps([])
    rec.metadata_json = json.dumps([])
    rec.unpack_ms = False
    rec.generate_calibrated_visibilities = False
    rec.clean_intermediate_files = False
    rec.archive_output_root = None
    rec.casa_data_root = None
    rec.skip_casa_data_update = False
    rec.files = []
    return rec


# ---------------------------------------------------------------------------
# Helpers: _json_list / _metadata_stats
# ---------------------------------------------------------------------------


def test_json_list_returns_list_from_valid_json():
    result = download_router._json_list(json.dumps(["a", "b"]))
    assert result == ["a", "b"]


def test_json_list_returns_empty_on_invalid_json():
    assert download_router._json_list("not-json") == []


def test_json_list_returns_empty_on_non_list():
    assert download_router._json_list(json.dumps({"key": "val"})) == []


def test_json_list_returns_empty_on_none():
    assert download_router._json_list(None) == []


def test_metadata_stats_with_rows():
    has_meta, count = download_router._metadata_stats(json.dumps([{"uid": "x"}]))
    assert has_meta is True
    assert count == 1


def test_metadata_stats_empty():
    has_meta, count = download_router._metadata_stats(json.dumps([]))
    assert has_meta is False
    assert count == 0


# ---------------------------------------------------------------------------
# GET /browse
# ---------------------------------------------------------------------------


def test_browse_directory_returns_subdirs(tmp_path, monkeypatch):
    subdir = tmp_path / "data"
    subdir.mkdir()

    monkeypatch.setattr(download_router.settings, "HOST_DIR", tmp_path)

    result = asyncio.run(download_router.browse_directory(path=str(tmp_path / "data")))
    assert result.current == str(tmp_path / "data")


def test_browse_directory_rejects_traversal(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(download_router.settings, "HOST_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.browse_directory(path="../../etc"))
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# POST /mkdir
# ---------------------------------------------------------------------------


def test_make_directory_creates_and_returns_result(tmp_path, monkeypatch):
    new_dir = tmp_path / "new_folder"
    monkeypatch.setattr(download_router.settings, "HOST_DIR", tmp_path)

    result = asyncio.run(download_router.make_directory(path=str(new_dir)))
    assert new_dir.exists()
    assert result.current == str(new_dir)


def test_make_directory_rejects_traversal(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(download_router.settings, "HOST_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.make_directory(path="../../etc/bad"))
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# POST /resolve
# ---------------------------------------------------------------------------


def test_resolve_download_products_returns_product_list(monkeypatch):
    from app.services.download_service import DataProduct

    fake_products = [
        DataProduct(
            access_url="https://a.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=1024,
            content_type="application/fits",
            product_type="fits",
        )
    ]
    monkeypatch.setattr(download_router, "resolve_products", lambda uids: fake_products)

    body = MagicMock()
    body.member_ous_uids = ["uid://A001/X1"]
    result = asyncio.run(download_router.resolve_download_products(body))

    assert result.total_count == 1
    assert result.total_size_bytes == 1024


def test_resolve_download_products_raises_400_for_empty_uids(monkeypatch):
    from fastapi import HTTPException

    body = MagicMock()
    body.member_ous_uids = []

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.resolve_download_products(body))
    assert exc_info.value.status_code == 400


def test_resolve_download_products_raises_502_on_network_error(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        download_router, "resolve_products", MagicMock(side_effect=IOError("timeout"))
    )
    body = MagicMock()
    body.member_ous_uids = ["uid://A001/X1"]

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.resolve_download_products(body))
    assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# POST /disk-space
# ---------------------------------------------------------------------------


def test_check_disk_space_returns_info(tmp_path, monkeypatch):
    monkeypatch.setattr(download_router.settings, "HOST_DIR", tmp_path)

    body = MagicMock()
    body.path = str(tmp_path)
    body.needed_bytes = 1

    result = asyncio.run(download_router.check_disk_space(body))
    assert result.total_bytes > 0


def test_check_disk_space_rejects_traversal(tmp_path, monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(download_router.settings, "HOST_DIR", tmp_path)

    body = MagicMock()
    body.path = "../../etc"
    body.needed_bytes = 0

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.check_disk_space(body))
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# POST /start
# ---------------------------------------------------------------------------


def test_start_download_raises_400_for_empty_uids(monkeypatch):
    from fastapi import HTTPException

    body = MagicMock()
    body.member_ous_uids = []

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.start_download(body=body, background_tasks=MagicMock()))
    assert exc_info.value.status_code == 400


def test_start_download_raises_502_on_resolve_error(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(download_router, "resolve_products", MagicMock(side_effect=IOError("net")))
    body = MagicMock()
    body.member_ous_uids = ["uid://A001/X1"]

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.start_download(body=body, background_tasks=MagicMock()))
    assert exc_info.value.status_code == 502


def test_start_download_raises_404_when_no_products_match_filter(monkeypatch):
    from app.services.download_service import DataProduct
    from fastapi import HTTPException

    fake_products = [
        DataProduct(
            access_url="https://a.org/a.fits",
            uid="uid://A001/X1",
            filename="a.fits",
            content_length=1024,
            content_type="application/fits",
            product_type="fits",
        )
    ]
    monkeypatch.setattr(download_router, "resolve_products", lambda uids: fake_products)
    monkeypatch.setattr(download_router, "filter_products", lambda prods, filt: [])

    body = MagicMock()
    body.member_ous_uids = ["uid://A001/X1"]
    body.product_filter = "raw"

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.start_download(body=body, background_tasks=MagicMock()))
    assert exc_info.value.status_code == 404


def test_start_download_creates_job_and_returns_response(monkeypatch):
    from app.services.download_service import DataProduct

    fake_product = DataProduct(
        access_url="https://a.org/a.fits",
        uid="uid://A001/X1",
        filename="a.fits",
        content_length=2048,
        content_type="application/fits",
        product_type="fits",
    )
    monkeypatch.setattr(download_router, "resolve_products", lambda uids: [fake_product])
    monkeypatch.setattr(download_router, "filter_products", lambda prods, filt: prods)

    store = MagicMock()
    monkeypatch.setattr(download_router, "download_store", store)

    bt = MagicMock()
    body = MagicMock()
    body.member_ous_uids = ["uid://A001/X1"]
    body.product_filter = "all"
    body.destination = "/tmp/dl"
    body.selected_metadata = []
    body.max_parallel = 3
    body.extract_tar = False
    body.unpack_ms = False
    body.generate_calibrated_visibilities = False
    body.clean_intermediate_files = False
    body.archive_output_root = None
    body.casa_data_root = None
    body.skip_casa_data_update = False

    result = asyncio.run(download_router.start_download(body=body, background_tasks=bt))
    assert result.total_files == 1
    assert result.total_bytes == 2048
    assert result.status == "pending"
    store.create.assert_called_once()
    bt.add_task.assert_called_once()


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------


def test_list_download_jobs_with_active_job(monkeypatch):
    job = _make_job()

    db_rec = MagicMock()
    db_rec.job_id = "job-1"
    db_rec.__str__ = lambda self: "job-1"

    store = MagicMock()
    store.list_all.return_value = [db_rec]
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.list_download_jobs())
    assert len(result) == 1
    assert result[0].job_id == "job-1"
    assert result[0].status == "completed"


def test_list_download_jobs_from_db_when_not_active(monkeypatch):
    rec = _make_db_rec()

    store = MagicMock()
    store.list_all.return_value = [rec]
    store.get.return_value = None
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.list_download_jobs())
    assert len(result) == 1
    assert result[0].status == "completed"


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------


def test_get_download_job_returns_active_job(monkeypatch):
    job = _make_job(files=[_make_file_status()])

    store = MagicMock()
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.get_download_job("job-1"))
    assert result.job_id == "job-1"
    assert len(result.files) == 1


def test_get_download_job_falls_back_to_db(monkeypatch):
    rec = _make_db_rec()

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = rec
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.get_download_job("job-1"))
    assert result.job_id == "job-1"
    assert result.status == "completed"


def test_get_download_job_raises_404_when_not_found(monkeypatch):
    from fastapi import HTTPException

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = None
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.get_download_job("no-such"))
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/metadata
# ---------------------------------------------------------------------------


def test_get_download_job_metadata_from_active_job(monkeypatch):
    job = _make_job()
    job.metadata_rows = [{"uid": "a"}]

    store = MagicMock()
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.get_download_job_metadata("job-1"))
    assert result.count == 1


def test_get_download_job_metadata_from_db(monkeypatch):
    rec = _make_db_rec()
    rec.metadata_json = json.dumps([{"uid": "x"}, {"uid": "y"}])

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = rec
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.get_download_job_metadata("job-1"))
    assert result.count == 2


def test_get_download_job_metadata_raises_404_when_job_missing(monkeypatch):
    from fastapi import HTTPException

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = None
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.get_download_job_metadata("no-such"))
    assert exc_info.value.status_code == 404


def test_get_download_job_metadata_raises_404_when_rows_empty(monkeypatch):
    from fastapi import HTTPException

    job = _make_job()
    job.metadata_rows = []

    store = MagicMock()
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.get_download_job_metadata("job-1"))
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/cancel
# ---------------------------------------------------------------------------


def test_cancel_download_job_active_job(monkeypatch):
    job = _make_job(status="running")

    store = MagicMock()
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.cancel_download_job("job-1"))
    assert result["status"] == "cancelled"
    store.update.assert_called_once_with("job-1", status="cancelled")
    store.persist.assert_called_once_with("job-1")


def test_cancel_download_job_raises_400_when_already_finished(monkeypatch):
    from fastapi import HTTPException

    job = _make_job(status="completed")

    store = MagicMock()
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.cancel_download_job("job-1"))
    assert exc_info.value.status_code == 400


def test_cancel_download_job_falls_back_to_db(monkeypatch):
    rec = _make_db_rec(status="running")

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = rec
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.cancel_download_job("job-1"))
    assert result["status"] == "cancelled"
    store.update_in_db.assert_called_once_with("job-1", status="cancelled")


def test_cancel_download_job_raises_404_when_not_found(monkeypatch):
    from fastapi import HTTPException

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = None
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.cancel_download_job("no-such"))
    assert exc_info.value.status_code == 404


def test_cancel_download_job_db_record_already_finished_raises_400(monkeypatch):
    from fastapi import HTTPException

    rec = _make_db_rec(status="completed")

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = rec
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.cancel_download_job("job-1"))
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /jobs/{job_id}
# ---------------------------------------------------------------------------


def test_delete_download_job_removes_record(tmp_path, monkeypatch):
    rec = _make_db_rec()
    rec.destination = str(tmp_path)
    rec.manifest_path = None
    rec.raw_measurement_sets = json.dumps([])
    rec.calibrated_measurement_sets = json.dumps([])

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = rec
    store.delete_from_db.return_value = True
    store._lock = __import__("threading").Lock()
    store._active = {}
    monkeypatch.setattr(download_router, "download_store", store)

    result = asyncio.run(download_router.delete_download_job("job-1"))
    assert result["deleted"] is True
    store.delete_from_db.assert_called_once_with("job-1")


def test_delete_download_job_raises_400_when_active(monkeypatch):
    from fastapi import HTTPException

    job = _make_job(status="running")

    store = MagicMock()
    store.get.return_value = job
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.delete_download_job("job-1"))
    assert exc_info.value.status_code == 400


def test_delete_download_job_raises_404_when_not_found(monkeypatch):
    from fastapi import HTTPException

    store = MagicMock()
    store.get.return_value = None
    store.get_from_db.return_value = None
    monkeypatch.setattr(download_router, "download_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(download_router.delete_download_job("no-such"))
    assert exc_info.value.status_code == 404
