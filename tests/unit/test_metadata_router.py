"""Unit tests for the metadata API router."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Pre-inject stubs for heavy dependencies so the router can be loaded without
# a live PostgreSQL connection or SQLAlchemy installed.
_db_stub = MagicMock()
for _mod in (
    "sqlalchemy",
    "sqlalchemy.orm",
    "database",
    "database.config",
    "database.models",
    "database.service",
):
    sys.modules.setdefault(_mod, _db_stub)

_ROUTER_PATH = REPO_ROOT / "backend" / "app" / "api" / "v1" / "routers" / "metadata.py"
_SPEC = importlib.util.spec_from_file_location("metadata_router_test_module", _ROUTER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
metadata_router = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(metadata_router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db():
    return MagicMock()


def _make_background_tasks():
    bt = MagicMock()
    bt.add_task = MagicMock()
    return bt


# ---------------------------------------------------------------------------
# GET /science-types
# ---------------------------------------------------------------------------


def test_get_science_types_returns_keywords_and_categories(monkeypatch):
    svc = MagicMock()
    svc.get_science_types.return_value = (["Galaxies"], ["Science"])
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    result = asyncio.run(metadata_router.get_science_types(db=_make_db()))
    assert result == {"keywords": ["Galaxies"], "categories": ["Science"]}


def test_get_science_types_raises_500_on_error(monkeypatch):
    from fastapi import HTTPException

    svc = MagicMock()
    svc.get_science_types.side_effect = RuntimeError("db down")
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.get_science_types(db=_make_db()))
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------


def test_query_metadata_returns_response(monkeypatch):
    svc = MagicMock()
    svc.query_by_science.return_value = [{"uid": "a"}, {"uid": "b"}]
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    query = MagicMock()
    query.to_params.return_value = {}
    result = asyncio.run(metadata_router.query_metadata(query=query, db=_make_db()))

    assert result.count == 2
    assert len(result.data) == 2


def test_query_metadata_raises_500_on_error(monkeypatch):
    from fastapi import HTTPException

    svc = MagicMock()
    svc.query_by_science.side_effect = RuntimeError("tap failed")
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    query = MagicMock()
    query.to_params.return_value = {}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.query_metadata(query=query, db=_make_db()))
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# POST /query/start
# ---------------------------------------------------------------------------


def test_start_query_creates_job_and_schedules_background_task(monkeypatch):
    svc = MagicMock()
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    store = MagicMock()
    monkeypatch.setattr(metadata_router, "query_store", store)

    bt = _make_background_tasks()
    query = MagicMock()
    query.to_params.return_value = {}

    result = asyncio.run(
        metadata_router.start_query(query=query, background_tasks=bt, db=_make_db())
    )

    assert result.status == "running"
    assert result.query_id
    store.create.assert_called_once()
    bt.add_task.assert_called_once()


def test_start_query_raises_500_on_error(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        metadata_router, "MetadataService", lambda db: (_ for _ in ()).throw(RuntimeError("boom"))
    )  # type: ignore[arg-type]

    store = MagicMock()
    store.create.side_effect = RuntimeError("store error")
    monkeypatch.setattr(metadata_router, "query_store", store)

    query = MagicMock()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            metadata_router.start_query(
                query=query, background_tasks=_make_background_tasks(), db=_make_db()
            )
        )
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# GET /query/{query_id}/results
# ---------------------------------------------------------------------------


def test_get_query_results_returns_page(monkeypatch):
    store = MagicMock()
    store.get.return_value = MagicMock()
    store.get_page.return_value = {
        "query_id": "qid-1",
        "page": 0,
        "rows": [{"uid": "x"}],
        "page_size": 500,
        "total_fetched": 1,
        "done": True,
        "status": "completed",
        "error": None,
    }
    monkeypatch.setattr(metadata_router, "query_store", store)

    result = asyncio.run(metadata_router.get_query_results("qid-1"))
    assert result.query_id == "qid-1"
    assert result.done is True


def test_get_query_results_raises_404_when_job_missing(monkeypatch):
    from fastapi import HTTPException

    store = MagicMock()
    store.get.return_value = None
    monkeypatch.setattr(metadata_router, "query_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.get_query_results("no-such-id"))
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /query/{query_id}
# ---------------------------------------------------------------------------


def test_cancel_query_succeeds(monkeypatch):
    store = MagicMock()
    store.cancel.return_value = True
    monkeypatch.setattr(metadata_router, "query_store", store)

    asyncio.run(metadata_router.cancel_query("qid-1"))
    store.cancel.assert_called_once_with("qid-1")


def test_cancel_query_raises_404_when_job_missing(monkeypatch):
    from fastapi import HTTPException

    store = MagicMock()
    store.cancel.return_value = False
    monkeypatch.setattr(metadata_router, "query_store", store)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.cancel_query("no-such-id"))
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /load/{file_path}
# ---------------------------------------------------------------------------


def test_load_metadata_resolves_from_data_dir(monkeypatch, tmp_path):
    safe_path = tmp_path / "data.csv"
    safe_path.touch()

    monkeypatch.setattr(metadata_router.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(metadata_router.settings, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(metadata_router, "resolve_safe_path", lambda path, base, **kw: safe_path)

    svc = MagicMock()
    svc.load_metadata.return_value = [{"uid": "a"}]
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    result = asyncio.run(metadata_router.load_metadata("data.csv", db=_make_db()))
    assert result.count == 1


def test_load_metadata_falls_back_to_output_dir(monkeypatch, tmp_path):
    from fastapi import HTTPException as FastAPIHTTPException

    output_path = tmp_path / "outputs" / "data.csv"
    output_path.parent.mkdir()
    output_path.touch()

    call_count = {"n": 0}

    def fake_resolve(path, base, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise FastAPIHTTPException(status_code=400, detail="outside")
        return output_path

    monkeypatch.setattr(metadata_router, "resolve_safe_path", fake_resolve)
    monkeypatch.setattr(metadata_router.settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(metadata_router.settings, "OUTPUT_DIR", tmp_path / "outputs")

    svc = MagicMock()
    svc.load_metadata.return_value = []
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    result = asyncio.run(metadata_router.load_metadata("data.csv", db=_make_db()))
    assert result.count == 0
    assert call_count["n"] == 2


def test_load_metadata_raises_500_on_service_error(monkeypatch, tmp_path):
    from fastapi import HTTPException

    monkeypatch.setattr(
        metadata_router, "resolve_safe_path", lambda path, base, **kw: tmp_path / path
    )
    monkeypatch.setattr(metadata_router.settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(metadata_router.settings, "OUTPUT_DIR", tmp_path / "outputs")

    svc = MagicMock()
    svc.load_metadata.side_effect = RuntimeError("parse error")
    monkeypatch.setattr(metadata_router, "MetadataService", lambda db: svc)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.load_metadata("bad.csv", db=_make_db()))
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# POST /save
# ---------------------------------------------------------------------------


def test_save_metadata_success(monkeypatch, tmp_path):
    dest = tmp_path / "query_results" / "out.json"
    dest.parent.mkdir()

    monkeypatch.setattr(metadata_router, "normalize_metadata_format", lambda fmt: "json")
    monkeypatch.setattr(metadata_router, "resolve_metadata_output_path", lambda *a, **kw: dest)
    monkeypatch.setattr(metadata_router, "save_metadata_records", MagicMock())
    monkeypatch.setattr(metadata_router.settings, "OUTPUT_DIR", tmp_path)

    payload = MagicMock()
    payload.format = "json"
    payload.path = "out.json"
    payload.data = [{"uid": "a"}, {"uid": "b"}]

    result = asyncio.run(metadata_router.save_metadata(payload))
    assert result.count == 2
    assert "successfully" in result.message


def test_save_metadata_raises_400_on_unsupported_format(monkeypatch):
    from fastapi import HTTPException

    from almasim.services.metadata.storage import UnsupportedMetadataFormatError

    monkeypatch.setattr(
        metadata_router,
        "normalize_metadata_format",
        lambda fmt: (_ for _ in ()).throw(UnsupportedMetadataFormatError("bad")),  # type: ignore[arg-type]
    )

    payload = MagicMock()
    payload.format = "xml"
    payload.path = "out.xml"
    payload.data = []

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.save_metadata(payload))
    assert exc_info.value.status_code == 400


def test_save_metadata_raises_400_on_invalid_path(monkeypatch):
    from fastapi import HTTPException

    from almasim.services.metadata.storage import InvalidMetadataPathError

    monkeypatch.setattr(metadata_router, "normalize_metadata_format", lambda fmt: "json")
    monkeypatch.setattr(
        metadata_router,
        "resolve_metadata_output_path",
        lambda *a, **kw: (_ for _ in ()).throw(InvalidMetadataPathError("bad path")),  # type: ignore[arg-type]
    )

    payload = MagicMock()
    payload.format = "json"
    payload.path = "../../etc/passwd"
    payload.data = []

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.save_metadata(payload))
    assert exc_info.value.status_code == 400


def test_save_metadata_raises_500_on_generic_error(monkeypatch, tmp_path):
    from fastapi import HTTPException

    monkeypatch.setattr(metadata_router, "normalize_metadata_format", lambda fmt: "json")
    monkeypatch.setattr(
        metadata_router,
        "resolve_metadata_output_path",
        lambda *a, **kw: tmp_path / "out.json",
    )
    monkeypatch.setattr(
        metadata_router,
        "save_metadata_records",
        MagicMock(side_effect=OSError("disk full")),
    )
    monkeypatch.setattr(metadata_router.settings, "OUTPUT_DIR", tmp_path)

    payload = MagicMock()
    payload.format = "json"
    payload.path = "out.json"
    payload.data = [{}]

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.save_metadata(payload))
    assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Preset endpoints
# ---------------------------------------------------------------------------


def _make_preset_row(name="my-preset"):
    from datetime import datetime

    row = MagicMock()
    row.query_name = name
    row.description = "desc"
    row.query_params = {"filters": {"band": [6]}}
    row.result_count = 42
    row.created_at = datetime(2024, 1, 1, 12, 0, 0)
    return row


def test_list_query_presets_returns_list(monkeypatch):
    svc = MagicMock()
    svc.list_query_results.return_value = [_make_preset_row("p1"), _make_preset_row("p2")]
    monkeypatch.setattr(metadata_router, "DatabaseService", lambda db: svc)

    result = asyncio.run(metadata_router.list_query_presets(db=_make_db()))
    assert len(result.presets) == 2
    assert result.presets[0].name == "p1"


def test_list_query_presets_raises_500_on_error(monkeypatch):
    from fastapi import HTTPException

    svc = MagicMock()
    svc.list_query_results.side_effect = RuntimeError("db error")
    monkeypatch.setattr(metadata_router, "DatabaseService", lambda db: svc)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.list_query_presets(db=_make_db()))
    assert exc_info.value.status_code == 500


def test_save_query_preset_creates_row(monkeypatch):
    db = _make_db()
    db.add = MagicMock()
    db.commit = MagicMock()
    db.refresh = MagicMock(
        side_effect=lambda r: setattr(r, "created_at", __import__("datetime").datetime(2024, 1, 1))
    )

    # Patch QueryResult so the router can instantiate it
    mock_qr = MagicMock(return_value=_make_preset_row("saved"))
    monkeypatch.setattr(metadata_router, "QueryResult", mock_qr)

    payload = MagicMock()
    payload.name = "saved"
    payload.description = "a preset"
    payload.filters = {"band": [6]}
    payload.result_count = 5

    result = asyncio.run(metadata_router.save_query_preset(payload=payload, db=db))
    assert result.name == "saved"


def test_save_query_preset_raises_500_on_error(monkeypatch):
    from fastapi import HTTPException

    db = _make_db()
    db.add.side_effect = RuntimeError("insert failed")

    mock_qr = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(metadata_router, "QueryResult", mock_qr)

    payload = MagicMock()
    payload.name = "bad"
    payload.description = ""
    payload.filters = {}
    payload.result_count = 0

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.save_query_preset(payload=payload, db=db))
    assert exc_info.value.status_code == 500


def test_get_query_preset_found(monkeypatch):
    svc = MagicMock()
    svc.get_query_result.return_value = _make_preset_row("p1")
    monkeypatch.setattr(metadata_router, "DatabaseService", lambda db: svc)

    result = asyncio.run(metadata_router.get_query_preset(name="p1", db=_make_db()))
    assert result.name == "p1"


def test_get_query_preset_raises_404_when_missing(monkeypatch):
    from fastapi import HTTPException

    svc = MagicMock()
    svc.get_query_result.return_value = None
    monkeypatch.setattr(metadata_router, "DatabaseService", lambda db: svc)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.get_query_preset(name="no-such", db=_make_db()))
    assert exc_info.value.status_code == 404


def test_delete_query_preset_succeeds(monkeypatch):
    svc = MagicMock()
    svc.delete_query_result.return_value = True
    monkeypatch.setattr(metadata_router, "DatabaseService", lambda db: svc)

    asyncio.run(metadata_router.delete_query_preset(name="p1", db=_make_db()))
    svc.delete_query_result.assert_called_once_with("p1")


def test_delete_query_preset_raises_404_when_missing(monkeypatch):
    from fastapi import HTTPException

    svc = MagicMock()
    svc.delete_query_result.return_value = False
    monkeypatch.setattr(metadata_router, "DatabaseService", lambda db: svc)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(metadata_router.delete_query_preset(name="gone", db=_make_db()))
    assert exc_info.value.status_code == 404
