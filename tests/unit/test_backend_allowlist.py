"""Request paths and scheduler addresses must be allowed ones (MADIV-28 T1.9)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core import path_utils  # noqa: E402
from app.core.config import settings  # noqa: E402
from fastapi import HTTPException  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def roots(tmp_path, monkeypatch):
    host, out, data, extra = (tmp_path / n for n in ("host", "out", "data", "skao"))
    for d in (host, out, data, extra):
        d.mkdir()
    monkeypatch.setattr(settings, "HOST_DIR", host)
    monkeypatch.setattr(settings, "OUTPUT_DIR", out)
    monkeypatch.setattr(settings, "DATA_DIR", data)
    monkeypatch.setattr(settings, "ALLOWED_ROOTS", [extra])
    return host, out, data, extra


def test_paths_under_any_root_are_accepted(roots):
    host, out, _, extra = roots
    assert (
        path_utils.require_allowed_path(str(extra / "dl"), field="destination")
        == (extra / "dl").resolve()
    )
    assert path_utils.require_allowed_path("dl", field="destination") == (host / "dl").resolve()
    assert (
        path_utils.require_allowed_path("ms", field="ms_export_dir", relative_to=out)
        == (out / "ms").resolve()
    )


@pytest.mark.parametrize("raw", ["/etc", "../../outside", "/tmp/x"])
def test_paths_outside_every_root_are_refused(roots, raw):
    with pytest.raises(HTTPException) as exc_info:
        path_utils.require_allowed_path(raw, field="destination")
    assert exc_info.value.status_code == 400


def test_a_symlink_out_of_a_root_is_refused(roots, tmp_path):
    host = roots[0]
    (host / "link").symlink_to(tmp_path)
    with pytest.raises(HTTPException):
        path_utils.require_allowed_path(str(host / "link" / "x"), field="destination")


def test_only_configured_schedulers_are_accepted(monkeypatch):
    monkeypatch.setattr(settings, "DASK_SCHEDULER", "tcp://sched:8786")
    monkeypatch.setattr(settings, "ALLOWED_SCHEDULERS", ["tcp://other:8786"])
    assert path_utils.require_allowed_scheduler("tcp://sched:8786") == "tcp://sched:8786"
    assert path_utils.require_allowed_scheduler("tcp://other:8786") == "tcp://other:8786"
    with pytest.raises(HTTPException) as exc_info:
        path_utils.require_allowed_scheduler("tcp://evil:8786")
    assert exc_info.value.status_code == 400


def test_allowed_roots_reads_the_agreed_env_name(monkeypatch, tmp_path):
    from app.core.config import Settings

    monkeypatch.setenv("ALMASIM_ALLOWED_ROOTS", f'["{tmp_path}"]')
    assert Settings().ALLOWED_ROOTS == [tmp_path]
