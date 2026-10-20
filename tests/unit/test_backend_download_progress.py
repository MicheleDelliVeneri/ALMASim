"""Download progress lists each file once (MADIV-28 T1.11).

download_products rewrites every access_url to spread transfers over the ALMA
mirrors, so the backend's progress tracking must not key files on the URL:
it used to append most files a second time.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def service(monkeypatch):
    """The backend download service, with its database modules stubbed for this test only."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src"))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "backend"))
    stub = MagicMock()
    for name in ("app.db", "app.db.config", "app.db.models", "app.db.service"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, stub)
    return importlib.import_module("app.services.download_service")


def _products(service, n):
    return [
        service.DataProduct(
            access_url=f"https://almascience.eso.org/dataPortal/file{i}.tar",
            uid=f"uid://A001/X1/X{i}",
            filename=f"file{i}.tar",
            content_length=100,
            content_type="application/x-tar",
            product_type="raw",
            semantics="#progenitor",
        )
        for i in range(n)
    ]


def _job(service, job_id):
    job = service.DownloadJob(
        job_id=job_id, destination="/tmp", member_ous_uids=["uid://A001/X1/X0"]
    )
    service.download_store._active.put(job_id, job)
    return job


def test_apply_file_update_matches_by_filename(service):
    job = _job(service, "job-filename")
    job.files = [
        service.FileDownloadStatus(
            filename="a.tar", access_url="https://eso/a.tar", content_length=100
        )
    ]
    mirrored = service.FileDownloadStatus(
        filename="a.tar",
        access_url="https://nrao/a.tar",  # rewritten to another mirror
        content_length=100,
        bytes_downloaded=100,
        status="completed",
    )
    service.download_store.apply_file_update("job-filename", mirrored)

    assert len(job.files) == 1
    assert job.files[0].status == "completed"
    assert (job.bytes_downloaded, job.files_completed, job.files_failed) == (100, 1, 0)
    service.download_store.remove("job-filename")


def test_run_download_job_lists_each_file_once(service, monkeypatch):
    products = _products(service, 6)
    job = _job(service, "job-run")
    seen_during_transfer = []

    def fake_download_products(products, destination, update_callback=None, **kwargs):
        statuses = []
        for i, product in enumerate(products):
            status = service.FileDownloadStatus(
                filename=product.filename,
                # Round-robin across mirrors, as _distribute_across_mirrors does.
                access_url=product.access_url.replace(
                    "almascience.eso.org", ["almascience.nrao.edu", "almascience.nao.ac.jp"][i % 2]
                ),
                content_length=product.content_length,
                bytes_downloaded=product.content_length,
                status="completed",
            )
            statuses.append(status)
            update_callback(status)
        seen_during_transfer.append(len(service.download_store.get("job-run").files))
        summary = MagicMock()
        summary.files = statuses
        summary.files_completed = len(statuses)
        summary.files_failed = 0
        summary.raw_measurement_sets = []
        summary.calibrated_measurement_sets = []
        summary.manifest_path = None
        return summary

    monkeypatch.setattr(service, "download_products", fake_download_products)
    monkeypatch.setattr(service.download_store, "persist", lambda job_id: None)
    service.run_download_job("job-run", products, "/tmp")

    assert seen_during_transfer == [6], "six files, not six plus their mirror-rewritten copies"
    assert job.status == "completed"
    assert job.files_completed == 6
    assert service.download_store.get("job-run") is None, "finished jobs leave memory"
