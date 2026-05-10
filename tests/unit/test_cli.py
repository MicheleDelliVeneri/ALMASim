"""Unit tests for ALMASim Typer CLI commands."""

from __future__ import annotations

import tarfile
from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from almasim import cli, cli_clean, cli_metadata, cli_products

runner = CliRunner()


def test_metadata_query_cancelled(monkeypatch):
    """Metadata query should stop before TAP calls when user declines prompt."""
    called = {"query": False}

    def _fake_query(*args, **kwargs):
        called["query"] = True
        raise AssertionError("query_metadata_by_science should not be called")

    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: False)
    monkeypatch.setattr(cli_metadata, "query_metadata_by_science", _fake_query)

    result = runner.invoke(cli.app, ["metadata", "query"])

    assert result.exit_code == 0
    assert "Query cancelled." in result.output
    assert called["query"] is False


def test_metadata_query_default_unlimited_rows(tmp_path, monkeypatch):
    """When --limit is omitted, metadata rows should not be truncated."""
    metadata = pd.DataFrame(
        {
            "ALMA_source_name": ["a", "b", "c"],
            "Band": [6, 6, 6],
            "Freq": [100.0, 101.0, 102.0],
            "member_ous_uid": ["uid://1", "uid://2", "uid://3"],
        }
    )

    monkeypatch.setattr(cli_metadata, "query_metadata_by_science", lambda **kwargs: metadata)

    output_csv = tmp_path / "metadata.csv"
    result = runner.invoke(
        cli.app,
        [
            "metadata",
            "query",
            "--yes",
            "--save-csv",
            str(output_csv),
        ],
    )

    assert result.exit_code == 0
    saved = pd.read_csv(output_csv)
    assert len(saved) == 3


def test_products_resolve_extracts_member_uids(tmp_path, monkeypatch):
    """Resolve should extract member_ous_uid values and write the list file."""
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "member_ous_uid": ["uid://A", "uid://B", "uid://A", None],
            "Band": [6, 7, 6, 3],
        }
    ).to_csv(metadata_csv, index=False)

    captured: dict[str, object] = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(), SimpleNamespace()]

    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", _fake_resolve)

    uid_list_path = tmp_path / "uids.txt"
    products_csv = tmp_path / "products.csv"
    result = runner.invoke(
        cli.app,
        [
            "products",
            "resolve",
            "--metadata-csv",
            str(metadata_csv),
            "--save-member-ous-uid-list",
            str(uid_list_path),
            "--save-products-csv",
            str(products_csv),
        ],
    )

    assert result.exit_code == 0
    assert uid_list_path.read_text(encoding="utf-8") == "uid://A\nuid://B\n"
    assert captured["member_ous_uid"] == ["uid://A", "uid://B"]


def test_products_download_invalid_filter_exits():
    """Invalid product filter should fail fast before resolution/download calls."""
    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            "uid://A",
            "--product-filter",
            "not-a-type",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid --product-filter" in result.output


def test_products_download_slurm_postprocess_path(tmp_path, monkeypatch):
    """Slurm mode should run download first, then post-process archive jobs."""
    products = [SimpleNamespace(content_length=10)]

    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)

    destination = tmp_path / "downloads"
    captured: dict[str, object] = {}

    def _fake_download_products(*args, **kwargs):
        captured.update(kwargs)
        assert callable(kwargs["update_callback"])
        destination.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(
            destination=str(destination),
            files_completed=1,
            files_failed=0,
            manifest_path=str(destination / "download_manifest.json"),
        )

    monkeypatch.setattr(cli_products, "download_products", _fake_download_products)
    monkeypatch.setattr(
        cli_products,
        "_run_parallel_archive_jobs",
        lambda **kwargs: (["/tmp/raw.ms"], ["/tmp/cal.ms"]),
    )

    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            "uid://A",
            "--yes",
            "--postprocess-backend",
            "slurm",
            "--unpack-ms",
            "--generate-calibrated-visibilities",
        ],
    )

    assert result.exit_code == 0
    assert callable(captured["update_callback"])
    assert "Raw MS products:" in result.output
    assert "Calibrated MS products:" in result.output


def test_products_download_sync_path(tmp_path, monkeypatch):
    """Sync mode should keep archive processing in download_products call."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)

    captured: dict[str, object] = {}

    def _fake_download_products(*args, **kwargs):
        captured.update(kwargs)
        assert callable(kwargs["update_callback"])
        return SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        )

    monkeypatch.setattr(cli_products, "download_products", _fake_download_products)

    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            "uid://A",
            "--yes",
            "--postprocess-backend",
            "sync",
            "--unpack-ms",
        ],
    )

    assert result.exit_code == 0
    assert callable(captured["update_callback"])
    assert captured["unpack_ms"] is True
    assert captured["generate_calibrated_visibilities"] is False


class _FakeFuture:
    def __init__(self, *, done_after: int = 1, status: str = "finished"):
        self._calls = 0
        self._done_after = done_after
        self.status = status

    def done(self):
        self._calls += 1
        return self._calls >= self._done_after


class _FakeAsyncBackend:
    def __init__(self, futures, gathered):
        self._futures = futures
        self._gathered = gathered
        self.compute_sync_values = []
        self.gather_called = False

    def compute(self, jobs, sync=True):
        self.compute_sync_values.append(sync)
        return self._futures

    def gather(self, futures):
        self.gather_called = True
        assert futures is self._futures
        return self._gathered


def test_compute_jobs_with_progress_uses_async_and_gather(monkeypatch):
    """Slurm progress helper should compute async and gather futures."""
    backend = _FakeAsyncBackend(
        futures=[_FakeFuture(done_after=1), _FakeFuture(done_after=2)],
        gathered=[["raw1.ms"], ["raw2.ms"]],
    )

    monkeypatch.setattr(cli_products, "sleep", lambda *_args, **_kwargs: None)

    results = cli_products._compute_jobs_with_progress(
        backend=backend,
        jobs=[object(), object()],
        job_uids=["uid://A", "uid://B"],
        stage_label="Slurm unpack",
    )

    assert backend.compute_sync_values == [False]
    assert backend.gather_called is True
    assert results == [["raw1.ms"], ["raw2.ms"]]


def test_products_extract_standalone(tmp_path):
    """Standalone extract should unpack tar archives from disk."""
    source_root = tmp_path / "downloads"
    source_root.mkdir(parents=True)
    payload = source_root / "data.txt"
    payload.write_text("hello", encoding="utf-8")
    archive_path = source_root / "bundle.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(payload, arcname="nested/data.txt")
    payload.unlink()

    result = runner.invoke(
        cli.app,
        [
            "products",
            "extract",
            "--source-root",
            str(source_root),
            "--no-recursive",
        ],
    )

    assert result.exit_code == 0
    assert (source_root / "nested" / "data.txt").is_file()
    assert "Extracted files: 1" in result.output


def test_products_unpack_standalone(monkeypatch):
    """Standalone unpack should delegate to unpack job runner and print outputs."""
    monkeypatch.setattr(
        cli_products,
        "_run_unpack_jobs",
        lambda **kwargs: ["/tmp/raw-a.ms", "/tmp/raw-b.ms"],
    )

    result = runner.invoke(
        cli.app,
        [
            "products",
            "unpack",
            "--asdm-uid",
            "uid://A,uid://B",
        ],
    )

    assert result.exit_code == 0
    assert "Raw MS products: 2" in result.output


def test_products_calibrate_standalone(monkeypatch):
    """Standalone calibrate should delegate to calibrate job runner and print outputs."""
    monkeypatch.setattr(
        cli_products,
        "_run_calibrate_jobs",
        lambda **kwargs: ["/tmp/cal-a.ms"],
    )

    result = runner.invoke(
        cli.app,
        [
            "products",
            "calibrate",
            "--asdm-uid",
            "uid://A",
        ],
    )

    assert result.exit_code == 0
    assert "Calibrated MS products: 1" in result.output


def test_clean_passthrough_runs_wsclean(monkeypatch):
    """Top-level clean command should forward unknown options to WSClean."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli_clean.shutil, "which", lambda executable: f"/usr/bin/{executable}")

    def _fake_run(command, cwd=None, check=False):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_clean.subprocess, "run", _fake_run)

    result = runner.invoke(
        cli.app,
        [
            "clean",
            "--wsclean-bin",
            "wsclean",
            "--",
            "-name",
            "img",
            "-niter",
            "1000",
            "input.ms",
        ],
    )

    assert result.exit_code == 0
    assert captured["command"] == [
        "/usr/bin/wsclean",
        "-name",
        "img",
        "-niter",
        "1000",
        "input.ms",
    ]
