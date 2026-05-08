"""Unit tests for ALMASim Typer CLI commands."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from typer.testing import CliRunner

from almasim import cli, cli_metadata, cli_products

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

    def _fake_download_products(*args, **kwargs):
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
    assert captured["unpack_ms"] is True
    assert captured["generate_calibrated_visibilities"] is False
