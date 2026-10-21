"""The metadata command (MADIV-28 T8.2: split from test_cli.py)."""

from __future__ import annotations

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


def test_metadata_query_saves_datalink_rows_that_download_reads(tmp_path, monkeypatch):
    """--save-products-csv resolves DataLink for the queried UIDs (products-CSV schema)."""
    metadata = pd.DataFrame({"Band": [6, 6, 6], "member_ous_uid": ["uid://1", None, "uid://3"]})
    monkeypatch.setattr(cli_metadata, "query_metadata_by_science", lambda **kwargs: metadata)
    captured: dict[str, object] = {}

    def _fake_resolve(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", _fake_resolve)
    products_csv = tmp_path / "products.csv"
    result = runner.invoke(
        cli.app,
        ["metadata", "query", "--yes", "--save-products-csv", str(products_csv)],
    )

    assert result.exit_code == 0, result.output
    assert captured["member_ous_uid"] == ["uid://1", "uid://3"]
    assert captured["save_products_csv_path"] == products_csv


def test_metadata_cli_uses_the_tap_service_directly():
    """No lazy contract: the CLI's names are the TAP service's own functions."""
    from almasim.services.metadata import tap

    assert cli_metadata.query_metadata_by_science is tap.query_metadata_by_science


def test_metadata_query_invalid_visible_column_shows_allowed_columns():
    """Metadata query should print allowed columns when visible columns are invalid."""
    result = runner.invoke(
        cli.app,
        [
            "metadata",
            "query",
            "--visible-column",
            "not-a-real-column",
            "--yes",
        ],
    )

    assert result.exit_code == 2
    assert "Allowed columns:" in result.output
