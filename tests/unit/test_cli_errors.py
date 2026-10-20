"""CLI usage errors exit 2 and failures exit 1, with a message on stderr (MADIV-28 T3.3)."""

from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from almasim import cli, cli_clean, cli_metadata, cli_products

pytestmark = pytest.mark.unit
runner = CliRunner()


def _invoke(*args):
    return runner.invoke(cli.app, [str(a) for a in args])


@pytest.mark.parametrize("command", ["extract", "unpack", "calibrate"])
def test_archive_stages_refuse_an_unknown_backend(command):
    result = _invoke("visibilities", command, "--postprocess-backend", "dask")
    assert result.exit_code == 2
    assert "--postprocess-backend must be one of: sync, slurm." in result.output


def test_extract_needs_an_existing_source_with_archives(tmp_path):
    result = _invoke("visibilities", "extract", "--source-root", tmp_path / "missing")
    assert result.exit_code == 2 and "--source-root is not a directory" in result.output
    result = _invoke("visibilities", "extract", "--source-root", tmp_path)
    assert result.exit_code == 1 and "No .tar/.tgz archives found" in result.output


def test_resolve_needs_member_uids(tmp_path):
    result = _invoke("products", "resolve")
    assert result.exit_code == 2
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"source": ["a"]}).to_csv(csv, index=False)
    result = _invoke("products", "resolve", "--metadata-csv", csv)
    assert result.exit_code == 2 and "does not contain member_ous_uid" in result.output


def test_download_validates_its_options_and_reports_empty_results(monkeypatch):
    result = _invoke("products", "download", "--member-ous-uid", "uid://A", "--product-filter", "x")
    assert result.exit_code == 2 and "Invalid --product-filter" in result.output
    result = _invoke(
        "products", "download", "--member-ous-uid", "uid://A", "--postprocess-backend", "dask"
    )
    assert result.exit_code == 2 and "--postprocess-backend must be one of" in result.output
    monkeypatch.setattr(cli_products, "resolve_products", lambda uids, **kw: [])
    result = _invoke("products", "download", "--member-ous-uid", "uid://A", "--yes")
    assert result.exit_code == 1


def test_metadata_query_refuses_unknown_array_types_and_reports_no_rows(monkeypatch):
    result = _invoke("metadata", "query", "--array-type", "9m")
    assert result.exit_code == 2 and "--array-type" in result.output
    monkeypatch.setattr(cli_metadata, "query_metadata_by_science", lambda **kw: pd.DataFrame())
    result = _invoke("metadata", "query", "--science-keyword", "Galaxies", "--yes")
    assert result.exit_code == 1 and "No metadata rows matched" in result.output


def test_simulation_run_checks_its_inputs(tmp_path):
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"member_ous_uid": ["uid://A"]}).to_csv(csv, index=False)
    cases = [
        (["--save-mode", "pickle"], "--save-mode must be one of"),
        (["--input", tmp_path / "none.fits"], "--input not found"),
        (["--kaggle-token", tmp_path / "kaggle.json"], "--kaggle-token not found"),
    ]
    for extra, message in cases:
        result = _invoke("simulation", "run", "--metadata-path", csv, *extra)
        assert result.exit_code == 2 and message in result.output, (extra, result.output)
    result = _invoke("simulation", "run", "--metadata-path", tmp_path / "missing.csv")
    assert result.exit_code == 2 and "--metadata-path not found" in result.output


def test_clean_needs_arguments_and_reports_a_wsclean_that_cannot_start(monkeypatch):
    from types import SimpleNamespace

    import typer

    with pytest.raises(typer.Exit) as info:
        cli_clean.clean_command(
            SimpleNamespace(args=[]), wsclean_bin="wsclean", cwd=None, dry_run=False
        )
    assert info.value.exit_code == 2

    def broken(*args, **kwargs):
        raise OSError("exec format error")

    monkeypatch.setattr(cli_clean.subprocess, "run", broken)
    monkeypatch.setenv("ALMASIM_WSCLEAN_BIN", "/bin/true")
    result = _invoke("clean", "--", "-name", "x", "in.ms")
    assert result.exit_code == 1 and "Failed to start WSClean" in result.output


def test_flag_summary_refuses_a_path_that_is_no_measurement_set(tmp_path):
    not_ms = tmp_path / "notes.txt"
    not_ms.write_text("x")
    result = _invoke("visibilities", "flag-summary", not_ms, "--output", tmp_path / "s.csv")
    assert result.exit_code == 1 and "Not a MeasurementSet or folder" in result.output
