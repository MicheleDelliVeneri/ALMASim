"""Unit tests for ALMASim Typer CLI commands."""

from __future__ import annotations

import tarfile
from types import SimpleNamespace

import click
import pandas as pd
import pytest
import typer
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


def test_products_wrapper_functions_delegate(monkeypatch):
    """Thin wrapper helpers should delegate to the underlying contract/functions."""
    import almasim.services.compute as compute_mod

    monkeypatch.setattr(compute_mod, "create_backend", lambda *args, **kwargs: (args, kwargs))
    monkeypatch.setattr(
        cli_products,
        "_download_contract",
        lambda: {
            "download_products": lambda *args, **kwargs: ("download", args, kwargs),
            "filter_products": lambda *args, **kwargs: ("filter", args, kwargs),
            "format_bytes": lambda *args, **kwargs: "fmt",
            "load_products_csv": lambda *args, **kwargs: ["loaded"],
            "resolve_products": lambda *args, **kwargs: ["resolved"],
            "save_products_csv": lambda *args, **kwargs: "saved.csv",
        },
    )

    assert cli_products.create_backend("slurm", n_workers=2) == (("slurm",), {"n_workers": 2})
    assert cli_products.download_products(["x"], destination="/tmp")[0] == "download"
    assert cli_products.filter_products([1], "all")[0] == "filter"
    assert cli_products.format_bytes(123) == "fmt"
    assert cli_products.load_products_csv("x.csv") == ["loaded"]
    assert cli_products.resolve_products(["uid://A"]) == ["resolved"]
    assert cli_products.save_products_csv([], "x.csv") == "saved.csv"


def test_preflight_casa_data_uses_resolved_path(monkeypatch, tmp_path):
    """Preflight should resolve CASA data path and pass skip flag through."""
    captured: dict[str, object] = {}

    def _fake_find_existing(input_root, output_root, casa_data_root):
        del input_root, output_root, casa_data_root
        return tmp_path / "resolved-casa"

    def _fake_ensure(path, skip_update=False, logger_fn=None):
        del logger_fn
        captured["path"] = path
        captured["skip_update"] = skip_update

    import types

    fake_unpack_mod = types.SimpleNamespace(
        ensure_casa_runtime_data=_fake_ensure,
        find_existing_casa_data=_fake_find_existing,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "almasim.services.archive.unpack_ms",
        fake_unpack_mod,
    )

    cli_products._preflight_casa_data(
        output_root=tmp_path,
        casa_data_root=None,
        skip_casa_data_update=True,
    )

    assert captured["path"] == tmp_path / "resolved-casa"
    assert captured["skip_update"] is True


def test_run_unpack_jobs_slurm_exits_when_no_uids(monkeypatch, tmp_path):
    """Slurm unpack should fail early when no ASDM inputs can be discovered."""
    monkeypatch.setattr(cli_products, "_extract_asdm_uids_from_download_root", lambda *_: [])

    with pytest.raises(typer.Exit):
        cli_products._run_unpack_jobs(
            input_root=tmp_path / "input",
            output_root=tmp_path / "out",
            asdm_uids=[],
            postprocess_backend="slurm",
            postprocess_backend_kwargs={},
            casa_data_root=None,
            skip_casa_data_update=False,
            overwrite_outputs=False,
        )


def test_run_calibrate_jobs_slurm_rejects_clean_intermediate(tmp_path):
    """Slurm calibrate should reject clean-intermediate mode."""
    with pytest.raises(typer.Exit):
        cli_products._run_calibrate_jobs(
            input_root=tmp_path / "input",
            raw_ms_root=tmp_path / "raw",
            output_root=tmp_path / "out",
            asdm_uids=["uid://A"],
            postprocess_backend="slurm",
            postprocess_backend_kwargs={},
            casa_data_root=None,
            skip_casa_data_update=False,
            overwrite_outputs=False,
            clean_intermediate=True,
        )


def test_run_unpack_jobs_slurm_preflights_and_submits(monkeypatch, tmp_path):
    """Slurm unpack should preflight CASA data and submit one job per UID."""
    captured: dict[str, object] = {}

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def delayed(self, fn):
            return lambda **kwargs: kwargs

    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_products, "create_backend", lambda *args, **kwargs: _Backend())

    def _fake_compute_jobs_with_progress(**kwargs):
        captured.update(kwargs)
        return [["/tmp/raw-a.ms"]]

    monkeypatch.setattr(
        cli_products, "_compute_jobs_with_progress", _fake_compute_jobs_with_progress
    )

    outputs = cli_products._run_unpack_jobs(
        input_root=tmp_path / "input",
        output_root=tmp_path / "out",
        asdm_uids=["uid://A"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_outputs=False,
    )

    assert outputs == ["/tmp/raw-a.ms"]
    assert captured["job_uids"] == ["uid://A"]


def test_run_calibrate_jobs_slurm_preflights_and_submits(monkeypatch, tmp_path):
    """Slurm calibrate should preflight CASA data and submit one job per UID."""
    captured: dict[str, object] = {}

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def delayed(self, fn):
            return lambda **kwargs: kwargs

    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_products, "create_backend", lambda *args, **kwargs: _Backend())

    def _fake_compute_jobs_with_progress(**kwargs):
        captured.update(kwargs)
        return [["/tmp/cal-a.ms"]]

    monkeypatch.setattr(
        cli_products, "_compute_jobs_with_progress", _fake_compute_jobs_with_progress
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid://A"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert outputs == ["/tmp/cal-a.ms"]
    assert captured["job_uids"] == ["uid://A"]


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


def test_calibrate_single_uid_streams_logs_and_returns_expected_output(tmp_path, monkeypatch):
    """Calibration worker should stream subprocess logs and return the expected split.cal path."""
    captured: dict[str, object] = {}

    class _FakeProcess:
        def __init__(self):
            self.stdout = iter(["first line\n", "second line\n"])

        def wait(self):
            return 0

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setenv("PYTHONPATH", "existing-path")
    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    uid = "uid___A001_X1_X1"
    output_root = tmp_path / "calibrated"
    expected = output_root / f"{uid}.ms.split.cal"
    expected.mkdir(parents=True)

    outputs = cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid=uid,
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=True,
        clean_intermediate=False,
    )

    assert outputs == [str(expected)]
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--overwrite-outputs" in cmd
    assert "--skip-casa-data-update" in cmd

    kwargs = captured["kwargs"]
    assert kwargs["cwd"].endswith("ALMASim")
    assert "PYTHONPATH" in kwargs["env"]
    assert kwargs["env"]["PYTHONPATH"].endswith(":existing-path")


def test_calibrate_single_uid_uses_fallback_glob_when_expected_missing(tmp_path, monkeypatch):
    """Calibration worker should discover split.cal outputs via glob
    when canonical path is absent.
    """

    class _FakeProcess:
        def __init__(self):
            self.stdout = iter(["ok\n"])

        def wait(self):
            return 0

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

    uid = "uid___A001_X1_X2"
    output_root = tmp_path / "calibrated"
    fallback = output_root / f"{uid}_extra.ms.split.cal"
    fallback.mkdir(parents=True)

    outputs = cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid=uid,
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )

    assert outputs == [str(fallback)]


def test_calibrate_single_uid_raises_with_bounded_log_tail(tmp_path, monkeypatch):
    """Calibration worker should include only the recent bounded tail on subprocess failure."""

    class _FakeProcess:
        def __init__(self):
            self.stdout = iter([f"line-{idx}\n" for idx in range(205)])

        def wait(self):
            return 7

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

    uid = "uid___A001_X1_X3"
    output_root = tmp_path / "calibrated"
    output_root.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Calibration failed") as exc:
        cli_products._calibrate_single_uid(
            input_root="/input",
            raw_ms_root="/raw",
            calibrated_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
            clean_intermediate=False,
        )

    msg = str(exc.value)
    assert "Return code: 7" in msg
    assert "Last 200 log lines" in msg
    assert "line-0" not in msg
    assert "line-204" in msg


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


def test_metadata_wrappers_delegate(monkeypatch):
    """Metadata wrapper helpers should delegate through the TAP contract."""
    monkeypatch.setattr(
        cli_metadata,
        "_tap_contract",
        lambda: {
            "query_metadata_by_science": lambda *args, **kwargs: ("meta", args, kwargs),
            "query_products": lambda *args, **kwargs: ("products", args, kwargs),
        },
    )

    assert cli_metadata.query_metadata_by_science(science_keyword=["Galaxies"])[0] == "meta"
    assert cli_metadata.query_products(["uid://A"])[0] == "products"


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


def test_invoke_click_command_raises_typer_exit_for_nonzero_int_result():
    """Click command return codes should map to Typer exit codes."""

    class _FakeClickCommand:
        def main(self, **kwargs):
            del kwargs
            return 9

    with pytest.raises(typer.Exit) as exc:
        cli._invoke_click_command(_FakeClickCommand(), args=[], prog_name="almasim")

    assert exc.value.exit_code == 9


def test_invoke_click_command_maps_click_exit_to_typer_exit():
    """Click Exit exceptions should be mapped to Typer Exit with same code."""

    class _FakeClickCommand:
        def main(self, **kwargs):
            del kwargs
            raise click.exceptions.Exit(3)

    with pytest.raises(typer.Exit) as exc:
        cli._invoke_click_command(_FakeClickCommand(), args=[], prog_name="almasim")

    assert exc.value.exit_code == 3
