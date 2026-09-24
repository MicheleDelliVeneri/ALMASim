"""Unit tests for ALMASim Typer CLI commands."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import click
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from almasim import cli, cli_clean, cli_image, cli_metadata, cli_predict, cli_products

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
    """Sync mode should download first, then run per-UID archive stages like Slurm does."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)

    captured: dict[str, object] = {}
    stage_kwargs: dict[str, object] = {}

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

    def _fake_stages(**kwargs):
        stage_kwargs.update(kwargs)
        return (["/tmp/raw-a.ms"], [])

    monkeypatch.setattr(cli_products, "download_products", _fake_download_products)
    monkeypatch.setattr(cli_products, "_run_parallel_archive_jobs", _fake_stages)

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
    # In-process unpack/calibrate inside download_products is no longer used.
    assert captured["unpack_ms"] is False
    assert captured["generate_calibrated_visibilities"] is False
    assert stage_kwargs["postprocess_backend"] == "sync"
    assert stage_kwargs["postprocess_backend_kwargs"] == {}
    assert stage_kwargs["unpack_ms"] is True
    assert stage_kwargs["continue_on_error"] is True
    assert "Raw MS products:" in result.output


def test_products_download_sync_without_postprocess_keeps_single_call(tmp_path, monkeypatch):
    """Plain downloads (no unpack/calibrate) still go through download_products alone."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)
    monkeypatch.setattr(
        cli_products,
        "_run_parallel_archive_jobs",
        lambda **kwargs: pytest.fail("no archive stages expected"),
    )
    monkeypatch.setattr(
        cli_products,
        "download_products",
        lambda *args, **kwargs: SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        ),
    )

    result = runner.invoke(
        cli.app, ["products", "download", "--member-ous-uid", "uid://A", "--yes"]
    )
    assert result.exit_code == 0


def test_products_download_skip_qa0_semipass_filters_raw_products(tmp_path, monkeypatch):
    """--skip-qa0-semipass drops SemiPass raw ASDMs and reports what it skipped."""
    from almasim.services.archive import qa0

    member = "uid://A001/X378a/X149"
    member_dir = (
        tmp_path
        / "downloads"
        / "2023.1.00879.S"
        / "science_goal.uid___A001_X378a_X147"
        / "group.uid___A001_X378a_X148"
        / "member.uid___A001_X378a_X149"
        / "qa"
    )
    member_dir.mkdir(parents=True)
    for eb in ("uid___A002_X1_Xpass", "uid___A002_X1_Xsemi"):
        (member_dir / f"{eb}.qa0_report.pdf").write_bytes(b"%PDF-1.4")

    def fake_reader(path, eb_uid=""):
        status = "SemiPass" if eb_uid.endswith("Xsemi") else "Pass"
        return qa0.QA0Report(eb_uid, status, "SUCCESS", 1.0, "")

    monkeypatch.setattr(qa0, "read_qa0_report", fake_reader)

    raw_pass = SimpleNamespace(
        product_type="raw",
        uid=member,
        filename="2023.1.00879.S_uid___A002_X1_Xpass.asdm.sdm.tar",
        content_length=10,
    )
    raw_semi = SimpleNamespace(
        product_type="raw",
        uid=member,
        filename="2023.1.00879.S_uid___A002_X1_Xsemi.asdm.sdm.tar",
        content_length=2048,
    )
    monkeypatch.setattr(
        cli_products, "_resolve_products_from_inputs", lambda **kwargs: [raw_pass, raw_semi]
    )
    captured: dict[str, object] = {}

    def fake_download(products, *args, **kwargs):
        captured["products"] = list(products)
        return SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=None,
            raw_measurement_sets=[],
            calibrated_measurement_sets=[],
        )

    monkeypatch.setattr(cli_products, "download_products", fake_download)

    result = runner.invoke(
        cli.app,
        [
            "products",
            "download",
            "--member-ous-uid",
            member,
            "--product-filter",
            "raw",
            "--destination",
            str(tmp_path / "downloads"),
            "--skip-qa0-semipass",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "QA0 filter: skipped 1 SemiPass raw product(s)" in result.output
    assert captured["products"] == [raw_pass]


def test_products_download_sync_cleanup_only_when_no_failures(tmp_path, monkeypatch):
    """--clean-intermediate-files runs after a fully successful sync post-processing run."""
    products = [SimpleNamespace(content_length=12)]
    monkeypatch.setattr(cli_products, "_resolve_products_from_inputs", lambda **kwargs: products)
    monkeypatch.setattr(cli_products, "filter_products", lambda products, product_filter: products)
    manifest = tmp_path / "downloads" / "download_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"raw_measurement_sets": [], "calibrated_measurement_sets": []}')
    monkeypatch.setattr(
        cli_products,
        "download_products",
        lambda *args, **kwargs: SimpleNamespace(
            destination=str(tmp_path / "downloads"),
            files_completed=1,
            files_failed=0,
            manifest_path=str(manifest),
            files=[],
            extracted_files=[],
        ),
    )
    cleaned: list[str] = []
    monkeypatch.setattr(
        cli_products, "_cleanup_after_download_postprocess", lambda **kwargs: cleaned.append("yes")
    )

    def _stages_ok(**kwargs):
        return (["/raw/a.ms"], ["/cal/a.ms.split.cal"])

    def _stages_failing(**kwargs):
        kwargs["failures"].append(cli_products.StageFailure("uid___B", "boom"))
        return (["/raw/a.ms"], ["/cal/a.ms.split.cal"])

    args = [
        "products",
        "download",
        "--member-ous-uid",
        "uid://A",
        "--yes",
        "--unpack-ms",
        "--generate-calibrated-visibilities",
        "--clean-intermediate-files",
    ]

    monkeypatch.setattr(cli_products, "_run_parallel_archive_jobs", _stages_ok)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0
    assert cleaned == ["yes"]
    import json

    assert json.loads(manifest.read_text())["calibrated_measurement_sets"] == [
        "/cal/a.ms.split.cal"
    ]

    monkeypatch.setattr(cli_products, "_run_parallel_archive_jobs", _stages_failing)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert cleaned == ["yes"]  # not called a second time
    assert "Skipping --clean-intermediate-files" in result.output


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


def test_predict_ms_from_image_single_ms_invokes_prediction(monkeypatch, tmp_path):
    """predict ms-from-image should handle a single MS without slurm."""
    ms_dir = tmp_path / "sample.ms"
    ms_dir.mkdir()
    out_dir = tmp_path / "models"
    out_dir.mkdir()

    captured: dict[str, object] = {}

    def _fake_predict_all_models_for_ms(input_ms, output_directory):
        captured["input_ms"] = input_ms
        captured["output_directory"] = output_directory
        return [output_directory / input_ms.stem / "SPW-0" / "sample.ms.predicted"]

    monkeypatch.setattr(cli_predict, "_predict_all_models_for_ms", _fake_predict_all_models_for_ms)

    result = runner.invoke(
        cli.app,
        ["predict", "ms-from-image", str(ms_dir), str(out_dir), "--no-use-slurm"],
    )

    assert result.exit_code == 0
    assert captured["input_ms"] == ms_dir
    assert captured["output_directory"] == out_dir


def test_predict_ms_from_image_slurm_dispatches_one_job_per_ms(monkeypatch, tmp_path):
    """predict ms-from-image should submit one Slurm command per MS in a folder."""
    ms_dir = tmp_path / "inputs"
    ms_dir.mkdir()
    (ms_dir / "a.ms").mkdir()
    (ms_dir / "b.ms").mkdir()
    out_dir = tmp_path / "models"
    out_dir.mkdir()

    captured: dict[str, object] = {}

    def _fake_run_with_slurm_cluster(*args, **kwargs):
        captured["commands"] = args[0]

    monkeypatch.setattr(cli_image, "_run_commands_with_slurm_cluster", _fake_run_with_slurm_cluster)

    result = runner.invoke(
        cli.app,
        ["predict", "ms-from-image", str(ms_dir), str(out_dir)],
    )

    assert result.exit_code == 0
    assert len(captured["commands"]) == 2
    assert captured["commands"][0][1][:3] == ["almasim", "predict", "ms-from-image"]


# ---------------------------------------------------------------------------
# Skip-on-failure and per-UID logging for archive post-processing stages
# ---------------------------------------------------------------------------


class _FakeFailingFuture(_FakeFuture):
    def __init__(self, message: str):
        super().__init__(done_after=1, status="error")
        self._message = message

    def exception(self):
        return RuntimeError(self._message)


class _FakePerFutureBackend:
    """Async backend whose gather() works on single-future lists."""

    def __init__(self, futures, results):
        self._futures = futures
        self._results = results

    def compute(self, jobs, sync=True):
        return self._futures

    def gather(self, futures):
        return [self._results[self._futures.index(future)] for future in futures]


def test_compute_jobs_with_progress_continue_on_error_skips_failed(monkeypatch):
    """A failed future should become a None placeholder and a recorded StageFailure."""
    futures = [_FakeFuture(done_after=1), _FakeFailingFuture("boom on B"), _FakeFuture()]
    backend = _FakePerFutureBackend(futures, results=[["a.ms"], None, ["c.ms"]])
    monkeypatch.setattr(cli_products, "sleep", lambda *_args, **_kwargs: None)
    failures: list[cli_products.StageFailure] = []

    results = cli_products._compute_jobs_with_progress(
        backend=backend,
        jobs=[object(), object(), object()],
        job_uids=["uid://A", "uid://B", "uid://C"],
        stage_label="Slurm calibrate",
        continue_on_error=True,
        log_paths=[Path("/logs/a.log"), Path("/logs/b.log"), Path("/logs/c.log")],
        failures=failures,
    )

    assert results == [["a.ms"], None, ["c.ms"]]
    assert [failure.uid for failure in failures] == ["uid://B"]
    assert "boom on B" in failures[0].error
    assert failures[0].log_path == "/logs/b.log"


def test_compute_jobs_with_progress_fail_fast_still_raises(monkeypatch):
    """Without continue_on_error the helper defers to backend.gather, which raises."""

    class _RaisingBackend(_FakePerFutureBackend):
        def gather(self, futures):
            raise RuntimeError("gather failed")

    backend = _RaisingBackend([_FakeFailingFuture("boom")], results=[None])
    monkeypatch.setattr(cli_products, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="gather failed"):
        cli_products._compute_jobs_with_progress(
            backend=backend,
            jobs=[object()],
            job_uids=["uid://A"],
            stage_label="Slurm calibrate",
        )


def test_compute_jobs_with_progress_heartbeat_prints_log_tail(monkeypatch, tmp_path):
    """While jobs run, the last line of each UID log should be echoed periodically."""
    log_path = tmp_path / "uid_A.calibrate.log"
    log_path.write_text("starting\napplycal 2/5 running\n", encoding="utf-8")
    future = _FakeFuture(done_after=3)
    backend = _FakePerFutureBackend([future], results=[["a.ms"]])
    written: list[str] = []

    class _Bar:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def set_postfix_str(self, *_args):
            pass

        def update(self, *_args):
            pass

        def write(self, text):
            written.append(text)

    monkeypatch.setattr(cli_products, "tqdm", _Bar)
    # The worker "writes" to the log while the stage is running.
    monkeypatch.setattr(cli_products, "sleep", lambda *_args, **_kwargs: os.utime(log_path, None))

    cli_products._compute_jobs_with_progress(
        backend=backend,
        jobs=[object()],
        job_uids=["uid://A"],
        stage_label="Slurm calibrate",
        log_paths=[log_path],
        heartbeat_interval=0.0,
    )

    assert any("applycal 2/5 running" in line for line in written)
    assert any("1 running" in line for line in written)


def test_run_uid_stage_sync_continues_after_failure(monkeypatch, tmp_path):
    """Sync stage runner should skip the failing UID and keep the others."""
    calls: list[str] = []

    def _task(*, asdm_uid: str) -> list[str]:
        calls.append(asdm_uid)
        if asdm_uid == "uid://B":
            raise RuntimeError("Calibration failed for uid://B.\nReturn code: 134")
        return [f"/out/{asdm_uid}.ms.split.cal"]

    failures: list[cli_products.StageFailure] = []
    uids = ["uid://A", "uid://B", "uid://C"]
    results = cli_products._run_uid_stage(
        backend=None,
        postprocess_backend="sync",
        task_fn=_task,
        job_kwargs=[{"asdm_uid": uid} for uid in uids],
        job_uids=uids,
        stage_label="Calibrate",
        log_paths=[tmp_path / f"{i}.log" for i in range(3)],
        continue_on_error=True,
        failures=failures,
    )

    assert calls == uids
    assert results == [["/out/uid://A.ms.split.cal"], [], ["/out/uid://C.ms.split.cal"]]
    assert [failure.uid for failure in failures] == ["uid://B"]
    assert failures[0].log_path == str(tmp_path / "1.log")


def test_run_uid_stage_sync_fail_fast_raises():
    """With continue_on_error disabled the first failure propagates."""

    def _task(*, asdm_uid: str) -> list[str]:
        raise RuntimeError(f"failed {asdm_uid}")

    with pytest.raises(RuntimeError, match="failed uid://A"):
        cli_products._run_uid_stage(
            backend=None,
            postprocess_backend="sync",
            task_fn=_task,
            job_kwargs=[{"asdm_uid": "uid://A"}],
            job_uids=["uid://A"],
            stage_label="Calibrate",
            log_paths=[],
            continue_on_error=False,
            failures=[],
        )


def test_run_calibrate_jobs_sync_multi_uid_isolates_each_uid(monkeypatch, tmp_path):
    """Sync calibrate over several UIDs should run one subprocess wrapper per UID."""
    seen: list[dict[str, object]] = []

    def _fake_single(**kwargs):
        seen.append(kwargs)
        if kwargs["asdm_uid"] == "uid___B":
            raise RuntimeError("casacore::ArrayError")
        return [f"{kwargs['calibrated_output_root']}/{kwargs['asdm_uid']}.ms.split.cal"]

    monkeypatch.setattr(cli_products, "_calibrate_single_uid", _fake_single)
    monkeypatch.setattr(
        cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "out" / ".casa-data"
    )
    monkeypatch.setattr(
        cli_products, "_extract_uids_from_raw_ms_root", lambda *_: ["uid___A", "uid___B", "uid___C"]
    )

    failures: list[cli_products.StageFailure] = []
    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=[],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_outputs=False,
        clean_intermediate=False,
        failures=failures,
    )

    assert [entry["asdm_uid"] for entry in seen] == ["uid___A", "uid___B", "uid___C"]
    assert all(entry["casa_data_root"] == str(tmp_path / "out" / ".casa-data") for entry in seen)
    assert all(entry["skip_casa_data_update"] is True for entry in seen)
    assert outputs == [
        str(tmp_path / "out" / "uid___A.ms.split.cal"),
        str(tmp_path / "out" / "uid___C.ms.split.cal"),
    ]
    assert [failure.uid for failure in failures] == ["uid___B"]


def test_run_calibrate_jobs_sync_single_uid_runs_in_process(monkeypatch, tmp_path):
    """A single UID (the per-UID worker entry point) must not spawn another subprocess."""
    import almasim.services.archive as archive_mod

    captured: dict[str, object] = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return [tmp_path / "out" / "uid___A.ms.split.cal"]

    monkeypatch.setattr(archive_mod, "create_calibrated_measurement_sets", _fake_create)
    monkeypatch.setattr(
        cli_products,
        "_calibrate_single_uid",
        lambda **kwargs: pytest.fail("subprocess wrapper must not be used for one UID"),
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert outputs == [str(tmp_path / "out" / "uid___A.ms.split.cal")]
    assert captured["asdm_uid"] == "uid___A"
    assert callable(captured["logger_fn"])


def test_run_calibrate_jobs_sync_skips_cleanup_when_failures(monkeypatch, tmp_path):
    """Intermediate cleanup must not run when any UID failed."""
    import almasim.services.archive as archive_mod

    monkeypatch.setattr(
        cli_products,
        "_calibrate_single_uid",
        lambda **kwargs: (
            (_ for _ in ()).throw(RuntimeError("boom"))
            if kwargs["asdm_uid"] == "uid___B"
            else ["/out/ok.ms.split.cal"]
        ),
    )
    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa")
    monkeypatch.setattr(
        archive_mod,
        "cleanup_intermediate_calibration_data",
        lambda *a, **k: pytest.fail("cleanup must be skipped after failures"),
    )

    failures: list[cli_products.StageFailure] = []
    cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A", "uid___B"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=True,
        failures=failures,
    )
    assert [failure.uid for failure in failures] == ["uid___B"]


def test_calibrate_single_uid_writes_per_uid_log_file(tmp_path, monkeypatch):
    """The subprocess wrapper should mirror child output into <output>/logs/<uid>.calibrate.log."""

    class _FakeProcess:
        def __init__(self):
            self.stdout = iter(["applying\n", "splitting\n"])

        def wait(self):
            return 0

    captured: dict[str, object] = {}

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _FakeProcess()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    uid = "uid___A001_X1_X9"
    output_root = tmp_path / "calibrated"
    (output_root / f"{uid}.ms.split.cal").mkdir(parents=True)

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

    log_path = output_root / "logs" / f"{uid}.calibrate.log"
    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("# ")
    assert "applying\nsplitting\n" in content
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"


def test_products_calibrate_exits_nonzero_with_failure_summary(monkeypatch):
    """The command should list outputs, print skipped UIDs, and exit 1."""

    def _fake_run(**kwargs):
        kwargs["failures"].append(
            cli_products.StageFailure("uid___B", "Calibration failed for uid___B.", "/logs/b.log")
        )
        assert kwargs["continue_on_error"] is True
        return ["/tmp/cal-a.ms"]

    monkeypatch.setattr(cli_products, "_run_calibrate_jobs", _fake_run)

    result = runner.invoke(cli.app, ["products", "calibrate", "--asdm-uid", "uid___A,uid___B"])

    assert result.exit_code == 1
    assert "Calibrated MS products: 1" in result.output
    assert "1 UID(s) failed and were skipped" in result.output
    assert "uid___B" in result.output
    assert "/logs/b.log" in result.output


def test_products_calibrate_fail_fast_flag_passthrough(monkeypatch):
    """--fail-fast should disable continue_on_error."""
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(cli_products, "_run_calibrate_jobs", _fake_run)
    result = runner.invoke(cli.app, ["products", "calibrate", "--fail-fast"])

    assert result.exit_code == 0
    assert captured["continue_on_error"] is False


def test_run_parallel_archive_jobs_skips_calibration_for_failed_unpack(monkeypatch, tmp_path):
    """UIDs whose unpack failed must not be submitted to the calibrate stage."""

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(cli_products, "create_backend", lambda *a, **k: _Backend())
    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa")
    monkeypatch.setattr(
        cli_products, "_extract_asdm_uids_from_download_root", lambda *_: ["uid___A", "uid___B"]
    )

    stages: list[tuple[str, list[str]]] = []

    def _fake_stage(**kwargs):
        stages.append((kwargs["stage_label"], list(kwargs["job_uids"])))
        if kwargs["stage_label"].endswith("unpack"):
            kwargs["failures"].append(cli_products.StageFailure("uid___B", "importasdm crashed"))
            return [["/raw/uid___A.ms"], []]
        return [["/cal/uid___A.ms.split.cal"]]

    monkeypatch.setattr(cli_products, "_run_uid_stage", _fake_stage)

    failures: list[cli_products.StageFailure] = []
    raw, cal = cli_products._run_parallel_archive_jobs(
        download_root=tmp_path / "dl",
        archive_output_root=tmp_path / "archive",
        unpack_ms=True,
        generate_calibrated_visibilities=True,
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=False,
        overwrite_archive_outputs=False,
        failures=failures,
    )

    assert raw == ["/raw/uid___A.ms"]
    assert cal == ["/cal/uid___A.ms.split.cal"]
    assert stages == [("Slurm unpack", ["uid___A", "uid___B"]), ("Slurm calibrate", ["uid___A"])]
    assert [failure.uid for failure in failures] == ["uid___B"]


def test_tail_log_line_ignores_stale_logs(tmp_path):
    """Logs older than the stage start are not reported as running jobs."""
    from time import time

    log_path = tmp_path / "old.log"
    log_path.write_text("first\nlast line\n", encoding="utf-8")
    os.utime(log_path, (time() - 3600, time() - 3600))

    assert cli_products._tail_log_line(log_path) == "last line"
    assert cli_products._tail_log_line(log_path, modified_since=time() - 60) is None
    assert cli_products._tail_log_line(tmp_path / "missing.log") is None


# ---------------------------------------------------------------------------
# Completed-calibration detection and working-copy handling
# ---------------------------------------------------------------------------


def test_partition_completed_calibrations_skips_marked_outputs(tmp_path):
    """Only outputs with a completion marker are treated as done."""
    from almasim.services.archive import write_calibration_marker

    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    write_calibration_marker(output_root, "uid___A")
    (output_root / "uid___B.ms.split.cal").mkdir()  # partial: no marker
    (output_root / "uid___C.ms.split.cal.done").touch()  # marker without output

    todo, done = cli_products._partition_completed_calibrations(
        output_root, ["uid___A", "uid___B", "uid___C", "uid___D"], overwrite=False
    )

    assert todo == ["uid___B", "uid___C", "uid___D"]
    assert done == [str(output_root / "uid___A.ms.split.cal")]

    todo, done = cli_products._partition_completed_calibrations(
        output_root, ["uid___A", "uid___B"], overwrite=True
    )
    assert todo == ["uid___A", "uid___B"]
    assert done == []


def test_run_calibrate_jobs_skips_completed_uids_before_submission(monkeypatch, tmp_path):
    """Completed UIDs are neither preflighted nor submitted, but appear in the outputs."""
    from almasim.services.archive import write_calibration_marker

    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    write_calibration_marker(output_root, "uid___A")

    submitted: list[str] = []

    def _fake_single(**kwargs):
        submitted.append(kwargs["asdm_uid"])
        assert kwargs["keep_working_copies"] is False
        return [f"{kwargs['calibrated_output_root']}/{kwargs['asdm_uid']}.ms.split.cal"]

    monkeypatch.setattr(cli_products, "_calibrate_single_uid", _fake_single)
    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa")

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=output_root,
        asdm_uids=["uid___A", "uid___B", "uid___C"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert submitted == ["uid___B", "uid___C"]
    assert outputs[0] == str(output_root / "uid___A.ms.split.cal")
    assert len(outputs) == 3


def test_run_calibrate_jobs_all_completed_does_nothing(monkeypatch, tmp_path):
    """When every UID is done the runner returns the existing outputs without a backend."""
    from almasim.services.archive import write_calibration_marker

    output_root = tmp_path / "cal"
    for uid in ("uid___A", "uid___B"):
        (output_root / f"{uid}.ms.split.cal").mkdir(parents=True)
        write_calibration_marker(output_root, uid)
    monkeypatch.setattr(
        cli_products, "_preflight_casa_data", lambda *a, **k: pytest.fail("no preflight")
    )
    monkeypatch.setattr(
        cli_products, "create_backend", lambda *a, **k: pytest.fail("no backend expected")
    )

    outputs = cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=output_root,
        asdm_uids=["uid___A", "uid___B"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )
    assert len(outputs) == 2


def test_calibrate_single_uid_forwards_keep_working_copies(tmp_path, monkeypatch):
    """The worker wrapper should pass --keep-working-copies to the child only when asked."""
    commands: list[list[str]] = []

    class _FakeProcess:
        stdout = iter([])

        def wait(self):
            return 0

    monkeypatch.setattr(
        "subprocess.Popen", lambda cmd, **kwargs: commands.append(cmd) or _FakeProcess()
    )
    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    common = dict(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid="uid___A",
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )
    cli_products._calibrate_single_uid(**common)
    cli_products._calibrate_single_uid(**common, keep_working_copies=True)

    assert "--keep-working-copies" not in commands[0]
    assert "--keep-working-copies" in commands[1]


def test_products_calibrate_keep_working_copies_passthrough(monkeypatch):
    """--keep-working-copies reaches the job runner."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_products, "_run_calibrate_jobs", lambda **kwargs: captured.update(kwargs) or []
    )

    assert runner.invoke(cli.app, ["products", "calibrate"]).exit_code == 0
    assert captured["keep_working_copies"] is False
    assert runner.invoke(cli.app, ["products", "calibrate", "--keep-working-copies"]).exit_code == 0
    assert captured["keep_working_copies"] is True


def test_casa_bundled_lib_dir_precedes_system_paths(tmp_path, monkeypatch):
    """Worker env must search casatools' bundled libs before /usr/lib64.

    The system libpmix otherwise shadows the copy bundled with casatools and the
    CASA extension modules fail with "undefined symbol: pmix_framework_names".
    """

    class _FakeProcess:
        stdout = iter([])

        def wait(self):
            return 0

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda cmd, **kwargs: captured.update(kwargs) or _FakeProcess(),
    )
    bundled = tmp_path / "site-packages" / "casatools" / "__casac__" / "lib"
    bundled.mkdir(parents=True)
    monkeypatch.setattr(cli_products, "_casa_bundled_lib_dir", lambda: str(bundled))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/spack/lib")

    output_root = tmp_path / "cal"
    (output_root / "uid___A.ms.split.cal").mkdir(parents=True)
    cli_products._calibrate_single_uid(
        input_root="/input",
        raw_ms_root="/raw",
        calibrated_output_root=str(output_root),
        asdm_uid="uid___A",
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
        clean_intermediate=False,
    )

    entries = captured["env"]["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == str(bundled)
    assert entries.index(str(bundled)) < entries.index("/usr/lib64")
    # Inherited (spack) paths still come last.
    assert entries[-1] == "/opt/spack/lib"


def test_casa_bundled_lib_dir_absent_keeps_system_paths(tmp_path, monkeypatch):
    """Without casatools installed the previous system-first ordering is unchanged."""

    class _FakeProcess:
        stdout = iter([])

        def wait(self):
            return 0

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "subprocess.Popen", lambda cmd, **kwargs: captured.update(kwargs) or _FakeProcess()
    )
    monkeypatch.setattr(cli_products, "_casa_bundled_lib_dir", lambda: None)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    output_root = tmp_path / "raw"
    (output_root / "working").mkdir(parents=True)
    cli_products._unpack_single_uid(
        input_root="/input",
        raw_output_root=str(output_root),
        asdm_uid="uid___A",
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite=False,
    )

    assert captured["env"]["LD_LIBRARY_PATH"].startswith("/lib64:/usr/lib64:")


def test_casa_bundled_lib_dir_returns_none_when_not_installed(monkeypatch):
    """A missing casatools package resolves to None rather than raising."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert cli_products._casa_bundled_lib_dir() is None


def test_one_uid_per_worker_limits_task_slots():
    """Slurm workers get a single task slot so --slurm-workers == UIDs in flight."""
    assert cli_products._one_uid_per_worker({"n_workers": 10}) == {
        "n_workers": 10,
        "worker_extra_args": ["--nthreads", "1"],
    }
    # An explicit caller setting wins.
    explicit = {"worker_extra_args": ["--nthreads", "4"]}
    assert cli_products._one_uid_per_worker(explicit) == explicit


def test_run_calibrate_jobs_slurm_pins_one_task_per_worker(monkeypatch, tmp_path):
    """The slurm calibrate path must not let a worker run many UIDs at once."""
    captured: dict[str, object] = {}

    class _Backend:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(
        cli_products,
        "create_backend",
        lambda backend, **kwargs: captured.update(kwargs) or _Backend(),
    )
    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa")
    monkeypatch.setattr(cli_products, "_run_uid_stage", lambda **kwargs: [["/cal/a.ms.split.cal"]])

    cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A"],
        postprocess_backend="slurm",
        postprocess_backend_kwargs={"n_workers": 10},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )

    assert captured["worker_extra_args"] == ["--nthreads", "1"]


def test_run_calibrate_jobs_sync_does_not_add_worker_args(monkeypatch, tmp_path):
    """The sync path has no Dask workers, so no worker arguments are injected."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli_products, "_preflight_casa_data", lambda *a, **k: tmp_path / "casa")
    monkeypatch.setattr(
        cli_products, "_run_uid_stage", lambda **kwargs: captured.update(kwargs) or [[], []]
    )

    cli_products._run_calibrate_jobs(
        input_root=tmp_path / "input",
        raw_ms_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        asdm_uids=["uid___A", "uid___B"],
        postprocess_backend="sync",
        postprocess_backend_kwargs={},
        casa_data_root=None,
        skip_casa_data_update=True,
        overwrite_outputs=False,
        clean_intermediate=False,
    )
    assert captured["postprocess_backend"] == "sync"


# ---------------------------------------------------------------------------
# A hard abort in the CASA child (casacore terminate(), OOM kill) must still
# leave a failure marker behind: the in-process ``finally`` never ran, so the
# subprocess wrapper is the only place that can write it.
# ---------------------------------------------------------------------------


class _CrashingProcess:
    """Child that prints a casacore abort and dies with SIGABRT (-6)."""

    def __init__(self, lines: list[str] | None = None, return_code: int = -6):
        self.stdout = iter(
            lines
            or [
                "Extracting calibration tables\n",
                "terminate called after throwing an instance of 'casacore::ArrayError'\n",
                "  what():  minMax - Array has no elements\n",
            ]
        )
        self._rc = return_code

    def wait(self):
        return self._rc


def test_calibrate_single_uid_hard_abort_writes_failure_marker(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _CrashingProcess())
    uid = "uid___A001_X1_X9"
    output_root = tmp_path / "calibrated"
    working_dir = output_root / "working" / f"{uid}.calibration"
    working_dir.mkdir(parents=True)
    (working_dir / f"{uid}.ms").mkdir()

    with pytest.raises(RuntimeError, match="Return code: -6"):
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

    marker = output_root / f"{uid}.ms.split.cal.failed"
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert payload["uid"] == uid
    assert "return code -6" in payload["error"]
    assert "minMax - Array has no elements" in payload["error"]
    assert Path(payload["log"]).is_file()
    assert not (output_root / f"{uid}.ms.split.cal.done").exists()
    assert not working_dir.exists(), "the multi-GB working copy is reclaimed"


def test_calibrate_single_uid_hard_abort_keeps_working_copy_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _CrashingProcess())
    uid = "uid___A001_X1_Xa"
    output_root = tmp_path / "calibrated"
    working_dir = output_root / "working" / f"{uid}.calibration"
    working_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError):
        cli_products._calibrate_single_uid(
            input_root="/input",
            raw_ms_root="/raw",
            calibrated_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
            clean_intermediate=False,
            keep_working_copies=True,
        )

    assert (output_root / f"{uid}.ms.split.cal.failed").is_file()
    assert working_dir.is_dir()


def test_unpack_single_uid_hard_abort_writes_failure_marker(tmp_path, monkeypatch):
    import json

    lines = ["importasdm::::casa\tstarting\n", "Killed\n"]
    monkeypatch.setattr(
        "subprocess.Popen", lambda *args, **kwargs: _CrashingProcess(lines, return_code=-9)
    )
    uid = "uid___A001_X1_Xb"
    output_root = tmp_path / "raw"
    (output_root / "working").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Return code: -9"):
        cli_products._unpack_single_uid(
            input_root="/input",
            raw_output_root=str(output_root),
            asdm_uid=uid,
            casa_data_root=None,
            skip_casa_data_update=True,
            overwrite=False,
        )

    marker = output_root / "working" / f"{uid}.ms.failed"
    assert marker.is_file()
    payload = json.loads(marker.read_text())
    assert "return code -9" in payload["error"]
    assert not (output_root / "working" / f"{uid}.ms.done").exists()
