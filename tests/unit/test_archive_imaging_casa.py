"""casacore steps of the imaging task, run against a stand-in module.

CI has no python-casacore; the fake records what the code asks of it, so the
MS reading and writing is exercised exactly as on the cluster.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from almasim.services.imaging import archive_imaging as ai


class _FakeTable:
    """The subset of ``casacore.tables.table`` the imaging code uses."""

    def __init__(self, columns: dict[str, np.ndarray]):
        self.columns = columns
        self.closed = False

    def colnames(self):
        return list(self.columns)

    def nrows(self):
        return len(next(iter(self.columns.values())))

    def iscelldefined(self, name, row):
        return name in self.columns

    def getcell(self, name, row):
        return self.columns[name][row]

    def getcol(self, name, startrow=0, nrow=-1):
        column = self.columns[name]
        return column[startrow:] if nrow < 0 else column[startrow : startrow + nrow]

    def putcol(self, name, value, startrow=0, nrow=-1):
        self.columns[name][startrow : startrow + len(value)] = value

    def getcoldesc(self, name):
        return {"source": name}

    def addcols(self, desc):
        self.columns[desc["name"]] = np.zeros_like(self.columns[desc["source"]])

    def removecols(self, name):
        del self.columns[name]

    def close(self):
        self.closed = True


@pytest.fixture
def fake_casacore(monkeypatch):
    """``casacore.tables`` serving the tables registered in the returned dict by path."""
    tables: dict[str, _FakeTable] = {}
    package = ModuleType("casacore")
    module = ModuleType("casacore.tables")
    module.table = lambda path, **kwargs: tables[str(path)]
    module.makecoldesc = lambda name, desc: {"name": name, **desc}
    package.tables = module
    monkeypatch.setitem(sys.modules, "casacore", package)
    monkeypatch.setitem(sys.modules, "casacore.tables", module)
    return tables


def _visibilities(n_rows=40, n_channels=64, line=slice(30, 36), seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n_channels)[None, :, None]
    continuum = (2.0 + 0.3 * x) * np.ones((n_rows, 1, 2))
    data = (continuum + 0.01 * rng.normal(size=(n_rows, n_channels, 2))).astype(np.complex64)
    data[:, line] += 1.5
    return data


def _ms(data, *, spectrum_weight=False, corrected=False):
    columns = {
        "DATA": data,
        "FLAG": np.zeros(data.shape, bool),
        "WEIGHT": np.ones((data.shape[0], data.shape[2])),
    }
    if spectrum_weight:
        columns["WEIGHT_SPECTRUM"] = np.ones(data.shape)
    if corrected:
        columns["CORRECTED_DATA"] = data.copy()
    return _FakeTable(columns)


@pytest.mark.unit
@pytest.mark.parametrize(("spectrum_weight", "corrected"), [(False, True), (True, False)])
def test_subtract_uv_continuum_writes_continuum_and_line_columns(
    fake_casacore, spectrum_weight, corrected
):
    data = _visibilities()
    fake_casacore["x.ms"] = ms = _ms(data, spectrum_weight=spectrum_weight, corrected=corrected)
    ms.columns["FLAG"][:, 5] = True  # a flagged channel

    result = ai.subtract_uv_continuum("x.ms", rows_per_chunk=9)

    assert result["source_column"] == ("CORRECTED_DATA" if corrected else "DATA")
    assert not result.get("skipped") and ms.closed and result["usable_channels"] == 63
    ranges = [tuple(map(int, r.split("~"))) for r in result["line_ranges"].split(";")]
    assert all(any(a <= c <= b for a, b in ranges) for c in range(30, 36))
    line = ms.columns[ai.LINE_COLUMN]
    assert np.abs(line[:, 30:36].real - 1.5).max() < 0.05, "the line stays in LINE_DATA"
    assert np.abs(line[:, :20]).max() < 0.05, "no continuum left"
    np.testing.assert_allclose(ms.columns[ai.CONTINUUM_COLUMN] + line, data, atol=1e-5)

    # A rerun replaces the columns it made.
    assert not ai.subtract_uv_continuum("x.ms").get("skipped")


@pytest.mark.unit
def test_subtract_uv_continuum_skips_without_rows_or_line_free_channels(fake_casacore):
    fake_casacore["empty.ms"] = _ms(_visibilities()[:0])
    assert ai.subtract_uv_continuum("empty.ms")["skipped"] == "the MS has no rows"

    fake_casacore["lines.ms"] = ms = _ms(_visibilities())
    result = ai.subtract_uv_continuum("lines.ms", min_line_free_fraction=0.99)
    assert "usable channels are line-free" in result["skipped"]
    assert ai.LINE_COLUMN not in ms.columns


@pytest.mark.unit
def test_line_free_channels_with_too_few_channels_to_fit():
    line_free = ai.line_free_channels(np.array([1.0, 2.0]), np.array([True, False]), order=1)
    assert list(line_free) == [True, False], "one usable channel cannot be fitted or clipped"


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.mark.unit
def test_subtract_uv_continuum_subprocess_returns_the_child_result(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(args=json.loads(cmd[-1]), env=kwargs["env"])
        return _Proc(stdout='{"line_channels": 3}\n', stderr="casacore note")

    monkeypatch.setattr(ai.subprocess, "run", fake_run)
    log = tmp_path / "task.log"
    result = ai.subtract_uv_continuum_subprocess("x.ms", order=0, log_path=log)

    assert result == {"line_channels": 3} and seen["args"] == {"ms": "x.ms", "order": 0}
    assert seen["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert log.read_text() == "casacore note\n"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (_Proc(returncode=1, stderr="Traceback\nRuntimeError: lock\n"), "return code 1: .*lock"),
        (_Proc(stdout=""), "wrote no result"),
        (subprocess.TimeoutExpired("python", 5), "killed after exceeding the 5s"),
    ],
)
def test_subtract_uv_continuum_subprocess_failures(monkeypatch, outcome, message):
    def fake_run(cmd, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(ai.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=message):
        ai.subtract_uv_continuum_subprocess("x.ms", timeout=5)


@pytest.mark.unit
def test_run_wsclean_task_fails_when_the_uv_subtraction_fails(tmp_path, monkeypatch):
    def extract(ms, spw, fields, out_path, **kwargs):
        (Path(out_path) / "table.dat").parent.mkdir(parents=True)
        (Path(out_path) / "table.dat").write_text("x")
        return {"rows": 1, "bytes": 0}

    monkeypatch.setattr(ai, "extract_single_window_ms_subprocess", extract)

    def broken(*args, **kwargs):
        raise RuntimeError("casacore lock")

    monkeypatch.setattr(ai, "subtract_uv_continuum_subprocess", broken)
    out = tmp_path / "out" / "SPW-25"
    with pytest.raises(RuntimeError, match="Could not subtract the continuum.*casacore lock"):
        ai.run_wsclean_task(
            command=["wsclean", "-name", str(out / "wsclean"), ai.SINGLE_WINDOW_PLACEHOLDER],
            cube_command=[
                "wsclean",
                "-name",
                str(out / "wsclean-cube"),
                ai.SINGLE_WINDOW_PLACEHOLDER,
            ],
            output_dir=str(out),
            ms_path=str(tmp_path / "x.ms"),
            spw=25,
            threads=2,
            single_window=True,
            uv_contsub_order=1,
        )
    marker = json.loads(ai.imaging_failure_marker_path(out).read_text())
    assert "casacore lock" in marker["error"]
    assert not list(out.glob("spw*.single.ms")), "the single-window copy is removed"


@pytest.mark.unit
def test_subtract_uv_continuum_ignores_rows_flagged_by_flag_row(fake_casacore):
    data = _visibilities()
    data[::5, 10:20] += 100.0  # garbage in rows the observatory flagged as a whole
    fake_casacore["x.ms"] = ms = _ms(data)
    ms.columns["FLAG_ROW"] = np.zeros(data.shape[0], bool)
    ms.columns["FLAG_ROW"][::5] = True

    result = ai.subtract_uv_continuum("x.ms")

    ranges = [tuple(map(int, r.split("~"))) for r in result["line_ranges"].split(";")]
    assert not any(a <= 15 <= b for a, b in ranges), "flagged rows never make a line"
    assert any(a <= 32 <= b for a, b in ranges), "the real line is still found"


@pytest.mark.unit
def test_subtract_uv_continuum_reads_chunks_sized_by_samples(fake_casacore, monkeypatch):
    fake_casacore["x.ms"] = ms = _ms(_visibilities(n_rows=40, n_channels=64))
    reads = []
    getcol = ms.getcol
    monkeypatch.setattr(
        ms,
        "getcol",
        lambda name, startrow=0, nrow=-1: reads.append(nrow) or getcol(name, startrow, nrow),
    )
    monkeypatch.setattr(ai, "_CHUNK_SAMPLES", 64 * 2 * 10)  # ten rows of 64 channels x 2

    ai.subtract_uv_continuum("x.ms")

    assert max(reads) == 10


@pytest.mark.unit
def test_rows_for_chunk():
    assert ai.rows_for_chunk(3840, 2) == 2604
    assert ai.rows_for_chunk(10**9, 4) == 1, "at least one row"
