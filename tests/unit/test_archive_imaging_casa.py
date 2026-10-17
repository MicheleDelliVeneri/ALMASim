"""casacore and CASA steps of the imaging task, run against stand-in modules.

CI has neither python-casacore nor casatools; these fakes record what the
code asks of them, so the MS reading and writing and the CASA calls are
exercised exactly as on the cluster.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from astropy.io import fits

from almasim import cli_image
from almasim.services.archive import unpack_ms
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

    def getcol(self, name, startrow=0, nrow=-1):
        column = self.columns[name]
        return column[startrow:] if nrow < 0 else column[startrow : startrow + nrow]

    def putcol(self, name, value, startrow=0, nrow=-1):
        self.columns[name][startrow : startrow + len(value)] = value

    def getcoldesc(self, name):
        return {"source": name}

    def addcols(self, desc):
        source = self.columns[desc["source"]]
        self.columns[desc["name"]] = np.zeros_like(source)

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
    n_rows = data.shape[0]
    a1, a2 = np.triu_indices(5, 1)
    columns = {
        "DATA": data,
        "FLAG": np.zeros(data.shape, bool),
        "WEIGHT": np.ones((n_rows, data.shape[2])),
        "FLAG_ROW": np.zeros(n_rows, bool),
        "ANTENNA1": np.resize(a1, n_rows),
        "ANTENNA2": np.resize(a2, n_rows),
    }
    if spectrum_weight:
        columns["WEIGHT_SPECTRUM"] = np.ones(data.shape)
    if corrected:
        columns["CORRECTED_DATA"] = data.copy()
    return _FakeTable(columns)


@pytest.mark.unit
@pytest.mark.parametrize("spectrum_weight", [False, True])
def test_source_spectrum_averages_every_row(spectrum_weight):
    data = _visibilities()
    ms = _ms(data, spectrum_weight=spectrum_weight)
    ms.columns["FLAG"][:, 5] = True  # a flagged channel

    spectrum, usable = ai._source_spectrum(ms, "DATA", rows_per_chunk=7)

    assert not usable[5] and usable.sum() == 63
    np.testing.assert_allclose(spectrum[usable], np.abs(data.mean(axis=(0, 2)))[usable], rtol=1e-5)
    assert ai._source_spectrum(_FakeTable({"DATA": data[:0]}), "DATA", 7) is None


@pytest.mark.unit
def test_subtract_uv_continuum_writes_continuum_and_line_columns(fake_casacore):
    data = _visibilities()
    fake_casacore["x.ms"] = ms = _ms(data, corrected=True)

    result = ai.subtract_uv_continuum("x.ms", rows_per_chunk=9)

    assert result["source_column"] == "CORRECTED_DATA" and not result.get("skipped")
    ranges = [tuple(map(int, r.split("~"))) for r in result["line_ranges"].split(";")]
    assert all(any(a <= c <= b for a, b in ranges) for c in range(30, 36)) and ms.closed
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
def test_drop_corrected_data(fake_casacore):
    fake_casacore["c.ms"] = ms = _ms(_visibilities(), corrected=True)
    assert ai.drop_corrected_data("c.ms") == {"dropped": True}
    assert "CORRECTED_DATA" not in ms.columns
    assert ai.drop_corrected_data("c.ms") == {"dropped": False}


@pytest.mark.unit
def test_selfcal_prepare_selects_line_free_channels_and_the_busiest_antenna(fake_casacore):
    fake_casacore["p.ms"] = ms = _ms(_visibilities())
    ms.columns["FLAG_ROW"][ms.columns["ANTENNA1"] == 0] = True  # antenna 0 mostly flagged
    fake_casacore["p.ms/ANTENNA"] = _FakeTable(
        {"NAME": np.array(["DA41", "DA42", "DV01", "DV02", "PM03"])}
    )

    prepared = ai.selfcal_prepare("p.ms")

    assert prepared["spw"].startswith("0:0~") and "~63" in prepared["spw"]
    assert prepared["line_free_channels"] < 64 and prepared["refant"] != "DA41"
    with pytest.raises(RuntimeError, match="no rows"):
        fake_casacore["e.ms"] = _ms(_visibilities()[:0])
        ai.selfcal_prepare("e.ms")


@pytest.fixture
def fake_casa(monkeypatch):
    """``casatasks`` and ``casatools`` recording the calls, and a caltable to read back."""
    calls: dict[str, list] = {"gaincal": [], "applycal": [], "configure": []}
    tasks = ModuleType("casatasks")
    tasks.gaincal = lambda **kwargs: calls["gaincal"].append(kwargs)
    tasks.applycal = lambda **kwargs: calls["applycal"].append(kwargs)
    tools = ModuleType("casatools")

    class _CalTable:
        def open(self, path):
            self.path = path

        def getcol(self, name):
            gains = np.array([[[np.exp(0.2j), 1.1 * np.exp(-0.1j), 1.0]]])
            return {
                "FLAG": np.array([[[False, False, True]]]),
                "CPARAM": gains,
                "TIME": np.array([1.0, 1.0, 2.0]),
            }[name]

        def close(self):
            pass

    tools.table = _CalTable
    monkeypatch.setitem(sys.modules, "casatasks", tasks)
    monkeypatch.setitem(sys.modules, "casatools", tools)
    monkeypatch.setattr(
        unpack_ms,
        "configure_casa_environment",
        lambda root, data, **kwargs: calls["configure"].append((data, kwargs["log_file"])),
    )
    return calls


@pytest.mark.unit
def test_casa_selfcal_solve_phase_and_amplitude(fake_casa):
    phase = ai.casa_selfcal_solve(
        "x.ms", "x.G", solint="inf", spw="0:0~9", refant="DA41", casa_data="/d", log_file="c.log"
    )
    (call,) = fake_casa["gaincal"]
    assert call["calmode"] == "p" and call["gaintype"] == "T" and not call["solnorm"]
    assert call["spw"] == "0:0~9" and call["gaintable"] == [] and call["interp"] == []
    assert fake_casa["configure"] == [("/d", "c.log")]
    assert phase["solutions"] == 2 and phase["possible"] == 3 and phase["intervals"] == 2
    assert phase["median_phase_deg"] == pytest.approx(8.6, abs=0.1)

    amp = ai.casa_selfcal_solve(
        "x.ms",
        "x.A",
        solint="inf",
        spw="0:0~9",
        refant="DA41",
        casa_data="/d",
        log_file="c.log",
        calmode="ap",
        pre_apply=["x.G"],
    )
    call = fake_casa["gaincal"][-1]
    assert call["calmode"] == "ap" and call["solnorm"] and call["gaintable"] == ["x.G"]
    assert call["interp"] == ["linear"] and amp["median_amp_dev"] == pytest.approx(0.05)


@pytest.mark.unit
def test_casa_selfcal_apply_both_tables(fake_casa):
    assert ai.casa_selfcal_apply("x.ms", ["x.G", "x.A"], casa_data="/d", log_file="c.log") == {
        "applied": True
    }
    (call,) = fake_casa["applycal"]
    assert call["gaintable"] == ["x.G", "x.A"] and call["calwt"] == [False, False]
    assert call["applymode"] == "calonly" and call["interp"] == ["linear", "linear"]


def _fits(path: Path, data: np.ndarray) -> Path:
    fits.PrimaryHDU(data.astype(np.float32)).writeto(path)
    return path


@pytest.mark.unit
def test_image_snr_and_peak(tmp_path):
    rng = np.random.default_rng(4)
    residual = rng.normal(0, 0.1, (1, 1, 64, 64))
    image = residual.copy()
    image[0, 0, 32, 32] = 5.0
    image_path = _fits(tmp_path / "i.fits", image)
    snr = ai.image_snr(image_path, _fits(tmp_path / "r.fits", residual))
    assert snr == pytest.approx(50, rel=0.1)
    assert ai.image_peak(image_path) == pytest.approx(5.0)
    assert ai.image_snr(image_path, _fits(tmp_path / "z.fits", np.zeros((4, 4)))) == 0.0


@pytest.mark.unit
def test_call_in_child_runs_a_function_of_the_module(tmp_path):
    image = _fits(tmp_path / "i.fits", np.full((4, 4), 2.5))
    assert ai._call_in_child("image_peak", {"image_path": str(image)}, what="peak") == 2.5
    assert ai._call_in_child("image_peak", {"image_path": str(image)}, what="p", casa=True) == 2.5


@pytest.mark.unit
def test_call_in_child_reports_failures_and_timeouts(tmp_path):
    log = tmp_path / "task.log"
    with pytest.raises(RuntimeError, match="peak exited with return code 1: .*FileNotFound"):
        ai._call_in_child(
            "image_peak", {"image_path": str(tmp_path / "missing.fits")}, what="peak", log_path=log
        )
    assert "Traceback" in log.read_text(), "the child's stderr goes to the task log"
    with pytest.raises(RuntimeError, match="peak killed after exceeding"):
        ai._call_in_child("image_peak", {"image_path": "x"}, what="peak", timeout=0.001)


@pytest.mark.unit
def test_call_in_child_needs_a_result(monkeypatch):
    class _Proc:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(ai.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="wrote no result"):
        ai._call_in_child("image_peak", {}, what="peak")


@pytest.mark.unit
def test_self_calibrate_turns_a_failed_step_into_a_task_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(ai, "image_snr", lambda image, residual: 50.0)
    monkeypatch.setattr(ai, "image_peak", lambda image: 1.0)

    def broken(function, kwargs, **_):
        raise RuntimeError("gaincal exploded")

    monkeypatch.setattr(ai, "_call_in_child", broken)
    with (tmp_path / "task.log").open("w") as log, pytest.raises(ValueError, match="exploded"):
        ai.self_calibrate(
            tmp_path / "x.ms",
            ["wsclean", "-name", str(tmp_path / "w"), "ms"],
            out=tmp_path,
            log=log,
            solints=["inf"],
            min_snr=20,
            run=lambda cmd: 1,
            fail=ValueError,
            casa_data="/d",
        )


class _Stop(Exception):
    """Ends a task once the step under test has run."""


def _wsclean_task(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(
        ai, "extract_single_window_ms_subprocess", lambda *a, **k: {"rows": 1, "bytes": 0}
    )
    out = tmp_path / "out" / "SPW-25"
    return ai.run_wsclean_task(
        command=["wsclean", "-name", str(out / "wsclean"), ai.SINGLE_WINDOW_PLACEHOLDER],
        cube_command=["wsclean", "-name", str(out / "wsclean-cube"), ai.SINGLE_WINDOW_PLACEHOLDER],
        output_dir=str(out),
        ms_path=str(tmp_path / "x.ms"),
        spw=25,
        threads=2,
        single_window=True,
        **kwargs,
    )


@pytest.mark.unit
def test_run_wsclean_task_self_calibrates_before_imaging(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ai, "_run_with_retries", lambda cmd, *args: seen.setdefault("ran", cmd))

    def fake_self_calibrate(ms, command, **kwargs):
        seen.update(kwargs, ms=ms, command=command)
        kwargs["run"](["wsclean", "-name", "probe"])
        raise _Stop

    monkeypatch.setattr(ai, "self_calibrate", fake_self_calibrate)
    with pytest.raises(_Stop):
        _wsclean_task(tmp_path, monkeypatch, selfcal_solints=["inf"], casa_data="/casa")

    assert Path(seen["ms"]).name == "spw25.single.ms" and seen["casa_data"] == "/casa"
    assert seen["solints"] == ["inf"] and seen["command"][-1].endswith("spw25.single.ms")
    assert seen["ran"] == ["wsclean", "-name", "probe"], "rounds image through the task runner"


@pytest.mark.unit
def test_run_wsclean_task_selfcal_needs_casa_data():
    with pytest.raises(ValueError, match="casa_data"):
        ai.run_wsclean_task(
            command=["wsclean"],
            output_dir="x",
            ms_path="x",
            spw=1,
            threads=1,
            single_window=True,
            selfcal_solints=["inf"],
        )


@pytest.mark.unit
def test_run_wsclean_task_fails_when_the_uv_subtraction_fails(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("casacore lock")

    monkeypatch.setattr(ai, "subtract_uv_continuum_subprocess", broken)
    with pytest.raises(RuntimeError, match="Could not subtract the continuum.*casacore lock"):
        _wsclean_task(tmp_path, monkeypatch, uv_contsub_order=1)
    (failed,) = (tmp_path / "out").rglob("*.failed")
    assert "casacore lock" in json.loads(failed.read_text())["error"]


@pytest.mark.unit
def test_prepare_selfcal_casa_data_populates_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        unpack_ms,
        "configure_casa_environment",
        lambda root, data, **kwargs: calls.append(("configure", Path(data))),
    )
    monkeypatch.setattr(
        unpack_ms,
        "ensure_casa_runtime_data",
        lambda data, **kwargs: calls.append(("ensure", Path(data))),
    )

    default = cli_image._prepare_selfcal_casa_data(tmp_path / "out", None)
    chosen = cli_image._prepare_selfcal_casa_data(tmp_path / "out", tmp_path / "casa")

    assert default == tmp_path / "out" / ".casa-data" and chosen == tmp_path / "casa"
    assert calls == [
        ("configure", default),
        ("ensure", default),
        ("configure", chosen),
        ("ensure", chosen),
    ]
