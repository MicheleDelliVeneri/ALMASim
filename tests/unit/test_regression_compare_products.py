"""Tests for scripts/regression/compare_products.py (MADIV-28 regression gate)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "regression" / "compare_products.py"


@pytest.fixture(scope="module")
def cp():
    spec = importlib.util.spec_from_file_location("compare_products", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_products"] = module
    spec.loader.exec_module(module)
    return module


def _write_task(root: Path, cube: np.ndarray, quality: dict, log_root: str) -> Path:
    root.mkdir(parents=True)
    header = fits.Header()
    header["BUNIT"] = "JY/BEAM"
    header["DATE"] = "2026-10-19T00:00:00"
    header["ORIGIN"] = f"written under {log_root}"
    beams = fits.BinTableHDU.from_columns(
        [fits.Column(name="BMAJ", format="E", array=np.array([1.0, 1.1], dtype=np.float32))],
        name="BEAMS",
    )
    fits.HDUList([fits.PrimaryHDU(cube, header), beams]).writeto(root / "wsclean-cube-image.fits")
    (root / "quality.json").write_text(json.dumps(quality))
    (root / "gains.csv").write_text(f"time,phase\n1,{log_root}\n")
    (root / "casa.log").write_text("timestamped noise that is never compared\n")
    return root


@pytest.fixture
def pair(tmp_path):
    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)
    cube[0, 0, 0, 0] = np.nan
    quality = {"continuum": {"snr": 44.5}, "completed_at": "now", "path": "/ref/x"}
    ref = _write_task(tmp_path / "ref", cube, quality, "/ref")
    new = _write_task(
        tmp_path / "new",
        cube.copy(),
        {**quality, "completed_at": "later", "path": "/run/x"},
        "/run",
    )
    return new, ref


def _compare(cp, new, ref, **kwargs):
    report = cp.Report()
    cp.compare_tree(report, new, ref, cp.PathMap([("ROOT", "/run", "/ref")]), **kwargs)
    return report


def test_identical_products_pass(cp, pair):
    report = _compare(cp, *pair)
    assert report.passed, report.render()


def test_without_path_mapping_paths_count_as_differences(cp, pair):
    report = cp.Report()
    cp.compare_tree(report, *pair, cp.PathMap())
    assert not report.passed


def test_single_ulp_pixel_change_fails(cp, pair):
    new, ref = pair
    with fits.open(new / "wsclean-cube-image.fits", mode="update") as hdus:
        hdus[0].data[0, 1, 2, 3] = np.nextafter(hdus[0].data[0, 1, 2, 3], np.inf)
    report = _compare(cp, new, ref)
    failed = [name for ok, name, _ in report.lines if not ok]
    assert failed == ["wsclean-cube-image.fits[0] data"]


def test_tolerance_accepts_small_float_differences(cp, pair):
    new, ref = pair
    with fits.open(new / "wsclean-cube-image.fits", mode="update") as hdus:
        hdus[0].data[0, 1, 2, 3] += 1e-4
    assert not _compare(cp, new, ref).passed
    assert _compare(cp, new, ref, tolerance=1e-3).passed


def test_nan_positions_must_match(cp, pair):
    new, ref = pair
    with fits.open(new / "wsclean-cube-image.fits", mode="update") as hdus:
        hdus[0].data[0, 1, 1, 1] = np.nan
    report = _compare(cp, new, ref, tolerance=1e6)
    assert not report.passed
    assert "NaN positions differ" in report.render()


def test_beam_table_change_fails(cp, pair):
    new, ref = pair
    with fits.open(new / "wsclean-cube-image.fits", mode="update") as hdus:
        hdus["BEAMS"].data["BMAJ"][1] = 2.0
    failed = [name for ok, name, _ in _compare(cp, new, ref).lines if not ok]
    assert failed == ["wsclean-cube-image.fits[1] table"]


def test_json_value_change_names_the_field(cp, pair):
    new, ref = pair
    (new / "quality.json").write_text(
        json.dumps({"continuum": {"snr": 44.6}, "completed_at": "x", "path": "/run/x"})
    )
    report = _compare(cp, new, ref)
    assert not report.passed
    assert ".continuum.snr" in report.render()


def test_missing_and_extra_files_fail(cp, pair):
    new, ref = pair
    (new / "gains.csv").rename(new / "gains-renamed.csv")
    report = _compare(cp, new, ref)
    detail = next(d for ok, name, d in report.lines if name == "file set")
    assert "extra gains-renamed.csv" in detail
    assert "missing gains.csv" in detail


def test_log_annotations_ignore_durations_and_paths(cp, tmp_path):
    a = tmp_path / "a.log"
    b = tmp_path / "b.log"
    a.write_text("# extracted 376200 rows, 2.2 GB, 56.3 s\n# wsclean -name /run/x\nnoise 1\n")
    b.write_text("# extracted 376200 rows, 2.2 GB, 61.0 s\n# wsclean -name /ref/x\nnoise 2\n")
    names = cp.PathMap([("ROOT", "/run", "/ref")])
    assert cp.log_annotations(a, names) == cp.log_annotations(b, names)


def test_report_needs_at_least_one_check(cp):
    assert not cp.Report().passed


def test_casacore_table_digest_detects_changes(cp, tmp_path):
    tables = pytest.importorskip("casacore.tables")
    desc = tables.maketabdesc(
        [tables.makearrcoldesc("DATA", 0j, shape=[2, 3], valuetype="complex")]
    )
    for name in ("new", "ref"):
        tab = tables.table(str(tmp_path / name), desc, nrow=4, ack=False)
        tab.putcol("DATA", np.ones((4, 2, 3), dtype=np.complex64))
        tab.close()
    report = cp.Report()
    cp.compare_digests(
        report, "t", cp.table_digest(tmp_path / "new"), cp.table_digest(tmp_path / "ref")
    )
    assert report.passed, report.render()

    tab = tables.table(str(tmp_path / "new"), readonly=False, ack=False)
    tab.putcell("DATA", 3, np.full((2, 3), 1 + 1e-6j, dtype=np.complex64))
    tab.close()
    report = cp.Report()
    cp.compare_digests(
        report, "t", cp.table_digest(tmp_path / "new"), cp.table_digest(tmp_path / "ref")
    )
    assert [name for ok, name, _ in report.lines if not ok] == ["t column data"]


def test_casacore_variable_shape_column(cp, tmp_path, monkeypatch):
    tables = pytest.importorskip("casacore.tables")
    # Small chunks so the adaptive read sizing is exercised across many reads.
    monkeypatch.setattr(cp, "CHUNK_BYTES", 64)
    desc = tables.maketabdesc([tables.makearrcoldesc("DATA", 0.0, valuetype="double")])
    for name in ("new", "ref"):
        tab = tables.table(str(tmp_path / name), desc, nrow=50, ack=False)
        for row in range(50):
            tab.putcell("DATA", row, np.arange(row % 7 + 1, dtype=float))
        tab.close()
    new, ref = cp.table_digest(tmp_path / "new"), cp.table_digest(tmp_path / "ref")
    assert new == ref

    tab = tables.table(str(tmp_path / "new"), readonly=False, ack=False)
    tab.putcell("DATA", 49, np.arange(2, dtype=float))  # same values, different shape
    tab.close()
    assert cp.table_digest(tmp_path / "new") != ref
