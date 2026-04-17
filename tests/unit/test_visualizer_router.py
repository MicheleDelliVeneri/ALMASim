"""Tests for visualizer file discovery and integration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from fastapi.dependencies import utils as fastapi_dependency_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

_VISUALIZER_PATH = REPO_ROOT / "backend" / "app" / "api" / "v1" / "routers" / "visualizer.py"
_VISUALIZER_SPEC = importlib.util.spec_from_file_location("visualizer_router_test_module", _VISUALIZER_PATH)
assert _VISUALIZER_SPEC is not None and _VISUALIZER_SPEC.loader is not None
fastapi_dependency_utils.ensure_multipart_is_installed = lambda: None
visualizer = importlib.util.module_from_spec(_VISUALIZER_SPEC)
_VISUALIZER_SPEC.loader.exec_module(visualizer)


def test_list_visualizer_files_includes_npz_fits_and_msv2(tmp_path):
    np.savez(tmp_path / "cube.npz", clean_cube=np.ones((2, 3, 4), dtype=np.float32))
    fits.PrimaryHDU(np.ones((2, 3, 4), dtype=np.float32)).writeto(tmp_path / "cube.fits")
    (tmp_path / "example.ms").mkdir()

    listed = visualizer._iter_supported_paths(tmp_path)
    by_name = {path.name: visualizer._classify_visualizer_path(path) for path in listed}
    assert by_name["cube.npz"] == "npz"
    assert by_name["cube.fits"] == "fits"
    assert by_name["example.ms"] == "msv2"


def test_integrate_uploaded_fits_file(tmp_path):
    cube = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    fits_path = tmp_path / "cube.fits"
    fits.PrimaryHDU(cube).writeto(fits_path)

    response = visualizer._build_integrated_response(
        fits_path,
        "sum",
        channel_start=1,
        channel_end=1,
    )
    payload = json.loads(response.body)
    assert payload["stats"]["format"] == "fits"
    assert payload["stats"]["integration_axis"] == 0
    assert payload["stats"]["channel_start"] == 1
    assert payload["stats"]["channel_end"] == 1
    assert payload["stats"]["window_shape"] == [1, 4, 5]
    assert payload["stats"]["integrated_shape"] == [4, 5]
    assert len(payload["image"]) == 4
    assert len(payload["image"][0]) == 5


def test_integrate_server_side_msv2_path(monkeypatch, tmp_path):
    ms_path = tmp_path / "example.ms"
    ms_path.mkdir()

    def fake_load_msv2_cube(path: Path):
        assert path == ms_path
        return np.ones((3, 4, 5), dtype=np.float32), "example.ms: visibility amplitude"

    monkeypatch.setattr(visualizer, "_load_msv2_cube", fake_load_msv2_cube)

    response = visualizer._build_integrated_response(ms_path, "mean")
    payload = json.loads(response.body)
    assert payload["stats"]["format"] == "msv2"
    assert payload["stats"]["integrated_shape"] == [4, 5]
    assert payload["stats"]["cube_name"].endswith("visibility amplitude")


def test_complex_component_projection_uses_imaginary_part():
    cube = np.array([[[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]]], dtype=np.complex64)
    projected = visualizer._project_complex_component(cube, complex_component="imag")
    assert projected.tolist() == [[[2.0, 4.0], [6.0, 8.0]]]
