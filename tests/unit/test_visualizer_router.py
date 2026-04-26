"""Tests for visualizer file discovery and integration."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from fastapi.dependencies import utils as fastapi_dependency_utils

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "src"))

_VISUALIZER_PATH = REPO_ROOT / "backend" / "app" / "api" / "v1" / "routers" / "visualizer.py"
_VISUALIZER_SPEC = importlib.util.spec_from_file_location(
    "visualizer_router_test_module", _VISUALIZER_PATH
)
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


# ---------------------------------------------------------------------------
# Tests for _resolve_visualizer_path (path-traversal guard)
# ---------------------------------------------------------------------------


def test_resolve_visualizer_path_valid_relative(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    result = visualizer._resolve_visualizer_path("subdir", tmp_path)
    assert result == sub.resolve()


def test_resolve_visualizer_path_valid_absolute_inside_base(tmp_path):
    sub = tmp_path / "inside"
    sub.mkdir()
    result = visualizer._resolve_visualizer_path(str(sub), tmp_path)
    assert result == sub.resolve()


def test_resolve_visualizer_path_rejects_traversal(tmp_path):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        visualizer._resolve_visualizer_path("../../etc/passwd", tmp_path)
    assert exc_info.value.status_code == 400


def test_resolve_visualizer_path_rejects_absolute_outside_base(tmp_path):
    from fastapi import HTTPException

    outside = tmp_path.parent / "outside"
    with pytest.raises(HTTPException) as exc_info:
        visualizer._resolve_visualizer_path(str(outside), tmp_path)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Tests for list_datacube_files endpoint
# ---------------------------------------------------------------------------


def test_list_datacube_files_no_dir_uses_output_dir(monkeypatch, tmp_path):
    np.savez(tmp_path / "cube.npz", clean_cube=np.ones((2, 3, 4), dtype=np.float32))
    monkeypatch.setattr(visualizer.settings, "OUTPUT_DIR", tmp_path)

    response = asyncio.run(visualizer.list_datacube_files(dir=None))
    payload = json.loads(response.body)
    assert payload["output_dir"] == str(tmp_path.resolve())
    assert any(f["name"] == "cube.npz" for f in payload["files"])


def test_list_datacube_files_with_valid_subdir(monkeypatch, tmp_path):
    subdir = tmp_path / "sub"
    subdir.mkdir()
    np.savez(subdir / "cube.npz", clean_cube=np.ones((2, 3, 4), dtype=np.float32))
    monkeypatch.setattr(visualizer.settings, "OUTPUT_DIR", tmp_path)

    response = asyncio.run(visualizer.list_datacube_files(dir="sub"))
    payload = json.loads(response.body)
    assert any(f["name"] == "cube.npz" for f in payload["files"])


def test_list_datacube_files_traversal_dir_rejected(monkeypatch, tmp_path):
    from fastapi import HTTPException

    monkeypatch.setattr(visualizer.settings, "OUTPUT_DIR", tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(visualizer.list_datacube_files(dir="../../etc"))
    assert exc_info.value.status_code == 400


def test_list_datacube_files_nonexistent_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(visualizer.settings, "OUTPUT_DIR", tmp_path / "nonexistent")

    response = asyncio.run(visualizer.list_datacube_files(dir=None))
    payload = json.loads(response.body)
    assert payload["files"] == []
    assert "does not exist" in payload["message"]
