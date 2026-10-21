"""The Wiener preview endpoint (MADIV-28 T5.12)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.api.v1.routers import imaging as imaging_router  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.schemas.imaging import WienerPreviewRequest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def cubes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path)
    model = np.zeros((1, 15, 15), dtype=np.float32)
    model[0, 7, 7] = 1.0
    yy, xx = np.indices((15, 15), dtype=np.float32)
    beam = np.exp(-0.5 * (((yy - 7) / 1.5) ** 2 + ((xx - 7) / 1.5) ** 2)).astype(np.float32)
    dirty = np.real(np.fft.ifft2(np.fft.fft2(model[0]) * np.fft.fft2(np.fft.ifftshift(beam))))
    np.savez(tmp_path / "clean-cube_0.npz", clean_cube=model)
    np.savez(tmp_path / "dirty-cube_0.npz", dirty_cube=dirty[None].astype(np.float32))
    np.savez(tmp_path / "beam-cube_0.npz", beam_cube=beam[None])
    return tmp_path


def _request(directory, **overrides):
    values = {
        "directory": str(directory),
        "dirty_cube_path": "dirty-cube_0.npz",
        "beam_cube_path": "beam-cube_0.npz",
        **overrides,
    }
    return WienerPreviewRequest(**values)


def test_preview_writes_the_cubes_and_returns_the_panels(cubes):
    response = imaging_router.wiener_preview_saved_products(
        _request(cubes, clean_cube_path="clean-cube_0.npz", epsilon=0.01)
    )
    assert response.reference_clean is not None and response.convolved_reference is not None
    assert response.metadata.epsilon == 0.01
    for name in ("wiener-model-cube_0", "restored-cube_0", "residual-cube_0"):
        assert (cubes / f"{name}.npz").is_file(), name
    assert (cubes / "convolved-clean-reference-cube_0.npz").is_file()


def test_preview_without_a_reference(cubes):
    response = imaging_router.wiener_preview_saved_products(_request(cubes))
    assert response.reference_clean is None and response.convolved_reference is None
    assert response.metadata.convolved_reference_path is None


def test_mismatched_cubes_are_a_bad_request(cubes):
    np.savez(cubes / "beam-cube_1.npz", beam_cube=np.ones((1, 9, 9), np.float32))
    with pytest.raises(HTTPException) as info:
        imaging_router.wiener_preview_saved_products(
            _request(cubes, beam_cube_path="beam-cube_1.npz")
        )
    assert info.value.status_code == 400


def test_an_unexpected_failure_is_a_server_error(cubes, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("FFT exploded")

    monkeypatch.setattr(imaging_router, "wiener_preview", broken)
    with pytest.raises(HTTPException) as info:
        imaging_router.wiener_preview_saved_products(_request(cubes))
    assert info.value.status_code == 500
    assert "FFT exploded" in info.value.detail


def test_missing_and_non_npz_inputs_are_rejected(cubes):
    with pytest.raises(HTTPException) as info:
        imaging_router.wiener_preview_saved_products(_request(cubes, beam_cube_path="nope.npz"))
    assert info.value.status_code == 404
    (cubes / "beam.fits").write_bytes(b"")
    with pytest.raises(HTTPException) as info:
        imaging_router.wiener_preview_saved_products(_request(cubes, beam_cube_path="beam.fits"))
    assert info.value.status_code == 400


def test_outputs_of_an_unprefixed_cube_keep_its_name(cubes):
    (cubes / "dirty-cube_0.npz").rename(cubes / "observed.npz")
    imaging_router.wiener_preview_saved_products(_request(cubes, dirty_cube_path="observed.npz"))
    assert (cubes / "wiener-model-cube_observed.npz").is_file()
