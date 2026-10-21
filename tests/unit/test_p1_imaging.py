"""Unit tests for P1 total-power and image reconstruction helpers."""

import numpy as np
import pytest

from almasim.services.imaging import (
    build_image_products,
    convolve_cube_with_beam,
    feather_merge_cube,
    regrid_cube_to_match,
    wiener_deconvolve_cube,
    wiener_preview,
)
from almasim.services.interferometry.total_power import (
    estimate_tp_beam_fwhm_arcsec,
    simulate_total_power_observation,
)


@pytest.mark.unit
def test_simulate_total_power_observation_smooths_point_source():
    """TP simulation should smooth a point source and preserve cube shapes."""
    model_cube = np.zeros((4, 16, 16), dtype=np.float32)
    model_cube[:, 8, 8] = 1.0
    freqs_hz = np.linspace(90e9, 110e9, 4)

    result = simulate_total_power_observation(
        model_cube,
        freqs_hz=freqs_hz,
        cell_size_arcsec=0.2,
        noise_profile=np.zeros(4, dtype=np.float32),
        antenna_diameter_m=12.0,
    )

    assert result["tp_model_cube"].shape == model_cube.shape
    assert result["tp_dirty_cube"].shape == model_cube.shape
    assert result["tp_beam_cube"].shape == model_cube.shape
    assert np.max(result["tp_model_cube"][0]) < np.max(model_cube[0])
    assert np.all(result["tp_beam_fwhm_arcsec"] > 0.0)


@pytest.mark.unit
def test_estimate_tp_beam_fwhm_arcsec_decreases_with_frequency():
    """Higher-frequency TP channels should have smaller beams."""
    beam_fwhm = estimate_tp_beam_fwhm_arcsec(np.array([90e9, 230e9]), antenna_diameter_m=12.0)
    assert beam_fwhm[1] < beam_fwhm[0]


@pytest.mark.unit
def test_regrid_cube_to_match_changes_shape():
    """Nearest-neighbor regridding should return the requested cube shape."""
    cube = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    regridded = regrid_cube_to_match(cube, (3, 6, 6))
    assert regridded.shape == (3, 6, 6)


@pytest.mark.unit
def test_wiener_deconvolve_cube_preserves_shape():
    """Wiener deconvolution should keep the cube dimensions unchanged."""
    dirty_cube = np.ones((2, 8, 8), dtype=np.float32)
    beam_cube = np.zeros((2, 8, 8), dtype=np.float32)
    beam_cube[:, 4, 4] = 1.0

    image_cube = wiener_deconvolve_cube(dirty_cube, beam_cube)

    assert image_cube.shape == dirty_cube.shape
    assert np.all(image_cube >= 0.0)


@pytest.mark.unit
def test_feather_merge_cube_returns_target_shape():
    """Feather merge should return an image cube on the interferometric grid."""
    int_cube = np.ones((2, 8, 8), dtype=np.float32)
    tp_cube = np.ones((2, 6, 6), dtype=np.float32) * 2.0
    tp_beam_cube = np.zeros((2, 6, 6), dtype=np.float32)
    tp_beam_cube[:, 3, 3] = 1.0

    merged = feather_merge_cube(int_cube, tp_cube, tp_beam_cube)

    assert merged.shape == int_cube.shape


@pytest.mark.unit
def test_build_image_products_produces_int_tp_and_merged_outputs():
    """P1 image products should include INT, TP, and TP+INT cubes when available."""
    dirty_cube = np.ones((2, 8, 8), dtype=np.float32)
    beam_cube = np.zeros((2, 8, 8), dtype=np.float32)
    beam_cube[:, 4, 4] = 1.0
    tp_dirty_cube = np.ones((2, 8, 8), dtype=np.float32) * 2.0
    tp_beam_cube = np.zeros((2, 8, 8), dtype=np.float32)
    tp_beam_cube[:, 4, 4] = 1.0

    products = build_image_products(
        int_results={
            "dirty_cube": dirty_cube,
            "beam_cube": beam_cube,
            "per_config_results": [
                {
                    "config_name": "alma12",
                    "array_type": "12m",
                    "dirty_cube": dirty_cube,
                    "beam_cube": beam_cube,
                }
            ],
        },
        tp_results={
            "tp_dirty_cube": tp_dirty_cube,
            "tp_beam_cube": tp_beam_cube,
        },
    )

    assert products["int_image_cube"] is not None
    assert products["tp_image_cube"] is not None
    assert products["tp_int_image_cube"] is not None
    assert len(products["per_config_image_results"]) == 1


def _blurred_point(position=(10, 10), sigma=1.8, size=21):
    model_cube = np.zeros((1, size, size), dtype=np.float32)
    model_cube[(0, *position)] = 1.0
    centre = size // 2
    yy, xx = np.indices((size, size), dtype=np.float32)
    beam = np.exp(-0.5 * (((yy - centre) / sigma) ** 2 + ((xx - centre) / sigma) ** 2))
    beam = (beam / np.max(beam)).astype(np.float32)
    dirty = np.real(np.fft.ifft2(np.fft.fft2(model_cube[0]) * np.fft.fft2(np.fft.ifftshift(beam))))
    return model_cube, dirty.astype(np.float32)[None, ...], beam[None, ...]


@pytest.mark.unit
def test_wiener_preview_sharpens_a_blurred_point_source():
    """The Wiener model is closer to the true point than the dirty image (MADIV-28 T5.12)."""
    model_cube, dirty_cube, beam_cube = _blurred_point()

    result = wiener_preview(dirty_cube, beam_cube, epsilon=1e-3)

    for name in ("model", "restored", "residual", "clean_beam"):
        assert result[name].shape == dirty_cube.shape, name
    assert np.unravel_index(int(np.argmax(result["model"][0])), (21, 21)) == (10, 10)
    assert np.mean((result["model"] - model_cube) ** 2) < np.mean((dirty_cube - model_cube) ** 2)
    assert np.max(np.abs(result["residual"])) < np.max(np.abs(dirty_cube))
    assert float(np.max(result["clean_beam"])) == pytest.approx(1.0)


@pytest.mark.unit
def test_wiener_preview_keeps_a_corner_source_in_place():
    model_cube, dirty_cube, beam_cube = _blurred_point(position=(3, 17))
    result = wiener_preview(dirty_cube, beam_cube)
    assert np.unravel_index(int(np.argmax(result["model"][0])), (21, 21)) == (3, 17)


@pytest.mark.unit
def test_wiener_preview_rejects_bad_inputs():
    _, dirty_cube, beam_cube = _blurred_point()
    with pytest.raises(ValueError, match="same shape"):
        wiener_preview(dirty_cube, beam_cube[:, :20, :20])
    with pytest.raises(ValueError, match="n_channels, ny, nx"):
        wiener_preview(dirty_cube[0], beam_cube[0])
    with pytest.raises(ValueError, match="epsilon"):
        wiener_preview(dirty_cube, beam_cube, epsilon=0.0)


@pytest.mark.unit
def test_convolve_cube_with_beam_matches_cube_shape():
    """Convolving a cube with a matched beam cube should preserve shape."""
    cube = np.zeros((2, 9, 9), dtype=np.float32)
    cube[:, 4, 4] = 1.0
    beam_cube = np.zeros((2, 9, 9), dtype=np.float32)
    beam_cube[:, 4, 4] = 1.0

    convolved = convolve_cube_with_beam(cube, beam_cube)

    assert convolved.shape == cube.shape
