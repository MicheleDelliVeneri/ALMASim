"""Unit tests for P1 total-power and image reconstruction helpers."""

import numpy as np
import pytest

from almasim.services.imaging import (
    build_image_products,
    clean_deconvolve_cube,
    convolve_cube_with_beam,
    feather_merge_cube,
    integrate_cube_preview,
    regrid_cube_to_match,
    wiener_deconvolve_cube,
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


@pytest.mark.unit
def test_clean_deconvolve_cube_improves_blurred_point_source():
    """Iterative CLEAN should move a blurred point source closer to the true model."""
    model_cube = np.zeros((1, 21, 21), dtype=np.float32)
    model_cube[0, 10, 10] = 1.0

    yy, xx = np.indices((21, 21), dtype=np.float32)
    sigma = 1.8
    beam = np.exp(-0.5 * (((yy - 10) / sigma) ** 2 + ((xx - 10) / sigma) ** 2)).astype(np.float32)
    beam /= np.max(beam)
    beam_cube = beam[None, ...]

    dirty_fft = np.fft.fft2(model_cube[0]) * np.fft.fft2(np.fft.ifftshift(beam))
    dirty_cube = np.real(np.fft.ifft2(dirty_fft)).astype(np.float32)[None, ...]

    result = clean_deconvolve_cube(dirty_cube, beam_cube, n_cycles=120, gain=0.12)
    restored = result["clean_cube"]
    residual = result["residual_cube"]

    dirty_error = np.mean((dirty_cube - model_cube) ** 2)
    restored_error = np.mean((restored - model_cube) ** 2)

    assert restored.shape == dirty_cube.shape
    assert residual.shape == dirty_cube.shape
    assert restored_error < dirty_error
    assert np.max(np.abs(residual)) < np.max(np.abs(dirty_cube))


@pytest.mark.unit
def test_clean_deconvolve_cube_can_resume_from_previous_state():
    """Continuing from saved CLEAN state should match a single longer run."""
    model_cube = np.zeros((1, 21, 21), dtype=np.float32)
    model_cube[0, 10, 10] = 1.0

    yy, xx = np.indices((21, 21), dtype=np.float32)
    sigma = 1.8
    beam = np.exp(-0.5 * (((yy - 10) / sigma) ** 2 + ((xx - 10) / sigma) ** 2)).astype(np.float32)
    beam /= np.max(beam)
    beam_cube = beam[None, ...]

    dirty_fft = np.fft.fft2(model_cube[0]) * np.fft.fft2(np.fft.ifftshift(beam))
    dirty_cube = np.real(np.fft.ifft2(dirty_fft)).astype(np.float32)[None, ...]

    first = clean_deconvolve_cube(dirty_cube, beam_cube, n_cycles=40, gain=0.12)
    resumed = clean_deconvolve_cube(
        dirty_cube,
        beam_cube,
        n_cycles=80,
        gain=0.12,
        initial_component_cube=first["component_cube"],
        initial_residual_cube=first["residual_cube"],
        initial_clean_beam_cube=first["clean_beam_cube"],
        initial_cycles_completed=first["cycles_completed"],
    )
    direct = clean_deconvolve_cube(dirty_cube, beam_cube, n_cycles=120, gain=0.12)

    assert resumed["cycles_completed"] == 120
    assert np.allclose(resumed["component_cube"], direct["component_cube"], atol=1e-5)
    assert np.allclose(resumed["restored_cube"], direct["restored_cube"], atol=1e-5)
    assert np.allclose(resumed["residual_cube"], direct["residual_cube"], atol=1e-5)


@pytest.mark.unit
def test_convolve_cube_with_beam_matches_cube_shape():
    """Convolving a cube with a matched beam cube should preserve shape."""
    cube = np.zeros((2, 9, 9), dtype=np.float32)
    cube[:, 4, 4] = 1.0
    beam_cube = np.zeros((2, 9, 9), dtype=np.float32)
    beam_cube[:, 4, 4] = 1.0

    convolved = convolve_cube_with_beam(cube, beam_cube)

    assert convolved.shape == cube.shape


@pytest.mark.unit
def test_integrate_cube_preview_returns_image_and_stats():
    """Preview integration should emit a normalized image payload."""
    cube = np.ones((4, 8, 8), dtype=np.float32)

    preview = integrate_cube_preview(cube, method="mean", cube_name="test_cube")

    assert len(preview["image"]) == 8
    assert len(preview["image"][0]) == 8
    assert preview["stats"]["shape"] == [4, 8, 8]
    assert preview["stats"]["cube_name"] == "test_cube"
    assert preview["method"] == "mean"
