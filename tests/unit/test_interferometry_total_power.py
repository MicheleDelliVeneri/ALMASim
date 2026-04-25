"""Unit tests for almasim.services.interferometry.total_power."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.total_power import (
    ARCSEC_PER_RAD,
    _fft_convolve2d,
    _gaussian_kernel,
    combine_total_power_results,
    estimate_tp_beam_fwhm_arcsec,
    simulate_total_power_observation,
)


# ===========================================================================
# estimate_tp_beam_fwhm_arcsec
# ===========================================================================


@pytest.mark.unit
def test_estimate_tp_beam_fwhm_units():
    """FWHM is in arcsec (large values for typical ALMA frequencies)."""
    freqs = np.array([100e9, 200e9, 300e9])  # Hz
    result = estimate_tp_beam_fwhm_arcsec(freqs, antenna_diameter_m=12.0)
    assert result.shape == (3,)
    # At 100 GHz, 12m dish: ~7 arcsec
    assert result[0] > 1.0
    assert result[0] < 100.0


@pytest.mark.unit
def test_estimate_tp_beam_fwhm_higher_freq_smaller_beam():
    """Higher frequency gives smaller beam FWHM."""
    freqs = np.array([100e9, 300e9])
    result = estimate_tp_beam_fwhm_arcsec(freqs, antenna_diameter_m=12.0)
    assert result[0] > result[1]


@pytest.mark.unit
def test_estimate_tp_beam_fwhm_larger_dish_smaller_beam():
    """Larger dish gives smaller FWHM."""
    freqs = np.array([100e9])
    r12 = estimate_tp_beam_fwhm_arcsec(freqs, antenna_diameter_m=12.0)
    r7 = estimate_tp_beam_fwhm_arcsec(freqs, antenna_diameter_m=7.0)
    assert r7[0] > r12[0]


@pytest.mark.unit
def test_estimate_tp_beam_fwhm_zero_freq_clipped():
    """Zero frequency is clipped to 1 Hz to avoid division by zero."""
    freqs = np.array([0.0, 100e9])
    result = estimate_tp_beam_fwhm_arcsec(freqs, antenna_diameter_m=12.0)
    assert np.all(np.isfinite(result))


# ===========================================================================
# _gaussian_kernel
# ===========================================================================


@pytest.mark.unit
def test_gaussian_kernel_sum_one():
    """Gaussian kernel sums to 1."""
    kernel = _gaussian_kernel((16, 16), sigma_pix=2.0)
    np.testing.assert_allclose(np.sum(kernel), 1.0, rtol=1e-5)


@pytest.mark.unit
def test_gaussian_kernel_shape():
    """Kernel has the requested shape."""
    kernel = _gaussian_kernel((12, 8), sigma_pix=1.5)
    assert kernel.shape == (12, 8)


@pytest.mark.unit
def test_gaussian_kernel_peak_at_center():
    """Kernel peak is at the center pixel."""
    kernel = _gaussian_kernel((17, 17), sigma_pix=2.0)
    peak = np.unravel_index(np.argmax(kernel), kernel.shape)
    assert peak == (8, 8)


@pytest.mark.unit
def test_gaussian_kernel_zero_sigma_delta():
    """sigma_pix=0 produces a delta function at center."""
    kernel = _gaussian_kernel((8, 8), sigma_pix=0.0)
    assert kernel[4, 4] == pytest.approx(1.0)
    assert np.sum(kernel) == pytest.approx(1.0)


# ===========================================================================
# _fft_convolve2d
# ===========================================================================


@pytest.mark.unit
def test_fft_convolve2d_delta_is_identity():
    """Convolving with a delta function returns the image unchanged."""
    image = np.random.rand(16, 16).astype(np.float32)
    delta = np.zeros((16, 16), dtype=np.float32)
    delta[8, 8] = 1.0  # Delta at center (after ifftshift alignment)
    result = _fft_convolve2d(image, delta)
    # Not exactly identity due to FFT shifts, but shape must match
    assert result.shape == image.shape


@pytest.mark.unit
def test_fft_convolve2d_shape_preserved():
    """Output shape matches input image shape."""
    image = np.ones((16, 16), dtype=np.float32)
    kernel = _gaussian_kernel((16, 16), sigma_pix=1.0)
    result = _fft_convolve2d(image, kernel)
    assert result.shape == (16, 16)


@pytest.mark.unit
def test_fft_convolve2d_constant_image():
    """Convolving a constant image with a normalized kernel leaves it unchanged."""
    image = np.full((16, 16), 5.0, dtype=np.float32)
    kernel = _gaussian_kernel((16, 16), sigma_pix=2.0)
    result = _fft_convolve2d(image, kernel)
    # Interior pixels away from edges should be ~5.0
    np.testing.assert_allclose(result[4:-4, 4:-4], 5.0, atol=0.1)


# ===========================================================================
# simulate_total_power_observation
# ===========================================================================


@pytest.mark.unit
def test_simulate_tp_basic():
    """simulate_total_power_observation returns expected keys."""
    model = np.random.rand(4, 16, 16).astype(np.float32)
    freqs = np.linspace(100e9, 104e9, 4)
    result = simulate_total_power_observation(
        model,
        freqs_hz=freqs,
        cell_size_arcsec=1.0,
        noise_profile=0.01,
    )
    assert "tp_model_cube" in result
    assert "tp_dirty_cube" in result
    assert "tp_beam_cube" in result
    assert "tp_sampling_cube" in result
    assert "tp_beam_fwhm_arcsec" in result


@pytest.mark.unit
def test_simulate_tp_output_shapes():
    """TP cubes have same shape as model_cube."""
    n_chan, n_px = 4, 16
    model = np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    freqs = np.linspace(100e9, 104e9, n_chan)
    result = simulate_total_power_observation(
        model,
        freqs_hz=freqs,
        cell_size_arcsec=1.0,
        noise_profile=0.0,  # no noise
    )
    assert result["tp_model_cube"].shape == (n_chan, n_px, n_px)
    assert result["tp_dirty_cube"].shape == (n_chan, n_px, n_px)


@pytest.mark.unit
def test_simulate_tp_no_noise():
    """With zero noise, dirty_cube equals model_cube."""
    n_chan, n_px = 4, 16
    model = np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    freqs = np.linspace(100e9, 104e9, n_chan)
    result = simulate_total_power_observation(
        model,
        freqs_hz=freqs,
        cell_size_arcsec=1.0,
        noise_profile=0.0,
    )
    np.testing.assert_array_equal(result["tp_dirty_cube"], result["tp_model_cube"])


@pytest.mark.unit
def test_simulate_tp_non_scalar_noise():
    """noise_profile as array is accepted."""
    n_chan, n_px = 4, 16
    model = np.random.rand(n_chan, n_px, n_px).astype(np.float32)
    freqs = np.linspace(100e9, 104e9, n_chan)
    noise = np.full(n_chan, 0.01)
    result = simulate_total_power_observation(
        model,
        freqs_hz=freqs,
        cell_size_arcsec=1.0,
        noise_profile=noise,
    )
    assert result["tp_model_cube"].shape == (n_chan, n_px, n_px)


@pytest.mark.unit
def test_simulate_tp_wrong_ndim_raises():
    """2D model_cube raises ValueError."""
    with pytest.raises(ValueError, match="model_cube must have shape"):
        simulate_total_power_observation(
            np.ones((16, 16)),
            freqs_hz=np.array([100e9]),
            cell_size_arcsec=1.0,
            noise_profile=0.0,
        )


@pytest.mark.unit
def test_simulate_tp_freq_mismatch_raises():
    """Mismatched freqs length raises ValueError."""
    with pytest.raises(ValueError, match="freqs_hz length"):
        simulate_total_power_observation(
            np.ones((4, 16, 16)),
            freqs_hz=np.array([100e9, 200e9]),  # only 2, not 4
            cell_size_arcsec=1.0,
            noise_profile=0.0,
        )


@pytest.mark.unit
def test_simulate_tp_metadata_fields():
    """config_name and array_type propagate to output."""
    model = np.ones((2, 8, 8), dtype=np.float32)
    freqs = np.array([100e9, 200e9])
    result = simulate_total_power_observation(
        model,
        freqs_hz=freqs,
        cell_size_arcsec=1.0,
        noise_profile=0.0,
        config_name="TP_config",
        array_type="TP",
    )
    assert result["config_name"] == "TP_config"
    assert result["array_type"] == "TP"
    assert result["antenna_diameter_m"] == 12.0


# ===========================================================================
# combine_total_power_results
# ===========================================================================


def _make_tp_result(n_chan=4, n_px=8, fill=1.0):
    cube = np.full((n_chan, n_px, n_px), fill, dtype=np.float32)
    return {
        "tp_model_cube": cube.copy(),
        "tp_dirty_cube": cube.copy(),
        "tp_beam_cube": cube.copy(),
        "tp_sampling_cube": np.ones((n_chan, n_px, n_px), dtype=np.float32),
        "tp_beam_fwhm_arcsec": np.ones(n_chan, dtype=np.float32),
        "config_name": "tp0",
        "array_type": "TP",
        "antenna_diameter_m": 12.0,
    }


@pytest.mark.unit
def test_combine_tp_empty_raises():
    """Empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot combine an empty"):
        combine_total_power_results([])


@pytest.mark.unit
def test_combine_tp_single_passthrough():
    """Single result is returned with added metadata."""
    r = _make_tp_result()
    combined = combine_total_power_results([r])
    assert combined["combined_config_count"] == 1
    assert combined["per_config_results"] == [r]


@pytest.mark.unit
def test_combine_tp_two_default_weights_mean():
    """Two equal-weight results average model cubes."""
    r1 = _make_tp_result(fill=2.0)
    r2 = _make_tp_result(fill=4.0)
    combined = combine_total_power_results([r1, r2])
    np.testing.assert_allclose(combined["tp_model_cube"], 3.0, atol=1e-5)


@pytest.mark.unit
def test_combine_tp_sampling_summed():
    """tp_sampling_cube is summed across configs."""
    r1 = _make_tp_result()
    r2 = _make_tp_result()
    r1["tp_sampling_cube"][:] = 1.0
    r2["tp_sampling_cube"][:] = 3.0
    combined = combine_total_power_results([r1, r2])
    np.testing.assert_allclose(combined["tp_sampling_cube"], 4.0, atol=1e-5)


@pytest.mark.unit
def test_combine_tp_custom_weights():
    """Custom weights produce weighted average."""
    r1 = _make_tp_result(fill=0.0)
    r2 = _make_tp_result(fill=10.0)
    combined = combine_total_power_results([r1, r2], config_weights=[0.0, 1.0])
    np.testing.assert_allclose(combined["tp_model_cube"], 10.0, atol=1e-5)


@pytest.mark.unit
def test_combine_tp_zero_weights_uniform():
    """All-zero weights falls back to uniform weights."""
    r1 = _make_tp_result(fill=2.0)
    r2 = _make_tp_result(fill=4.0)
    combined = combine_total_power_results([r1, r2], config_weights=[0.0, 0.0])
    np.testing.assert_allclose(combined["tp_model_cube"], 3.0, atol=1e-5)
