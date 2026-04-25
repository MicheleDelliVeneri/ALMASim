"""Unit tests for almasim.services.interferometry.multiconfig."""

from __future__ import annotations

import numpy as np
import pytest

from almasim.services.interferometry.multiconfig import (
    _combine_baseline_cubes,
    combine_interferometric_results,
)


# ===========================================================================
# helpers
# ===========================================================================


def _make_cube(shape=(4, 6, 8)) -> np.ndarray:
    return np.random.rand(*shape).astype(np.float32)


def _make_result(
    n_chan: int = 4,
    n_px: int = 8,
    n_baselines: int = 6,
    with_visibility_table: bool = False,
) -> dict:
    """Build a minimal per-config result dictionary."""
    r: dict = {
        "model_cube": _make_cube((n_chan, n_px, n_px)),
        "dirty_cube": _make_cube((n_chan, n_px, n_px)),
        "model_vis": np.ones((n_chan, n_px), dtype=np.complex64),
        "dirty_vis": np.ones((n_chan, n_px), dtype=np.complex64),
        "beam_cube": _make_cube((n_chan, n_px, n_px)),
        "totsampling_cube": np.abs(_make_cube((n_chan, n_px, n_px))),
        "u_cube": _make_cube((n_chan, n_baselines, n_px)),
        "v_cube": _make_cube((n_chan, n_baselines, n_px)),
        # Scalar extras that propagate via dict(per_config_results[0])
        "extra_key": "kept",
    }
    if with_visibility_table:
        r["visibility_table"] = {
            "uvw_m": np.zeros((n_baselines, 3)),
            "antenna1": np.zeros(n_baselines, dtype=np.int32),
            "antenna2": np.ones(n_baselines, dtype=np.int32),
            "time_mjd_s": np.zeros(n_baselines),
            "interval_s": np.ones(n_baselines),
            "exposure_s": np.ones(n_baselines),
            "data": np.zeros((n_baselines, 1, n_chan), dtype=np.complex64),
            "model_data": np.zeros((n_baselines, 1, n_chan), dtype=np.complex64),
            "flag": np.ones((n_baselines, 1, n_chan), dtype=bool),
            "weight": np.ones((n_baselines, 1), dtype=np.float32),
            "sigma": np.ones((n_baselines, 1), dtype=np.float32),
            "channel_freq_hz": np.ones(n_chan),
            "antenna_names": ["A1", "A2"],
            "antenna_positions_m": np.zeros((2, 3)),
            "source_name": "TEST",
            "field_ra_rad": 0.0,
            "field_dec_rad": 0.0,
            "observation_date": "2024-01-01",
        }
    return r


# ===========================================================================
# _combine_baseline_cubes
# ===========================================================================


@pytest.mark.unit
def test_combine_baseline_cubes_empty_raises():
    """_combine_baseline_cubes raises ValueError on empty list."""
    with pytest.raises(ValueError, match="No baseline cubes"):
        _combine_baseline_cubes([])


@pytest.mark.unit
def test_combine_baseline_cubes_single():
    """_combine_baseline_cubes with one cube returns it unchanged."""
    cube = np.ones((4, 6, 8))
    result = _combine_baseline_cubes([cube])
    np.testing.assert_array_equal(result, cube)


@pytest.mark.unit
def test_combine_baseline_cubes_compatible():
    """Two cubes with matching chan/px axes are concatenated along axis=1."""
    a = np.ones((4, 6, 8))
    b = np.ones((4, 3, 8)) * 2
    result = _combine_baseline_cubes([a, b])
    assert result.shape == (4, 9, 8)


@pytest.mark.unit
def test_combine_baseline_cubes_incompatible_chan():
    """Mismatched chan axis falls back to first cube."""
    a = np.ones((4, 6, 8))
    b = np.ones((3, 6, 8))
    result = _combine_baseline_cubes([a, b])
    np.testing.assert_array_equal(result, a)


@pytest.mark.unit
def test_combine_baseline_cubes_incompatible_px():
    """Mismatched px axis falls back to first cube."""
    a = np.ones((4, 6, 8))
    b = np.ones((4, 6, 16))
    result = _combine_baseline_cubes([a, b])
    np.testing.assert_array_equal(result, a)


@pytest.mark.unit
def test_combine_baseline_cubes_non_3d_falls_back():
    """Non-3D cube list falls back to first cube."""
    a = np.ones((4, 6))  # 2D — ndim != 3
    b = np.ones((4, 6))
    result = _combine_baseline_cubes([a, b])
    np.testing.assert_array_equal(result, a)


# ===========================================================================
# combine_interferometric_results
# ===========================================================================


@pytest.mark.unit
def test_combine_empty_list_raises():
    """combine_interferometric_results raises ValueError on empty list."""
    with pytest.raises(ValueError, match="Cannot combine an empty"):
        combine_interferometric_results([])


@pytest.mark.unit
def test_combine_single_result_passthrough():
    """Single-element list is returned with metadata added."""
    r = _make_result()
    combined = combine_interferometric_results([r])
    assert combined["combined_config_count"] == 1
    assert combined["per_config_results"] == [r]
    # Original key preserved
    assert combined["extra_key"] == "kept"


@pytest.mark.unit
def test_combine_two_results_shape():
    """Two results produce combined_config_count=2 and correct cube shapes."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    combined = combine_interferometric_results([r1, r2])
    assert combined["combined_config_count"] == 2
    assert combined["model_cube"].shape == r1["model_cube"].shape
    assert combined["dirty_cube"].shape == r1["dirty_cube"].shape


@pytest.mark.unit
def test_combine_default_weights_are_equal():
    """Default equal weights give model_cube == simple mean."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    r1["model_cube"][:] = 2.0
    r2["model_cube"][:] = 4.0
    combined = combine_interferometric_results([r1, r2])
    np.testing.assert_allclose(combined["model_cube"], 3.0, atol=1e-5)


@pytest.mark.unit
def test_combine_custom_weights():
    """Custom weights produce weighted average in model_cube."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    r1["model_cube"][:] = 0.0
    r2["model_cube"][:] = 10.0
    # Weight 0 for first, 1 for second → should be 10
    combined = combine_interferometric_results([r1, r2], config_weights=[0.0, 1.0])
    np.testing.assert_allclose(combined["model_cube"], 10.0, atol=1e-5)


@pytest.mark.unit
def test_combine_all_zero_weights_treated_as_uniform():
    """All-zero weight vector falls back to uniform weights."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    r1["model_cube"][:] = 2.0
    r2["model_cube"][:] = 4.0
    combined = combine_interferometric_results([r1, r2], config_weights=[0.0, 0.0])
    np.testing.assert_allclose(combined["model_cube"], 3.0, atol=1e-5)


@pytest.mark.unit
def test_combine_model_vis_is_summed():
    """model_vis and dirty_vis are summed (not averaged)."""
    n_chan, n_px = 4, 8
    r1 = _make_result(n_chan=n_chan, n_px=n_px)
    r2 = _make_result(n_chan=n_chan, n_px=n_px)
    r1["model_vis"][:] = 1.0
    r2["model_vis"][:] = 2.0
    combined = combine_interferometric_results([r1, r2])
    np.testing.assert_allclose(np.abs(combined["model_vis"]), 3.0, atol=1e-5)


@pytest.mark.unit
def test_combine_totsampling_is_summed():
    """totsampling_cube is summed across configs."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    r1["totsampling_cube"][:] = 1.0
    r2["totsampling_cube"][:] = 3.0
    combined = combine_interferometric_results([r1, r2])
    np.testing.assert_allclose(combined["totsampling_cube"], 4.0, atol=1e-5)


@pytest.mark.unit
def test_combine_uv_mask_derived_from_totsampling():
    """uv_mask_cube contains only 0/1 values derived from totsampling_cube."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    combined = combine_interferometric_results([r1, r2])
    unique = np.unique(combined["uv_mask_cube"])
    assert set(unique.tolist()).issubset({0, 1})


@pytest.mark.unit
def test_combine_beam_cube_normalized():
    """beam_cube center value is 1 after normalization."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    # Force beam to be uniform so center is well defined
    r1["beam_cube"][:] = 2.0
    r2["beam_cube"][:] = 2.0
    combined = combine_interferometric_results([r1, r2])
    beam = combined["beam_cube"]
    cx = beam.shape[1] // 2
    cy = beam.shape[2] // 2
    np.testing.assert_allclose(beam[:, cx, cy], 1.0, atol=1e-5)


@pytest.mark.unit
def test_combine_with_visibility_table():
    """visibility_table is concatenated when present in results."""
    r1 = _make_result(with_visibility_table=True)
    r2 = _make_result(with_visibility_table=True)
    combined = combine_interferometric_results([r1, r2])
    assert "visibility_table" in combined
    vt = combined["visibility_table"]
    n1 = r1["visibility_table"]["uvw_m"].shape[0]
    n2 = r2["visibility_table"]["uvw_m"].shape[0]
    assert vt["uvw_m"].shape[0] == n1 + n2


@pytest.mark.unit
def test_combine_without_visibility_table_no_key():
    """When no configs have visibility_table, the key is absent."""
    r1 = _make_result(with_visibility_table=False)
    r2 = _make_result(with_visibility_table=False)
    combined = combine_interferometric_results([r1, r2])
    assert "visibility_table" not in combined


@pytest.mark.unit
def test_combine_scalars_beam_totsampling_uv_mask():
    """Scalar 'beam', 'totsampling', 'uv_mask' keys are present in combined."""
    r1 = _make_result(n_chan=4, n_px=8)
    r2 = _make_result(n_chan=4, n_px=8)
    combined = combine_interferometric_results([r1, r2])
    assert "beam" in combined
    assert "totsampling" in combined
    assert "uv_mask" in combined
    # Each is a 2D slice
    assert combined["beam"].ndim == 2
    assert combined["totsampling"].ndim == 2
    assert combined["uv_mask"].ndim == 2
