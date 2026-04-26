"""Unit tests for interferometry baselines module."""

import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from almasim.services.interferometry.baselines import (
    generate_via_astropy,
    prepare_baselines,
    set_baselines,
    set_noise,
)


@pytest.mark.unit
def test_prepare_baselines_basic():
    """Test basic baseline preparation."""
    Nant = 4
    nH = 10
    Hcov = [0.0, 1.0]

    result = prepare_baselines(Nant, nH, Hcov)
    Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims = result

    # Check number of baselines: Nant * (Nant - 1) / 2 = 4 * 3 / 2 = 6
    assert Nbas == 6
    assert B.shape == (6, 3)  # 3D baseline vector components (x, y, z)
    assert basnum.shape == (4, 3)
    assert basidx.shape == (4, 4)
    assert antnum.shape == (6, 2)
    assert Gains.shape == (6, 10)
    assert Noise.shape == (6, 10)
    assert len(H) == 2
    assert H[0].shape == (10,)
    assert H[1].shape == (10,)
    assert u.shape == (6, 10)
    assert v.shape == (6, 10)
    assert ravelDims == (6, 10)


@pytest.mark.unit
def test_prepare_baselines_validation():
    """Test baseline preparation input validation."""
    # Too few antennas
    with pytest.raises(ValueError, match="at least 2"):
        prepare_baselines(1, 10, [0.0, 1.0])

    # Too few hour angle samples
    with pytest.raises(ValueError, match="at least 1"):
        prepare_baselines(4, 0, [0.0, 1.0])

    # Invalid hour angle coverage
    with pytest.raises(ValueError, match="must have"):
        prepare_baselines(4, 10, [0.0])


@pytest.mark.unit
def test_prepare_baselines_antnum_pairs():
    """Test that antenna pairs are correctly generated."""
    Nant = 3
    nH = 5
    Hcov = [0.0, 1.0]

    _, _, _, _, antnum, _, _, _, _, _, _ = prepare_baselines(Nant, nH, Hcov)

    # For 3 antennas, we should have 3 baselines: (0,1), (0,2), (1,2)
    expected_pairs = {(0, 1), (0, 2), (1, 2)}
    actual_pairs = {tuple(antnum[i]) for i in range(3)}
    assert actual_pairs == expected_pairs


@pytest.mark.unit
def test_prepare_baselines_hour_angles():
    """Test that hour angles are correctly calculated."""
    Nant = 2
    nH = 5
    Hcov = [0.0, 2.0]

    _, _, _, _, _, _, _, H, _, _, _ = prepare_baselines(Nant, nH, Hcov)

    # Check that H contains sin and cos
    assert len(H) == 2
    assert np.allclose(H[0], np.sin(np.linspace(0.0, 2.0, 5)))
    assert np.allclose(H[1], np.cos(np.linspace(0.0, 2.0, 5)))


@pytest.mark.unit
def test_set_noise_basic():
    """Test setting noise values."""
    Noise = np.zeros((3, 5), dtype=np.complex64)
    noise_level = 0.1

    result = set_noise(noise_level, Noise)

    assert result is Noise  # Should modify in place
    assert np.std(Noise.real) > 0
    assert np.std(Noise.imag) > 0
    assert np.allclose(np.mean(Noise.real), 0.0, atol=0.1)
    assert np.allclose(np.mean(Noise.imag), 0.0, atol=0.1)


@pytest.mark.unit
def test_set_noise_validation():
    """Test noise setting input validation."""
    Noise = np.zeros((3, 5), dtype=np.complex64)

    with pytest.raises(ValueError, match="non-negative"):
        set_noise(-0.1, Noise)


@pytest.mark.unit
def test_set_baselines_basic():
    """Test setting baseline vectors and UV coordinates."""
    Nbas = 3
    nH = 5
    antnum = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int8)
    B = np.zeros((Nbas, nH, 3), dtype=np.float32)
    u = np.zeros((Nbas, nH))
    v = np.zeros((Nbas, nH))
    antPos = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]
    trlat = [0.0, 1.0]
    trdec = [0.0, 1.0]
    H = [np.sin(np.linspace(0, 1, nH)), np.cos(np.linspace(0, 1, nH))]
    wavelength = [1.0, 1.1, 1.05]  # [min, max, mean]

    # Fix B shape - it should be 2D, not 3D
    B = np.zeros((Nbas, 3), dtype=np.float32)

    result = set_baselines(Nbas, antnum, B, u, v, antPos, trlat, trdec, H, wavelength)
    B_out, u_out, v_out = result

    assert B_out is B
    assert u_out is u
    assert v_out is v
    assert B.shape == (Nbas, 3)
    assert u.shape == (Nbas, nH)
    assert v.shape == (Nbas, nH)
    # Check that values are set (not all zeros)
    assert np.any(B != 0)
    assert np.any(u != 0)
    assert np.any(v != 0)


@pytest.mark.unit
def test_set_baselines_validation():
    """Test baseline setting input validation."""
    Nbas = 2
    nH = 5
    antnum = np.array([[0, 1]], dtype=np.int8)
    B = np.zeros((Nbas, 3), dtype=np.float32)
    u = np.zeros((Nbas, nH))
    v = np.zeros((Nbas, nH))
    antPos = [[0.0, 0.0], [100.0, 0.0]]
    trlat = [0.0, 1.0]
    trdec = [0.0, 1.0]
    H = [np.sin(np.linspace(0, 1, nH)), np.cos(np.linspace(0, 1, nH))]

    # Invalid wavelength
    with pytest.raises(ValueError, match="at least 3"):
        set_baselines(Nbas, antnum, B, u, v, antPos, trlat, trdec, H, [1.0, 1.1])

    # Invalid trlat
    with pytest.raises(ValueError, match="at least 2"):
        set_baselines(Nbas, antnum, B, u, v, antPos, [0.0], trdec, H, [1.0, 1.1, 1.05])

    # Invalid trdec
    with pytest.raises(ValueError, match="at least 2"):
        set_baselines(Nbas, antnum, B, u, v, antPos, trlat, [0.0], H, [1.0, 1.1, 1.05])


@pytest.mark.unit
def test_set_baselines_uv_calculation():
    """Test that UV coordinates are correctly calculated from baselines."""
    Nbas = 1
    nH = 3
    antnum = np.array([[0, 1]], dtype=np.int8)
    B = np.zeros((Nbas, 3), dtype=np.float32)
    u = np.zeros((Nbas, nH))
    v = np.zeros((Nbas, nH))
    antPos = [[0.0, 0.0], [100.0, 0.0]]  # Antenna 1 is 100m east
    trlat = [0.0, 1.0]
    trdec = [0.0, 1.0]
    H = [np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.866, 0.5])]  # sin, cos
    wavelength = [1.0, 1.1, 1.05]

    set_baselines(Nbas, antnum, B, u, v, antPos, trlat, trdec, H, wavelength)

    # B[0, 1] should be dx / wavelength = 100 / 1.05 ≈ 95.24
    assert np.isclose(B[0, 1], 100.0 / 1.05, rtol=1e-3)
    # u should depend on B and H
    assert u.shape == (1, 3)
    assert v.shape == (1, 3)


@pytest.mark.unit
def test_generate_via_astropy_accepts_geodetic_lat_lon():
    """Geodetic [lat, lon] input should match equivalent XYZ input."""
    lat_lon_deg = np.array(
        [
            [-23.0290, -67.7550],
            [-23.0286, -67.7543],
            [-23.0282, -67.7536],
        ],
        dtype=np.float64,
    )
    loc_xyz = EarthLocation.from_geodetic(
        lon=lat_lon_deg[:, 1] * u.deg,
        lat=lat_lon_deg[:, 0] * u.deg,
        height=np.zeros(lat_lon_deg.shape[0]) * u.m,
    )
    xyz_m = np.column_stack(
        [
            loc_xyz.x.to_value(u.m),
            loc_xyz.y.to_value(u.m),
            loc_xyz.z.to_value(u.m),
        ]
    )

    ra = 3.26 * u.rad
    dec = -1.05 * u.rad
    time = Time("2024-01-01T00:00:00", scale="utc")

    uvw_from_xyz = generate_via_astropy(xyz_m, ra, dec, time)
    uvw_from_latlon = generate_via_astropy(lat_lon_deg, ra, dec, time)

    assert uvw_from_xyz.shape == uvw_from_latlon.shape
    np.testing.assert_allclose(uvw_from_latlon, uvw_from_xyz, rtol=1e-9, atol=1e-6)


@pytest.mark.unit
def test_generate_via_astropy_rejects_invalid_shape():
    """Invalid antenna position shape should raise a clear error."""
    with pytest.raises(ValueError, match=r"shape \(N, 3\) or \(N, 2\)"):
        generate_via_astropy(
            np.ones((3, 4), dtype=np.float64), 3.26 * u.rad, -1.05 * u.rad, Time.now()
        )
