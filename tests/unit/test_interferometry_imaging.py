"""Unit tests for interferometry imaging functions."""

import numpy as np
import pytest
import astropy.units as astrounits
from astropy.io import fits

from almasim.services.interferometry.imaging import (
    _grid_uv,
    _prepare_model,
    add_thermal_noise,
    check_lfac,
    image_channel,
    image_channel_ducc0,
    observe,
    prepare_2d_arrays,
    set_beam,
    set_primary_beam,
)


@pytest.fixture
def sample_npix():
    """Sample number of pixels."""
    return 32


@pytest.fixture
def sample_arrays(sample_npix):
    """Create sample 2D arrays."""
    return prepare_2d_arrays(sample_npix)


@pytest.fixture
def sample_wavelength():
    """Sample wavelength list."""
    return [1.0, 1.1, 1.05]  # in mm


@pytest.fixture
def sample_header():
    """Create a sample FITS header."""
    header = fits.Header()
    header["NAXIS"] = 3
    header["NAXIS1"] = 32
    header["NAXIS2"] = 32
    header["NAXIS3"] = 16
    return header


@pytest.mark.unit
def test_prepare_2d_arrays(sample_npix):
    """Test preparing 2D arrays for interferometric processing."""
    result = prepare_2d_arrays(sample_npix)

    assert len(result) == 8
    (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    ) = result

    assert beam.shape == (sample_npix, sample_npix)
    assert totsampling.shape == (sample_npix, sample_npix)
    assert dirtymap.shape == (sample_npix, sample_npix)
    assert noisemap.shape == (sample_npix, sample_npix)
    assert robustsamp.shape == (sample_npix, sample_npix)
    assert Gsampling.shape == (sample_npix, sample_npix)
    assert Grobustsamp.shape == (sample_npix, sample_npix)
    assert GrobustNoise.shape == (sample_npix, sample_npix)

    assert beam.dtype == np.float32
    assert totsampling.dtype == np.float32
    assert dirtymap.dtype == np.float32
    assert noisemap.dtype == np.complex64
    assert robustsamp.dtype == np.float32
    assert Gsampling.dtype == np.complex64
    assert Grobustsamp.dtype == np.complex64
    assert GrobustNoise.dtype == np.complex64


@pytest.mark.unit
def test_set_beam(sample_arrays, sample_npix):
    """Test setting beam with robust weighting."""
    (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    ) = sample_arrays

    # Initialize arrays with some values
    totsampling[:] = 1.0
    Gsampling[:] = 1.0 + 1j
    noisemap[:] = 0.1 + 0.1j

    robfac = 0.1
    Nphf = sample_npix // 2

    result = set_beam(
        robfac,
        totsampling,
        robustsamp,
        Gsampling,
        GrobustNoise,
        Grobustsamp,
        noisemap,
        beam,
        Nphf,
    )

    beam_out, beamScale, robustsamp_out, GrobustNoise_out, Grobustsamp_out = result

    assert beam_out.shape == (sample_npix, sample_npix)
    assert isinstance(beamScale, (float, np.floating))
    assert robustsamp_out.shape == (sample_npix, sample_npix)
    assert GrobustNoise_out.shape == (sample_npix, sample_npix)
    assert Grobustsamp_out.shape == (sample_npix, sample_npix)
    assert beamScale > 0


@pytest.mark.unit
def test_check_lfac():
    """Test checking and adjusting length factor for UV coordinates."""
    # Test case 1: mw < 0.1 and lfac == 1.0e6
    # mw = 2.0 * Xmax / wavelength[2] / lfac
    # mw = 2.0 * 0.01 / 1.05 / 1e6 = 1.9e-8 < 0.1
    wavelength = [1.0, 1.1, 1.05]
    Xmax = 0.01
    lfac = 1.0e6
    result = check_lfac(Xmax, wavelength, lfac)
    lfac_out, ulab, vlab = result
    assert lfac_out == 1.0e3  # Should be adjusted down

    # Test case 2: mw >= 100.0 and lfac == 1.0e3
    # mw = 2.0 * Xmax / wavelength[2] / lfac
    # For mw >= 100.0: 2.0 * Xmax / 1.05 / 1e3 >= 100.0
    # Xmax >= 100.0 * 1.05 * 1e3 / 2.0 = 52500.0
    Xmax = 60000.0  # Large enough to trigger adjustment
    lfac = 1.0e3
    result = check_lfac(Xmax, wavelength, lfac)
    lfac_out, ulab, vlab = result
    assert lfac_out == 1.0e6  # Should be adjusted up

    # Test case 3: Normal case (no adjustment)
    # mw = 2.0 * Xmax / 1.05 / lfac
    # For no adjustment with lfac=1.0e6:
    # mw >= 0.1 (so Xmax >= 0.1 * 1.05 * 1e6 / 2.0 = 52500)
    # But we want mw < 0.1 to NOT trigger adjustment,
    # so we need a different lfac
    # Actually, if lfac=1.0e3 and mw < 100.0, no adjustment
    # mw = 2.0 * Xmax / 1.05 / 1e3 < 100.0 means Xmax < 52500.0
    Xmax = 1000.0  # mw = 2.0 * 1000 / 1.05 / 1e3 = 1.9 (between 0.1 and 100)
    lfac = 1.0e3
    result = check_lfac(Xmax, wavelength, lfac)
    lfac_out, ulab, vlab = result
    assert lfac_out == 1.0e3  # Should remain unchanged (no adjustment)
    assert "U (k$\\lambda$)" in ulab
    assert "V (k$\\lambda$)" in vlab


@pytest.mark.unit
def test_prepare_model_same_size(sample_npix):
    """Test preparing model when image size matches Nphf."""
    Nphf = sample_npix // 2
    Np4 = sample_npix // 4
    zooming = 1

    # Create image that matches Nphf
    img = np.random.rand(Nphf, Nphf).astype(np.float32)

    result = _prepare_model(sample_npix, img, Nphf, Np4, zooming)
    modelim, modelimTrue = result

    assert len(modelim) == 2
    assert modelimTrue.shape == (sample_npix, sample_npix)
    assert np.all(modelimTrue >= 0)  # Should clip negative values


@pytest.mark.unit
def test_prepare_model_different_size(sample_npix):
    """Test preparing model when image size differs from Nphf."""
    Nphf = sample_npix // 2
    Np4 = sample_npix // 4
    zooming = 1

    # Create image smaller than Nphf
    img = np.random.rand(Nphf // 2, Nphf // 2).astype(np.float32)

    result = _prepare_model(sample_npix, img, Nphf, Np4, zooming)
    modelim, modelimTrue = result

    assert len(modelim) == 2
    assert modelimTrue.shape == (sample_npix, sample_npix)
    assert np.all(modelimTrue >= 0)


@pytest.mark.unit
def test_add_thermal_noise():
    """Test adding thermal noise to image."""
    img = np.ones((32, 32), dtype=np.float32) * 1.0
    noise = 0.1

    result = add_thermal_noise(img, noise)

    assert result.shape == img.shape
    # Note: np.random.normal returns float64 by default, but function may convert
    assert result.dtype in (np.float32, np.float64)
    # Should have some variation (not all exactly 1.0)
    assert np.std(result) > 0


@pytest.mark.unit
def test_set_primary_beam(sample_header, sample_wavelength):
    """Test setting primary beam and calculating model FFT."""
    Npix = 32
    distmat = np.random.rand(Npix, Npix).astype(np.float32) * 100.0
    Diameters = [12.0]  # 12m antenna diameter
    modelim = [np.random.rand(Npix, Npix).astype(np.float32)]
    modelimTrue = np.random.rand(Npix, Npix).astype(np.float32)

    result = set_primary_beam(
        sample_header,
        distmat,
        sample_wavelength,
        Diameters,
        modelim,
        modelimTrue,
    )

    modelfft, modelim_out = result

    assert modelfft.shape == (Npix, Npix)
    assert len(modelim_out) == 1
    assert modelim_out[0].shape == (Npix, Npix)
    assert "BMAJ" in sample_header
    assert "BMIN" in sample_header
    assert "BPA" in sample_header


@pytest.mark.unit
def test_observe(sample_arrays, sample_npix):
    """Test observing model through interferometer."""
    (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    ) = sample_arrays

    # Initialize arrays
    GrobustNoise[:] = 0.1 + 0.1j
    Grobustsamp[:] = 1.0 + 0.5j
    modelfft = np.fft.fft2(np.random.rand(sample_npix, sample_npix))
    beamScale = 1.0

    result = observe(
        dirtymap,
        GrobustNoise,
        modelfft,
        Grobustsamp,
        beamScale,
    )

    dirtymap_out, modelvis, dirtyvis = result

    assert dirtymap_out.shape == (sample_npix, sample_npix)
    assert modelvis.shape == (sample_npix, sample_npix)
    assert dirtyvis.shape == (sample_npix, sample_npix)
    assert np.isrealobj(dirtymap_out)


@pytest.mark.unit
def test_grid_uv_basic(sample_arrays, sample_npix):
    """Test basic UV gridding functionality."""
    (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    ) = sample_arrays

    Nbas = 10
    Nphf = sample_npix // 2
    nH = 60
    imsize = 10.0  # arcsec
    robust = 0.0

    # Create sample UV coordinates
    u = np.random.randn(Nbas, nH) * 1000.0  # in meters
    v = np.random.randn(Nbas, nH) * 1000.0
    Gains = np.random.randn(Nbas, nH) + 1j * np.random.randn(Nbas, nH)
    Noise = np.random.randn(Nbas, nH) + 1j * np.random.randn(Nbas, nH)

    result = _grid_uv(
        Nbas,
        totsampling,
        Gsampling,
        noisemap,
        u,
        v,
        Nphf,
        Gains,
        Noise,
        robust,
        nH,
        imsize,
    )

    (
        pixpos,
        totsampling_out,
        Gsampling_out,
        noisemap_out,
        UVpixsize,
        baseline_phases,
        bas2change,
        robfac,
    ) = result

    assert len(pixpos) == Nbas
    assert totsampling_out.shape == (sample_npix, sample_npix)
    assert Gsampling_out.shape == (sample_npix, sample_npix)
    assert noisemap_out.shape == (sample_npix, sample_npix)
    assert isinstance(UVpixsize, (float, np.floating))
    assert UVpixsize > 0
    assert isinstance(robfac, (float, np.floating))
    assert len(baseline_phases) == Nbas


@pytest.mark.unit
@pytest.mark.slow
def test_image_channel(sample_header, sample_wavelength):
    """Test processing a single channel through interferometric simulation."""
    Npix = 32
    Nant = 12
    Hcov = [0.0, 1.0, 2.0]
    nH = 60
    noise = 0.01
    # antPos needs to be list of lists with 2 elements each [x, y]
    antPos = [
        [0.0, 0.0],
        [100.0, 0.0],
        [0.0, 100.0],
        [100.0, 100.0],
        [50.0, 0.0],
        [0.0, 50.0],
        [50.0, 50.0],
        [25.0, 25.0],
        [75.0, 25.0],
        [25.0, 75.0],
        [75.0, 75.0],
        [50.0, 100.0],
    ]
    # Ensure all positions have exactly 2 elements
    antPos = [[float(x), float(y)] for x, y in antPos]
    robfac = 0.1
    # trlat and trdec need 2 elements each (sin and cos components)
    trlat = [0.0, 1.0]
    trdec = [0.0, 1.0]
    Diameters = [12.0]
    imsize = 10.0
    Xmax = 1.0
    lfac = 1.0e6
    distmat = np.random.rand(Npix, Npix).astype(np.float32) * 100.0
    Nphf = Npix // 2
    Np4 = Npix // 4
    zooming = 1
    robust = 0.0

    # Create sample image
    img = np.random.rand(Nphf, Nphf).astype(np.float32) * 0.1

    modelim, dirtymap, modelvis, dirtyvis, u, v, beam, totsampling, raw_visibility = image_channel(
        img,
        sample_wavelength,
        Npix,
        Nant,
        Hcov,
        nH,
        noise,
        antPos,
        robfac,
        trlat,
        trdec,
        Diameters,
        imsize,
        Xmax,
        lfac,
        distmat,
        Nphf,
        Np4,
        zooming,
        sample_header,
        robust,
    )

    assert modelim.shape == (Npix, Npix)
    assert dirtymap.shape == (Npix, Npix)
    assert modelvis.shape == (Npix, Npix)
    assert dirtyvis.shape == (Npix, Npix)
    assert u.shape[0] > 0  # Should have baselines
    assert v.shape[0] > 0
    assert beam.shape == (Npix, Npix)
    assert totsampling.shape == (Npix, Npix)
    assert "data" in raw_visibility
    assert raw_visibility["uvw_m"].shape[1] == 3


@pytest.mark.unit
def test_image_channel_ducc0_requires_pointing_metadata(sample_wavelength):
    """image_channel_ducc0 should fail fast when header lacks pointing coordinates."""
    header = fits.Header()
    header["DATE-OBS"] = "2024-01-01T00:00:00"

    with pytest.raises(ValueError, match="pointing coordinates"):
        image_channel_ducc0(
            img=np.zeros((16, 16), dtype=np.float32),
            wavelengths=sample_wavelength,
            Npix=32,
            Nant=4,
            Hcov=[0.0, 1.0],
            nH=10,
            noise=0.01,
            antPos=[[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]],
            robfac=0.1,
            trlat=[0.0, 1.0],
            trdec=[0.0, 1.0],
            Diameters=[12.0],
            imsize=10.0,
            Xmax=1.0,
            lfac=1.0e6,
            distmat=np.zeros((32, 32), dtype=np.float32),
            Nphf=16,
            Np4=8,
            zooming=1,
            header=header,
            robust=0.0,
        )


@pytest.mark.unit
def test_image_channel_ducc0_validates_antpos_dimension(sample_wavelength):
    """image_channel_ducc0 should validate antenna position array dimensions."""
    header = fits.Header()
    header["DATE-OBS"] = "2024-01-01T00:00:00"
    header["OBSRA"] = 180.0
    header["OBSDEC"] = -30.0

    # Test with 1D array (invalid dimensions)
    with pytest.raises(ValueError, match="2D array-like"):
        image_channel_ducc0(
            img=np.zeros((16, 16), dtype=np.float32),
            wavelengths=sample_wavelength,
            Npix=32,
            Nant=4,
            Hcov=[0.0, 1.0],
            nH=10,
            noise=0.01,
            antPos=[0.0, 0.0, 100.0, 0.0],  # 1D array instead of 2D
            robfac=0.1,
            trlat=[0.0, 1.0],
            trdec=[0.0, 1.0],
            Diameters=[12.0],
            imsize=10.0,
            Xmax=1.0,
            lfac=1.0e6,
            distmat=np.zeros((32, 32), dtype=np.float32),
            Nphf=16,
            Np4=8,
            zooming=1,
            header=header,
            robust=0.0,
        )


@pytest.mark.unit
def test_image_channel_ducc0_happy_path(sample_wavelength, monkeypatch):
    """image_channel_ducc0 should process valid inputs without raising."""
    header = fits.Header()
    header["DATE-OBS"] = "2024-01-01T00:00:00"
    header["OBSRA"] = 180.0
    header["OBSDEC"] = -30.0
    
    # Valid 2D antenna positions (4 antennas with 3D ECEF coords in meters)
    valid_antpos = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [100.0, 100.0, 0.0],
    ], dtype=np.float64)
    
    # Mock the ducc0 predict/dirty calls to avoid incomplete implementation
    mock_visibilities = np.zeros((64, 8), dtype=np.complex64)
    mock_weights = np.ones((64, 8), dtype=np.float32)
    mock_dirtymap = np.zeros((32, 32), dtype=np.float32)
    
    def mock_image_based_predict(*args, **kwargs):
        return mock_visibilities, mock_weights
    
    def mock_vis2dirty(*args, **kwargs):
        return mock_dirtymap
    
    # Patch the functions that would fail due to incomplete implementation
    monkeypatch.setattr(
        "almasim.services.interferometry.imaging.image_based_predict",
        mock_image_based_predict
    )
    monkeypatch.setattr(
        "ducc0.wgridder.vis2dirty",
        mock_vis2dirty
    )
    
    # Should not raise and should return a tuple
    result = image_channel_ducc0(
        img=np.ones((16, 16), dtype=np.float32),
        wavelengths=sample_wavelength,
        Npix=32,
        Nant=4,
        Hcov=[0.0, 1.0],
        nH=10,
        noise=0.01,
        antPos=valid_antpos,
        robfac=0.1,
        trlat=[0.0, 1.0],
        trdec=[0.0, 1.0],
        Diameters=[12.0],
        imsize=10.0,
        Xmax=1.0,
        lfac=1.0e6,
        distmat=np.zeros((32, 32), dtype=np.float32),
        Nphf=16,
        Np4=8,
        zooming=1,
        header=header,
        robust=0.0,
    )
    
    # Verify result is a tuple (function completed without raising)
    assert isinstance(result, tuple)


@pytest.mark.unit
def test_image_channel_ducc0_integration():
    """Test image_channel_ducc0 with real ducc0 operations using realistic ALMA data."""
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    from almasim.services.interferometry.baselines import generate_via_astropy

    # ALMA reference geodetic position.
    alma_ref = EarthLocation.of_site("ALMA")
    alma_lat = alma_ref.geodetic[0].deg
    alma_lon = alma_ref.geodetic[1].deg

    # Local ENU offsets (meters) from antenna_coordinates.csv.
    local_enu = np.array(
        [
            [-33.8941259635723, -712.751648379266, -2.33008949622916],
            [-17.4539088663487, -709.608697236419, -2.32867141985557],
            [-22.5589292009902, -691.9838606867, -2.32374238442929],
            [-5.42101314454732, -723.791104543484, -2.33425051860951],
            [25.2459389865028, -744.431844526921, -2.33668425597284],
        ],
        dtype=np.float64,
    )

    # Convert ENU offsets to per-antenna geodetic coords using ALMA reference ECEF.
    lat_rad = np.deg2rad(alma_lat)
    lon_rad = np.deg2rad(alma_lon)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)

    alma_ecef = np.array(
        [
            alma_ref.x.to(astrounits.m).value,
            alma_ref.y.to(astrounits.m).value,
            alma_ref.z.to(astrounits.m).value,
        ],
        dtype=np.float64,
    )

    geodetic_antpos = []
    for e, n, u_up in local_enu:
        dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u_up
        dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u_up
        dz = cos_lat * n + sin_lat * u_up
        ant_ecef = alma_ecef + np.array([dx, dy, dz], dtype=np.float64)

        ant_loc = EarthLocation.from_geocentric(
            ant_ecef[0] * astrounits.m,
            ant_ecef[1] * astrounits.m,
            ant_ecef[2] * astrounits.m,
        )
        geodetic_antpos.append([ant_loc.geodetic[0].deg, ant_loc.geodetic[1].deg])

    geodetic_antpos = np.asarray(geodetic_antpos, dtype=np.float64)

    header = fits.Header()
    header["DATE-OBS"] = "2024-01-15T12:00:00"
    # Near zenith pointing at ALMA to keep W relatively small.
    header["OBSRA"] = alma_lon
    header["OBSDEC"] = alma_lat

    # Keep ALMA-band-like setup around 100 GHz while avoiding large multi-channel spread.
    frequencies_ghz = [100.0, 100.0, 100.0]
    wavelengths_m = [3e8 / (f * 1e9) for f in frequencies_ghz]

    # Validate UVW spread independently before calling ducc0 path.
    obs_time = Time(header["DATE-OBS"], format="isot", scale="utc")
    uvw_debug = generate_via_astropy(
        geodetic_antpos,
        float(header["OBSRA"]) * astrounits.deg,
        float(header["OBSDEC"]) * astrounits.deg,
        obs_time,
    )
    u_spread = float(np.ptp(uvw_debug[:, 0]))
    v_spread = float(np.ptp(uvw_debug[:, 1]))
    w_spread = float(np.ptp(uvw_debug[:, 2]))

    assert u_spread > 0.0
    assert v_spread > 0.0
    assert w_spread <= 1.2 * max(u_spread, v_spread)

    # Run real ducc0 computation (no mocks).
    result = image_channel_ducc0(
        img=np.ones((16, 16), dtype=np.float32),
        wavelengths=wavelengths_m,
        Npix=32,
        Nant=5,
        Hcov=[0.0, 1.0],
        nH=10,
        noise=0.01,
        antPos=geodetic_antpos,
        robfac=0.1,
        trlat=[alma_lat - 5.0, alma_lat + 5.0],
        trdec=[alma_lat - 5.0, alma_lat + 5.0],
        Diameters=[12.0],
        imsize=60.0,
        Xmax=1.0,
        lfac=1.0e6,
        distmat=np.zeros((32, 32), dtype=np.float32),
        Nphf=16,
        Np4=8,
        zooming=1,
        header=header,
        robust=0.0,
    )

    assert isinstance(result, tuple)
    assert len(result) == 9
    modelim, dirtymap, modelvis, dirtyvis, u_vals, v_vals, beam, totsampling, raw_visibility = result
    assert isinstance(modelim, np.ndarray)
    assert isinstance(dirtymap, np.ndarray)
    assert dirtymap.shape == (32, 32)
    assert isinstance(u_vals, np.ndarray)
    assert isinstance(v_vals, np.ndarray)
    assert isinstance(beam, np.ndarray)
    assert isinstance(totsampling, np.ndarray)
    assert isinstance(raw_visibility, dict)
    assert "uvw_m" in raw_visibility
    assert "visibilities" in raw_visibility
    assert "weights" in raw_visibility
