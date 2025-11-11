"""Unit tests for interferometry imaging functions."""
import pytest
import numpy as np
from astropy.io import fits

from almasim.services.interferometry.imaging import (
    prepare_2d_arrays,
    _grid_uv,
    set_beam,
    check_lfac,
    _prepare_model,
    add_thermal_noise,
    set_primary_beam,
    observe,
    image_channel,
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
    header['NAXIS'] = 3
    header['NAXIS1'] = 32
    header['NAXIS2'] = 32
    header['NAXIS3'] = 16
    return header


@pytest.mark.unit
def test_prepare_2d_arrays(sample_npix):
    """Test preparing 2D arrays for interferometric processing."""
    result = prepare_2d_arrays(sample_npix)
    
    assert len(result) == 8
    beam, totsampling, dirtymap, noisemap, robustsamp, Gsampling, Grobustsamp, GrobustNoise = result
    
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
    # For no adjustment with lfac=1.0e6: mw >= 0.1 (so Xmax >= 0.1 * 1.05 * 1e6 / 2.0 = 52500)
    # But we want mw < 0.1 to NOT trigger adjustment, so we need a different lfac
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
    assert 'BMAJ' in sample_header
    assert 'BMIN' in sample_header
    assert 'BPA' in sample_header


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
    
    pixpos, totsampling_out, Gsampling_out, noisemap_out, UVpixsize, baseline_phases, bas2change, robfac = result
    
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
    from dask.distributed import Client
    
    Npix = 32
    Nant = 12
    Hcov = [0.0, 1.0, 2.0]
    nH = 60
    noise = 0.01
    # antPos needs to be list of lists with 2 elements each [x, y]
    antPos = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0],
              [50.0, 0.0], [0.0, 50.0], [50.0, 50.0], [25.0, 25.0],
              [75.0, 25.0], [25.0, 75.0], [75.0, 75.0], [50.0, 100.0]]
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
    
    with Client() as client:
        result = image_channel(
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
        
        # Compute the delayed result
        computed = client.compute(result).result()
        
        modelim, dirtymap, modelvis, dirtyvis, u, v, beam, totsampling = computed
        
        assert modelim.shape == (Npix, Npix)
        assert dirtymap.shape == (Npix, Npix)
        assert modelvis.shape == (Npix, Npix)
        assert dirtyvis.shape == (Npix, Npix)
        assert u.shape[0] > 0  # Should have baselines
        assert v.shape[0] > 0
        assert beam.shape == (Npix, Npix)
        assert totsampling.shape == (Npix, Npix)

