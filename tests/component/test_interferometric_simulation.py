"""Component tests for interferometric simulation."""
import pytest
import numpy as np
import astropy.units as U
import pandas as pd
from pathlib import Path

from almasim.services import interferometry
from almasim import skymodels


@pytest.fixture
def sample_skymodel():
    """Create a sample sky model array for testing."""
    # Create a simple 3D sky model: (n_channels, n_pix, n_pix)
    n_channels = 16
    n_pix = 32
    model = np.random.rand(n_channels, n_pix, n_pix) * 0.1  # Jy
    return model


@pytest.fixture
def sample_fits_header():
    """Create a sample FITS header for testing."""
    from astropy.io import fits
    
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = 32
    header['NAXIS2'] = 32
    header['NAXIS3'] = 16
    header['DATE-OBS'] = '2020-01-01T00:00:00'
    return header


@pytest.fixture
def sample_antenna_array(test_data_dir):
    """Load a realistic antenna array string from sample metadata."""
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    return metadata["antenna_arrays"].iloc[0]


@pytest.mark.component
def test_interferometer_initialization(main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array):
    """Test initializing the Interferometer class."""
    interferometer = interferometry.Interferometer(
        idx=0,
        skymodel=sample_skymodel,
        client=None,
        main_dir=str(main_dir),
        output_dir=str(tmp_path),
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        central_freq=100.0 * U.GHz,
        bandwidth=10.0 * U.GHz,
        fov=10.0,
        antenna_array=sample_antenna_array,
        noise=0.01,
        snr=1.3,
        integration_time=1.0,
        observation_date="2020-01-01",
        header=sample_fits_header,
        save_mode="npz",
        robust=0.0,
    )
    
    assert interferometer.idx == 0
    assert interferometer.skymodel.shape == sample_skymodel.shape
    assert interferometer.main_dir == str(main_dir)
    assert interferometer.output_dir == str(tmp_path)
    assert interferometer.antenna_array == sample_antenna_array


@pytest.mark.component
def test_interferometer_baseline_preparation(main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array):
    """Test baseline preparation in interferometer."""
    interferometer = interferometry.Interferometer(
        idx=0,
        skymodel=sample_skymodel,
        client=None,
        main_dir=str(main_dir),
        output_dir=str(tmp_path),
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        central_freq=100.0 * U.GHz,
        bandwidth=10.0 * U.GHz,
        fov=10.0,
        antenna_array=sample_antenna_array,
        noise=0.01,
        snr=1.3,
        integration_time=1.0,
        observation_date="2020-01-01",
        header=sample_fits_header,
        save_mode="npz",
        robust=0.0,
    )
    
    # Check that baseline-related attributes are set
    assert hasattr(interferometer, 'Nant')
    assert hasattr(interferometer, 'antPos')
    assert interferometer.Nant > 0
    assert len(interferometer.antPos) > 0


@pytest.mark.component
@pytest.mark.slow
def test_interferometer_simulation_run(main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array):
    """Test running a full interferometric simulation."""
    from dask.distributed import Client
    
    with Client() as client:
        interferometer = interferometry.Interferometer(
            idx=0,
            skymodel=sample_skymodel,
            client=client,
            main_dir=str(main_dir),
            output_dir=str(tmp_path),
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            central_freq=100.0 * U.GHz,
            bandwidth=10.0 * U.GHz,
            fov=10.0,
            antenna_array=sample_antenna_array,
            noise=0.01,
            snr=1.3,
            integration_time=0.1,  # Short integration for testing
            observation_date="2020-01-01",
            header=sample_fits_header,
            save_mode="npz",
            robust=0.0,
        )
        
        results = interferometer.run_interferometric_sim()
        
        assert results is not None
        # Check for snake_case keys (actual return format)
        assert 'model_cube' in results or 'modelCube' in results
        assert 'dirty_cube' in results or 'dirtyCube' in results
        assert 'model_vis' in results or 'modelVis' in results
        assert 'dirty_vis' in results or 'dirtyVis' in results
        
        # Check output files were created
        output_files = list(tmp_path.glob("**/*.npz"))
        assert len(output_files) > 0


@pytest.mark.component
def test_interferometer_progress_signal(main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array):
    """Test interferometer progress signal."""
    progress_values = []
    
    def progress_callback(value):
        progress_values.append(value)
    
    interferometer = interferometry.Interferometer(
        idx=0,
        skymodel=sample_skymodel,
        client=None,
        main_dir=str(main_dir),
        output_dir=str(tmp_path),
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        central_freq=100.0 * U.GHz,
        bandwidth=10.0 * U.GHz,
        fov=10.0,
        antenna_array=sample_antenna_array,
        noise=0.01,
        snr=1.3,
        integration_time=1.0,
        observation_date="2020-01-01",
        header=sample_fits_header,
        save_mode="npz",
        robust=0.0,
    )
    
    interferometer.progress_signal.connect(progress_callback)
    interferometer.progress_signal.emit(50)
    interferometer.progress_signal.emit(100)
    
    assert len(progress_values) == 2
    assert progress_values == [50, 100]


@pytest.mark.component
def test_interferometer_save_modes(main_dir, tmp_path, sample_skymodel, sample_fits_header, sample_antenna_array):
    """Test different save modes for interferometer."""
    from dask.distributed import Client
    
    save_modes = ["npz", "fits"]
    
    for save_mode in save_modes:
        with Client() as client:
            interferometer = interferometry.Interferometer(
                idx=0,
                skymodel=sample_skymodel,
                client=client,
                main_dir=str(main_dir),
                output_dir=str(tmp_path / save_mode),
                ra=0.0 * U.deg,
                dec=0.0 * U.deg,
                central_freq=100.0 * U.GHz,
                bandwidth=10.0 * U.GHz,
                fov=10.0,
                antenna_array=sample_antenna_array,
                noise=0.01,
                snr=1.3,
                integration_time=0.05,  # Very short for testing
                observation_date="2020-01-01",
                header=sample_fits_header,
                save_mode=save_mode,
                robust=0.0,
            )
            
            # Just verify it can be initialized with different save modes
            assert interferometer.save_mode == save_mode

