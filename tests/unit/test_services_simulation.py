"""Unit tests for simulation service functions."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from almasim.services.simulation import SimulationParams, run_simulation


@pytest.fixture
def sample_metadata_row_dict():
    """Create a sample metadata row as dictionary."""
    return {
        "ALMA_source_name": "J1234+5678",
        "member_ous_uid": "uid://A001/X123/X456",
        "Band": 6.0,
        "RA": 188.5,
        "Dec": -5.2,
        "Ang.res.": 0.1,
        "Vel.res.": 10.0,
        "FOV": 0.01,
        "Obs.date": "2020-01-01",
        "PWV": 2.0,
        "Int.Time": 3600.0,
        "Bandwidth": 1.875,
        "Freq": 250.0,
        "Freq.sup.": "[250.0..252.0GHz,31250.00kHz,2mJy/beam@10km/s,78.5uJy/beam@native, XX YY]",
        "Cont_sens_mJybeam": 0.1,
        "antenna_arrays": "A001:DA01 A002:DA02",
        "redshift": 0.5,
        "rest_frequency": 500.0,
        "lum_infrared": 1e10,
    }


@pytest.fixture
def sample_metadata_row_series(sample_metadata_row_dict):
    """Create a sample metadata row as pandas Series."""
    return pd.Series(sample_metadata_row_dict)


@pytest.mark.unit
def test_simulation_params_from_dict(tmp_path, sample_metadata_row_dict):
    """Test creating SimulationParams from dictionary."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    assert params.idx == 0
    assert params.source_name == "J1234+5678"
    assert params.member_ouid == "uid://A001/X123/X456"
    assert params.band == 6.0
    assert params.ra == 188.5
    assert params.dec == -5.2
    assert params.project_name == "test"


@pytest.mark.unit
def test_simulation_params_from_series(tmp_path, sample_metadata_row_series):
    """Test creating SimulationParams from pandas Series."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_series,
        idx=1,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    assert params.idx == 1
    assert params.source_name == "J1234+5678"
    assert isinstance(params.main_dir, str)
    assert Path(params.main_dir).exists()


@pytest.mark.unit
def test_simulation_params_missing_required_field(tmp_path, sample_metadata_row_dict):
    """Test that missing required fields raise KeyError."""
    # Remove required field
    del sample_metadata_row_dict["RA"]
    
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    with pytest.raises(KeyError, match="Missing required metadata column"):
        SimulationParams.from_metadata_row(
            sample_metadata_row_dict,
            idx=0,
            main_dir=main_dir,
            output_dir=tmp_path / "output",
            tng_dir=tmp_path / "tng",
            galaxy_zoo_dir=tmp_path / "galaxy_zoo",
            hubble_dir=tmp_path / "hubble",
            project_name="test",
        )


@pytest.mark.unit
def test_simulation_params_alternative_column_names(tmp_path):
    """Test that alternative column names are accepted."""
    row = {
        "source_name": "J1234+5678",  # Alternative to ALMA_source_name
        "member_ouid": "uid://A001/X123/X456",
        "band": 6.0,  # Lowercase
        "RA": 188.5,
        "Dec": -5.2,
        "ang_res": 0.1,  # Alternative format
        "vel_res": 10.0,
        "fov": 0.01,
        "obs_date": "2020-01-01",
        "PWV": 2.0,
        "int_time": 3600.0,
        "bandwidth": 1.875,
        "frequency": 250.0,  # Alternative to Freq
        "frequency_support": "[250.0..252.0GHz]",
        "cont_sens": 0.1,
        "antenna_array": "A001:DA01",
    }
    
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        row,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    assert params.source_name == "J1234+5678"
    assert params.band == 6.0
    assert params.freq == 250.0


@pytest.mark.unit
def test_simulation_params_nan_handling(tmp_path, sample_metadata_row_dict):
    """Test that NaN values are handled correctly."""
    sample_metadata_row_dict["n_pix"] = np.nan
    sample_metadata_row_dict["n_channels"] = np.nan
    
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    assert params.n_pix is None
    assert params.n_channels is None


@pytest.mark.unit
def test_simulation_params_path_resolution(tmp_path, sample_metadata_row_dict):
    """Test that paths are properly resolved."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    # Use relative paths
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir="./main",
        output_dir="./output",
        tng_dir="./tng",
        galaxy_zoo_dir="./galaxy_zoo",
        hubble_dir="./hubble",
        project_name="test",
    )
    
    # Paths should be resolved to absolute
    assert Path(params.main_dir).is_absolute()
    assert Path(params.output_dir).is_absolute()


@pytest.mark.unit
def test_simulation_params_line_names_string(tmp_path, sample_metadata_row_dict):
    """Test parsing line_names from string format."""
    # line_names parsing happens in run_simulation, not in from_metadata_row
    # So we just test that it's stored as-is
    sample_metadata_row_dict["line_names"] = '["CO(3-2)", "CII"]'
    
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        line_names=sample_metadata_row_dict["line_names"],
    )
    
    # Should be stored as string (parsing happens in run_simulation)
    assert isinstance(params.line_names, str) or params.line_names is None


@pytest.mark.unit
def test_simulation_params_defaults(tmp_path, sample_metadata_row_dict):
    """Test that default values are applied correctly."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    assert params.source_type == "point"  # Default
    assert params.snr == 1.3  # Default
    assert params.save_mode == "npz"  # Default
    assert params.inject_serendipitous is False  # Default
    assert params.remote is False  # Default
    assert params.ncpu >= 1  # Should use CPU count or 1


@pytest.mark.unit
def test_simulation_params_overrides(tmp_path, sample_metadata_row_dict):
    """Test that parameter overrides work correctly."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="gaussian",
        snr=2.5,
        save_mode="fits",
        n_pix=256,
        n_channels=64,
        inject_serendipitous=True,
        remote=True,
    )
    
    assert params.source_type == "gaussian"
    assert params.snr == 2.5
    assert params.save_mode == "fits"
    assert params.n_pix == 256
    assert params.n_channels == 64
    assert params.inject_serendipitous is True
    assert params.remote is True


@pytest.mark.unit
@patch('almasim.services.simulation.process_spectral_data')
@patch('almasim.services.simulation.usm')
@patch('almasim.services.simulation.uin')
def test_run_simulation_point_source(
    mock_interferometry,
    mock_skymodels,
    mock_process_spectral,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Test running simulation with point source."""
    from dask.distributed import Client
    
    # Setup mocks
    mock_process_spectral.return_value = (
        np.ones(32) * 0.1,  # continuum
        np.array([1.0, 0.5]),  # line_fluxes
        ["CO(3-2)", "CII"],  # line_names
        0.5,  # redshift
        np.array([250.0, 350.0]) * 1e9,  # line_frequency
        [10, 11],  # source_channel_index
        32,  # n_channels_nw
        1.875,  # bandwidth
        0.1,  # freq_support
        np.array([250.0]) * 1e9,  # cont_frequencies
        [2.0, 2.0],  # fwhm_z
        1e10,  # lum_infrared
    )
    
    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(32, 32, 32) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (16, 16, 0)
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 32
    
    mock_pointlike = Mock()
    mock_pointlike.insert.return_value = mock_datacube
    mock_skymodels.PointlikeSkyModel.return_value = mock_pointlike
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = Mock()
    
    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.return_value = {
        'model_cube': np.random.rand(32, 32, 32),
        'dirty_cube': np.random.rand(32, 32, 32),
    }
    mock_interferometry.Interferometer.return_value = mock_interferometer
    
    # Create params
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="point",
    )
    
    with Client() as client:
        results = run_simulation(
            params,
            logger=print,
            dask_client=client,
        )
    
    assert results is not None
    assert 'model_cube' in results or 'dirty_cube' in results
    mock_pointlike.insert.assert_called_once()


@pytest.mark.unit
@patch('almasim.services.simulation.process_spectral_data')
@patch('almasim.services.simulation.usm')
@patch('almasim.services.simulation.uin')
def test_run_simulation_gaussian_source(
    mock_interferometry,
    mock_skymodels,
    mock_process_spectral,
    tmp_path,
    sample_metadata_row_dict,
    main_dir,
):
    """Test running simulation with Gaussian source."""
    from dask.distributed import Client
    
    # Setup mocks
    mock_process_spectral.return_value = (
        np.ones(32) * 0.1,
        np.array([1.0]),
        ["CO(3-2)"],
        0.5,
        np.array([250.0]) * 1e9,
        [10],
        32,
        1.875,
        0.1,
        np.array([250.0]) * 1e9,
        [2.0],
        1e10,
    )
    
    mock_datacube = Mock()
    mock_datacube._array = Mock()
    mock_datacube._array.to_value.return_value = np.random.rand(32, 32, 32) * 0.1
    mock_datacube.wcs = Mock()
    mock_datacube.wcs.sub.return_value.wcs_world2pix.return_value = (16, 16, 0)
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 32
    
    mock_gaussian = Mock()
    mock_gaussian.insert.return_value = mock_datacube
    mock_skymodels.GaussianSkyModel.return_value = mock_gaussian
    mock_skymodels.DataCube.return_value = mock_datacube
    mock_skymodels.get_datacube_header.return_value = Mock()
    
    mock_interferometer = Mock()
    mock_interferometer.run_interferometric_sim.return_value = {
        'model_cube': np.random.rand(32, 32, 32),
    }
    mock_interferometry.Interferometer.return_value = mock_interferometer
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="gaussian",
    )
    
    with Client() as client:
        results = run_simulation(
            params,
            logger=print,
            dask_client=client,
        )
    
    assert results is not None
    mock_gaussian.insert.assert_called_once()


@pytest.mark.unit
def test_simulation_params_dataclass_fields(tmp_path, sample_metadata_row_dict):
    """Test that SimulationParams has all required fields."""
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    # Check that all expected fields exist and are accessible
    required_fields = [
        'idx', 'source_name', 'member_ouid', 'main_dir', 'output_dir',
        'tng_dir', 'galaxy_zoo_dir', 'hubble_dir', 'project_name',
        'ra', 'dec', 'band', 'ang_res', 'vel_res', 'fov', 'obs_date',
        'pwv', 'int_time', 'bandwidth', 'freq', 'freq_support',
        'cont_sens', 'antenna_array', 'n_pix', 'n_channels',
        'source_type', 'tng_api_key', 'ncpu', 'rest_frequency',
        'redshift', 'lum_infrared', 'snr', 'n_lines', 'line_names',
        'save_mode', 'inject_serendipitous', 'remote',
    ]
    
    # Verify all fields exist and can be accessed
    for field in required_fields:
        assert hasattr(params, field), f"Missing field: {field}"
        getattr(params, field)  # Should not raise AttributeError


@pytest.mark.unit
def test_simulation_params_clean_function(tmp_path, sample_metadata_row_dict):
    """Test the clean function handles None and NaN correctly."""
    sample_metadata_row_dict["n_pix"] = None
    sample_metadata_row_dict["n_channels"] = np.nan
    
    main_dir = tmp_path / "main"
    main_dir.mkdir()
    
    params = SimulationParams.from_metadata_row(
        sample_metadata_row_dict,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path / "output",
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )
    
    # Clean function should convert None/NaN to None
    assert params.n_pix is None
    assert params.n_channels is None

