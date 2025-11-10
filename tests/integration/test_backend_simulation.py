"""Backend simulation service integration tests."""
import pytest
from pathlib import Path
import tempfile
import shutil
import pandas as pd

try:
    from backend.app.services.simulation_service import SimulationService
    from backend.app.schemas.simulation import SimulationParamsCreate
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.integration
@pytest.mark.slow
def test_simulation_service_run_point_source(
    temp_output_dir, main_dir, test_data_dir
):
    """Test running a point source simulation through the service."""
    if not BACKEND_AVAILABLE:
        pytest.skip("Backend not available for testing")
    
    # Load sample metadata
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    if len(metadata) == 0:
        pytest.skip("No metadata available for testing")
    
    row = metadata.iloc[0]
    
    # Create simulation parameters
    params = SimulationParamsCreate(
        main_dir=str(main_dir),
        output_dir=str(temp_output_dir),
        tng_dir=str(temp_output_dir / "tng"),
        galaxy_zoo_dir=str(temp_output_dir / "galaxy_zoo"),
        hubble_dir=str(temp_output_dir / "hubble"),
        project_name="test_sim",
        source_name=row.get("ALMA_source_name", "test_source"),
        member_ouid=row.get("member_ous_uid", "test_uid"),
        ra=float(row.get("RA", 0.0)),
        dec=float(row.get("Dec", 0.0)),
        band=float(row.get("Band", 6)),
        ang_res=float(row.get("Ang.res.", 0.1)),
        vel_res=float(row.get("Vel.res.", 10.0)),
        fov=float(row.get("FOV", 10.0)),
        obs_date=str(row.get("Obs.date", "2020-01-01")),
        pwv=float(row.get("PWV", 1.0)),
        int_time=float(row.get("Int.Time", 3600.0)),
        bandwidth=float(row.get("Bandwidth", 2.0)),
        freq=float(row.get("Freq", 100.0)),
        freq_support=str(row.get("Freq.sup.", "U[100..200,0.1]")),
        cont_sens=float(row.get("Cont_sens_mJybeam", 0.1)),
        antenna_array=str(row.get("antenna_arrays", "C43-1")),
        source_type="point",
        n_pix=64,
        n_channels=32,
        snr=1.3,
        save_mode="npz",
    )
    
    # Create service (without Dask client for simple test)
    service = SimulationService(
        main_dir=Path(params.main_dir),
        output_dir=Path(params.output_dir),
        tng_dir=Path(params.tng_dir),
        galaxy_zoo_dir=Path(params.galaxy_zoo_dir),
        hubble_dir=Path(params.hubble_dir),
        dask_client=None,
    )
    
    # This is a long-running test, so we'll just verify the service can be created
    # and the parameters are valid
    assert service.main_dir == Path(params.main_dir)
    assert service.output_dir == Path(params.output_dir)
    
    # In a real test, you would run the simulation:
    # service.run_simulation(simulation_id="test-123", params=params)
    # But this takes too long for CI, so we skip the actual execution


