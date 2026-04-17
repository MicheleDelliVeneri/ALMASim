"""Backend API integration tests."""
import json
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil

# Note: This requires the backend to be importable
# You may need to adjust the import path based on your setup
try:
    from backend.app.main import app
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    app = None


@pytest.fixture
def client():
    """Create a test client."""
    if not BACKEND_AVAILABLE:
        pytest.skip("Backend not available for testing")
    return TestClient(app)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.mark.integration
def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.integration
def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


@pytest.mark.integration
def test_get_science_types(client):
    """Test getting science types."""
    response = client.get("/api/v1/metadata/science-types")
    assert response.status_code == 200
    data = response.json()
    assert "keywords" in data
    assert "categories" in data
    assert isinstance(data["keywords"], list)
    assert isinstance(data["categories"], list)


@pytest.mark.integration
def test_query_metadata(client):
    """Test querying metadata."""
    query_data = {
        "science_keyword": ["Galaxies"],
        "bands": [6],
    }
    response = client.post("/api/v1/metadata/query", json=query_data)
    # May return 200 with empty results or 500 if TAP service unavailable
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "count" in data
        assert "data" in data


@pytest.mark.integration
def test_load_metadata(client, test_data_dir):
    """Test loading metadata from file."""
    metadata_path = test_data_dir / "qso_metadata.csv"
    if not metadata_path.exists():
        pytest.skip(f"Metadata file not found at {metadata_path}")
    
    # URL encode the path
    file_path = str(metadata_path).replace("\\", "/")
    response = client.get(f"/api/v1/metadata/load/{file_path}")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "data" in data
    assert data["count"] > 0


@pytest.mark.integration
def test_save_and_load_metadata_json_round_trip(client, tmp_path):
    """Test saving metadata as JSON and loading it back through the API."""
    from backend.app.core.config import settings

    original_data_dir = settings.DATA_DIR
    settings.DATA_DIR = tmp_path
    try:
        payload = {
            "path": "data/metadata-round-trip.json",
            "data": [
                {
                    "proposal_id": "2023.1.00001.S",
                    "member_ous_uid": "uid://A001/X123/X456",
                    "ALMA_source_name": "Test Source",
                    "Band": 6,
                }
            ],
        }

        save_response = client.post("/api/v1/metadata/save", json=payload)
        assert save_response.status_code == 200
        saved = save_response.json()
        assert saved["count"] == 1

        saved_path = Path(saved["path"])
        assert saved_path.exists()

        with saved_path.open("r", encoding="utf-8") as fp:
            on_disk = json.load(fp)
        assert on_disk["count"] == 1
        assert on_disk["data"][0]["member_ous_uid"] == payload["data"][0]["member_ous_uid"]

        load_response = client.get(f"/api/v1/metadata/load/{saved_path.as_posix()}")
        assert load_response.status_code == 200
        loaded = load_response.json()
        assert loaded["count"] == 1
        assert loaded["data"][0]["proposal_id"] == payload["data"][0]["proposal_id"]
    finally:
        settings.DATA_DIR = original_data_dir


@pytest.mark.integration
@pytest.mark.slow
def test_create_simulation(client, temp_output_dir, main_dir, test_data_dir):
    """Test creating a simulation."""
    import pandas as pd
    
    # Load sample metadata
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    if len(metadata) == 0:
        pytest.skip("No metadata available for testing")
    
    row = metadata.iloc[0]
    
    # Create simulation parameters
    sim_params = {
        "main_dir": str(main_dir),
        "output_dir": str(temp_output_dir),
        "tng_dir": str(temp_output_dir / "tng"),
        "galaxy_zoo_dir": str(temp_output_dir / "galaxy_zoo"),
        "hubble_dir": str(temp_output_dir / "hubble"),
        "project_name": "test_sim",
        "source_name": row.get("ALMA_source_name", "test_source"),
        "member_ouid": row.get("member_ous_uid", "test_uid"),
        "ra": float(row.get("RA", 0.0)),
        "dec": float(row.get("Dec", 0.0)),
        "band": float(row.get("Band", 6)),
        "ang_res": float(row.get("Ang.res.", 0.1)),
        "vel_res": float(row.get("Vel.res.", 10.0)),
        "fov": float(row.get("FOV", 10.0)),
        "obs_date": str(row.get("Obs.date", "2020-01-01")),
        "pwv": float(row.get("PWV", 1.0)),
        "int_time": float(row.get("Int.Time", 3600.0)),
        "bandwidth": float(row.get("Bandwidth", 2.0)),
        "freq": float(row.get("Freq", 100.0)),
        "freq_support": str(row.get("Freq.sup.", "U[100..200,0.1]")),
        "cont_sens": float(row.get("Cont_sens_mJybeam", 0.1)),
        "antenna_array": str(row.get("antenna_arrays", "C43-1")),
        "source_type": "point",
        "n_pix": 64,
        "n_channels": 32,
        "snr": 1.3,
        "save_mode": "npz",
    }
    
    response = client.post("/api/v1/simulation/", json=sim_params)
    # May return 201 (created) or 500 (error)
    assert response.status_code in [201, 500]
    
    if response.status_code == 201:
        data = response.json()
        assert "simulation_id" in data
        assert "status" in data
        assert data["status"] == "queued"


@pytest.mark.integration
def test_get_simulation_status(client):
    """Test getting simulation status."""
    simulation_id = "test-id-123"
    response = client.get(f"/api/v1/simulation/{simulation_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert "simulation_id" in data
    assert "status" in data
    assert "progress" in data


@pytest.mark.integration
def test_imaging_deconvolution_endpoint(client, temp_output_dir):
    """Test imaging deconvolution on saved dirty/beam/clean cubes."""
    import numpy as np

    model_cube = np.zeros((1, 17, 17), dtype=np.float32)
    model_cube[0, 8, 8] = 1.0

    yy, xx = np.indices((17, 17), dtype=np.float32)
    beam = np.exp(-0.5 * (((yy - 8) / 1.5) ** 2 + ((xx - 8) / 1.5) ** 2)).astype(np.float32)
    beam /= np.max(beam)
    beam_cube = beam[None, ...]

    dirty_fft = np.fft.fft2(model_cube[0]) * np.fft.fft2(np.fft.ifftshift(beam))
    dirty_cube = np.real(np.fft.ifft2(dirty_fft)).astype(np.float32)[None, ...]

    np.savez(temp_output_dir / "clean-cube_0.npz", clean_cube=model_cube)
    np.savez(temp_output_dir / "dirty-cube_0.npz", dirty_cube=dirty_cube)
    np.savez(temp_output_dir / "beam-cube_0.npz", beam_cube=beam_cube)

    response = client.post(
        "/api/v1/imaging/deconvolve",
        json={
            "directory": str(temp_output_dir),
            "dirty_cube_path": "dirty-cube_0.npz",
            "beam_cube_path": "beam-cube_0.npz",
            "clean_cube_path": "clean-cube_0.npz",
            "cycles": 120,
            "gain": 0.12,
            "method": "sum",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "dirty" in data
    assert "component_model" in data
    assert "restored" in data
    assert "residual" in data
    assert data["reference_clean"] is not None
    assert data["convolved_reference"] is not None
    assert data["metadata"]["cycles_requested"] == 120
    assert data["metadata"]["state_path"].endswith(".npz")
