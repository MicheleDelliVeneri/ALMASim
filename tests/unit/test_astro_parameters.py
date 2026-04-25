"""Unit tests for astro parameters module."""

import numpy as np
import pytest

from almasim.services.astro.parameters import write_sim_parameters


@pytest.fixture
def sample_sim_params():
    """Create sample simulation parameters."""
    return {
        "source_name": "J1234+5678",
        "member_ouid": "uid://A001/X123/X456",
        "ra": 188.5,
        "dec": -5.2,
        "ang_res": 0.1,
        "vel_res": 10.0,
        "int_time": 3600.0,
        "band": 6.0,
        "band_range": 1.875,
        "central_freq": 250.0,
        "redshift": 0.5,
        "line_fluxes": np.array([1.0, 0.5]),
        "line_names": ["CO(3-2)", "CII"],
        "line_frequencies": np.array([250.0, 350.0]),
        "continum": np.array([0.1, 0.2, 0.15]),
        "fov": 10.0,
        "beam_size": 0.05,
        "cell_size": 0.01,
        "n_pix": 64,
        "n_channels": 32,
        "snapshot": None,
        "subhalo": None,
        "lum_infrared": 1e10,
        "fwhm_z": np.array([2.0, 3.0]),
        "source_type": "point",
    }


@pytest.mark.unit
def test_write_sim_parameters_point_source(tmp_path, sample_sim_params):
    """Test writing simulation parameters for point source."""
    output_file = tmp_path / "sim_params.txt"

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    content = output_file.read_text()

    assert "Simulation Parameters:" in content
    assert "J1234+5678" in content
    assert "uid://A001/X123/X456" in content
    assert "Band: 6.0" in content
    assert "Redshift: 0.5" in content
    assert "Cube Size: 64 x 64 x 32 pixels" in content
    assert "CO(3-2)" in content
    assert "CII" in content
    assert "FWHM_x" not in content  # Should not appear for point source
    assert "Projection Angle" not in content  # Should not appear for point source


@pytest.mark.unit
def test_write_sim_parameters_gaussian_source(tmp_path, sample_sim_params):
    """Test writing simulation parameters for gaussian source."""
    output_file = tmp_path / "sim_params_gaussian.txt"
    sample_sim_params["source_type"] = "gaussian"
    sample_sim_params["fwhm_x"] = 5.0
    sample_sim_params["fwhm_y"] = 6.0
    sample_sim_params["angle"] = 45.0

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    content = output_file.read_text()

    assert "FWHM_x (pixels): 5.0" in content
    assert "FWHM_y (pixels): 6.0" in content
    assert "Projection Angle: 45.0" in content


@pytest.mark.unit
def test_write_sim_parameters_extended_source(tmp_path, sample_sim_params):
    """Test writing simulation parameters for extended source."""
    output_file = tmp_path / "sim_params_extended.txt"
    sample_sim_params["source_type"] = "extended"
    sample_sim_params["snapshot"] = 99
    sample_sim_params["subhalo"] = 12345
    sample_sim_params["angle"] = 30.0

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    content = output_file.read_text()

    assert "Projection Angle: 30.0" in content
    assert "TNG Snapshot ID: 99" in content
    assert "TNG Subhalo ID: 12345" in content
    assert "FWHM_x" not in content  # Should not appear for extended source


@pytest.mark.unit
def test_write_sim_parameters_with_tng(tmp_path, sample_sim_params):
    """Test writing simulation parameters with TNG data."""
    output_file = tmp_path / "sim_params_tng.txt"
    sample_sim_params["snapshot"] = 50
    sample_sim_params["subhalo"] = 67890

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    content = output_file.read_text()

    assert "TNG Snapshot ID: 50" in content
    assert "TNG Subhalo ID: 67890" in content


@pytest.mark.unit
def test_write_sim_parameters_multiple_lines(tmp_path, sample_sim_params):
    """Test writing simulation parameters with multiple spectral lines."""
    output_file = tmp_path / "sim_params_multiline.txt"
    sample_sim_params["line_fluxes"] = np.array([1.0, 0.5, 0.3])
    sample_sim_params["line_names"] = ["CO(3-2)", "CII", "NII"]
    sample_sim_params["line_frequencies"] = np.array([250.0, 350.0, 400.0])
    sample_sim_params["fwhm_z"] = np.array([2.0, 3.0, 4.0])

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    content = output_file.read_text()

    assert "CO(3-2)" in content
    assert "CII" in content
    assert "NII" in content
    assert content.count("Line:") == 3


@pytest.mark.unit
def test_write_sim_parameters_creates_directory(tmp_path, sample_sim_params):
    """Test that write_sim_parameters creates parent directories."""
    output_file = tmp_path / "nested" / "dir" / "sim_params.txt"

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    assert output_file.parent.exists()


@pytest.mark.unit
def test_write_sim_parameters_continuum_calculation(tmp_path, sample_sim_params):
    """Test that continuum flux is correctly calculated and written."""
    output_file = tmp_path / "sim_params_continuum.txt"
    # Set specific continuum values
    sample_sim_params["continum"] = np.array([0.1, 0.2, 0.3, 0.4])
    expected_mean = 0.25

    write_sim_parameters(output_file, **sample_sim_params)

    content = output_file.read_text()
    assert f"Mean Continum Flux: {expected_mean}" in content


@pytest.mark.unit
def test_write_sim_parameters_gaussian_without_optional(tmp_path, sample_sim_params):
    """Test gaussian source without optional fwhm_x, fwhm_y, angle."""
    output_file = tmp_path / "sim_params_gaussian_minimal.txt"
    sample_sim_params["source_type"] = "gaussian"
    # Don't set fwhm_x, fwhm_y, angle

    write_sim_parameters(output_file, **sample_sim_params)

    assert output_file.exists()
    content = output_file.read_text()
    # Should not crash, but also shouldn't write None values
    assert "FWHM_x" not in content or "None" not in content
    assert "Projection Angle" not in content  # angle is None, so shouldn't write
