import unittest
from unittest.mock import patch, mock_open
from app.alma_utils import (
    estimate_alma_beam_size,
    get_fov_from_band,
    generate_antenna_config_file_from_antenna_array,
    get_max_baseline_from_antenna_config,
    get_max_baseline_from_antenna_array,
    query_by_targets,
    validate_science_filters, compute_distance,
)
from astropy import units as u
import pandas as pd
import os
import numpy as np

class TestAlmaUtilsRemaining(unittest.TestCase):
    def test_estimate_alma_beam_size(self):
        """Test estimate_alma_beam_size with valid input."""
        result = estimate_alma_beam_size(230 * u.GHz, 1.5 * u.km)  # Central frequency in GHz, max baseline in km
        self.assertGreater(result, 0)

    def test_estimate_alma_beam_size_not_quantity(self):
        result = estimate_alma_beam_size(230, 1.5)  # Central frequency in GHz, max baseline in km
        self.assertGreater(result, 0)
    def test_compute_distance(self):
        """Test compute_distance"""
        result = compute_distance(1, 1, 1, 2, 2, 2)
        self.assertGreaterEqual(result, 0)

    def test_estimate_alma_beam_size_invalid_input(self):
        """Test estimate_alma_beam_size with invalid input."""
        with self.assertRaises(ValueError):
            estimate_alma_beam_size(-230, 1.5)
        with self.assertRaises(ValueError):
            estimate_alma_beam_size(230, -1.5)
        with self.assertRaises(ValueError):
            estimate_alma_beam_size(0, 0)

    def test_get_fov_from_band(self):
        """Test get_fov_from_band with valid band input."""
        result = get_fov_from_band(6)
        self.assertGreater(result, 0)

    def test_get_fov_from_band_invalid_band(self):
        """Test get_fov_from_band with invalid band input."""
        with self.assertRaises(ValueError):
            get_fov_from_band(11)  # Invalid band
        with self.assertRaises(ValueError):
            get_fov_from_band(-1)

    @patch("app.alma_utils.pd.read_csv")
    @patch("builtins.open")
    def test_generate_antenna_config_file_from_antenna_array(self, mock_open, mock_read_csv):
        """Test generate_antenna_config_file_from_antenna_array."""
        mock_read_csv.return_value = pd.DataFrame({
            "name": ["A1", "A2"],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "z": [1.0, 2.0],
        })

        generate_antenna_config_file_from_antenna_array(
            "A1 A2", "test_master_path", "test_output_dir"
        )

        mock_open.assert_called_once()
        mock_read_csv.assert_called_once_with(os.path.join("test_master_path", "antenna_config", "antenna_coordinates.csv"))

    @patch("app.alma_utils.pd.read_csv", side_effect=FileNotFoundError("File not found"))
    @patch("builtins.open", new_callable=mock_open)
    def test_generate_antenna_config_file_file_not_found(self, mock_open, mock_read_csv):
        """Test generate_antenna_config_file_from_antenna_array with missing file."""
        mock_open.side_effect = FileNotFoundError("File not found")
        with self.assertRaises(FileNotFoundError):
            generate_antenna_config_file_from_antenna_array("A1 A2", "test_master_path", "test_output_dir")

    @patch("app.alma_utils.np.loadtxt", return_value=np.array([[-33.8941259635723, -712.751648379266, -2.33008949622916], [-17.4539088663487, -709.608697236419, -2.32867141985557], [25.2459389865028, -744.431844526921, -2.33668425597284]]))
    def test_get_max_baseline_from_antenna_config(self, mock_loadtxt):
        """Test get_max_baseline_from_antenna_config."""
        result = get_max_baseline_from_antenna_config("antenna.cfg")
        self.assertGreater(result, 0)
        mock_loadtxt.assert_called_once_with("antenna.cfg", usecols=(0, 1, 2), comments="#", dtype=float)

    @patch("app.alma_utils.np.loadtxt", side_effect=FileNotFoundError("File not found"))
    def test_get_max_baseline_from_antenna_config_file_not_found(self, mock_loadtxt):
        """Test get_max_baseline_from_antenna_config with missing file."""
        with self.assertRaises(FileNotFoundError):
            get_max_baseline_from_antenna_config("missing_file.cfg")

    @patch("app.alma_utils.pd.read_csv")
    def test_get_max_baseline_from_antenna_array(self, mock_read_csv):
        """Test get_max_baseline_from_antenna_array."""
        mock_read_csv.return_value = pd.DataFrame({
            "name": ["A1", "A2"],
            "x": [1.0, 4.0],
            "y": [2.0, 5.0],
            "z": [3.0, 6.0],
        })

        result = get_max_baseline_from_antenna_array("A1 A2", "test_master_path")
        self.assertGreater(result, 0)
        mock_read_csv.assert_called_once_with(os.path.join("test_master_path", "antenna_config", "antenna_coordinates.csv"))

    @patch("app.alma_utils.pd.read_csv", return_value=pd.DataFrame(columns=["name", "x", "y", "z"]))
    def test_get_max_baseline_from_antenna_array_no_antennas(self, mock_read_csv):
        """Test get_max_baseline_from_antenna_array with no antennas."""
        with patch("app.alma_utils.np.linalg.norm", return_value=np.array([[0]])):
            result = get_max_baseline_from_antenna_array("A1 A2", "test_master_path")
            self.assertEqual(result, 0)
        mock_read_csv.assert_called_once_with(os.path.join("test_master_path", "antenna_config", "antenna_coordinates.csv"))

    @patch("app.alma_utils.query_alma", return_value=pd.DataFrame())
    def test_query_by_targets_empty_list(self, mock_query):
        """Test query_by_targets with an empty target list."""
        targets = []
        with patch("app.alma_utils.generate_query_for_targets", return_value="SELECT * FROM ivoa.obscore WHERE 1=0"):
            result = query_by_targets(targets)
            self.assertTrue(result.empty)
        mock_query.assert_called_once()

    def test_validate_science_filters_missing_key(self):
        """Test exception when required keys are missing in science filters."""
        invalid_filters = {"invalid_key": (10, 20)}  # Invalid key
        with self.assertRaises(ValueError) as context:
            validate_science_filters(invalid_filters)
        self.assertIn("Invalid key", str(context.exception))

if __name__ == "__main__":
    unittest.main()
