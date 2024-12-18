from unittest.mock import patch, mock_open
import unittest
import pandas as pd
import os
import numpy as np
from app.alma_utils import (
    get_tap_service,
    query_by_targets,
    query_by_science,
    fetch_science_types,
    validate_science_filters,
    estimate_alma_beam_size,
    get_fov_from_band,
    generate_antenna_config_file_from_antenna_array,
    get_max_baseline_from_antenna_config,
    get_max_baseline_from_antenna_array,
)
from pyvo.dal import DALServiceError
import astropy.units as u

class TestAlmaUtils(unittest.TestCase):
    def test_get_tap_service_success(self):
        with patch("app.alma_utils.TAPService") as mock_tap:
            mock_tap.return_value.search.return_value = True
            service = get_tap_service()
            self.assertIsNotNone(service)

    def test_get_tap_service_failure(self):
        with patch("app.alma_utils.TAPService") as mock_tap:
            mock_tap.side_effect = Exception("Service Error")
            with self.assertRaises(Exception):
                get_tap_service()

    def test_query_by_targets(self):
        targets = [("NGC253", "uid://A001/X122/X1")]
        with patch("app.alma_utils.query_alma") as mock_query:
            mock_query.return_value = "Mock Result"
            result = query_by_targets(targets)
            self.assertEqual(result, "Mock Result")

    def test_query_by_science(self):
        filters = {"science_keywords": ["AGN"], "bands": [3, 7]}
        with patch("app.alma_utils.query_alma") as mock_query:
            mock_query.return_value = "Mock Result"
            result = query_by_science(filters)
            self.assertEqual(result, "Mock Result")

    def test_fetch_science_types(self):
        mock_response = pd.DataFrame({
            "science_keyword": ["AGN", "Star Formation"],
            "scientific_category": ["Galaxy Dynamics", "ISM"],
        })
        with patch("app.alma_utils.query_alma") as mock_query:
            mock_query.return_value = mock_response
            keywords, categories = fetch_science_types()
            self.assertEqual(keywords, ["AGN", "Star Formation"])
            self.assertEqual(categories, ["Galaxy Dynamics", "ISM"])

    def test_get_tap_service_failures(self):
        """Test exception when all TAP services fail."""
        with patch("app.alma_utils.TAPService", side_effect=Exception("Service Error")):
            with self.assertRaises(Exception) as context:
                get_tap_service()
            self.assertEqual(str(context.exception), "All TAP services failed!")

    def test_validate_science_filters_invalid_key(self):
        """Test exception when invalid keys are present in filters."""
        invalid_filters = {"invalid_key": ["AGN"]}
        with self.assertRaises(ValueError) as context:
            validate_science_filters(invalid_filters)
        self.assertIn("Invalid key", str(context.exception))

    def test_validate_science_filters_invalid_range(self):
        """Test exception when invalid ranges are provided in filters."""
        invalid_filters = {"frequency": "invalid_range"}
        with self.assertRaises(ValueError) as context:
            validate_science_filters(invalid_filters)
        self.assertIn("must be a tuple", str(context.exception))

    def test_query_by_targets_service_error(self):
        """Test exception when querying targets fails due to service error."""
        targets = [("NGC253", "uid://A001/X122/X1")]
        with patch("app.alma_utils.query_alma", side_effect=DALServiceError("Query Failed")):
            with self.assertRaises(DALServiceError) as context:
                query_by_targets(targets)
            self.assertIn("Query Failed", str(context.exception))

    def test_query_by_science_service_error(self):
        """Test exception when querying science filters fails due to service error."""
        science_filters = {"science_keywords": ["AGN"]}
        with patch("app.alma_utils.query_alma", side_effect=DALServiceError("Query Failed")):
            with self.assertRaises(DALServiceError) as context:
                query_by_science(science_filters)
            self.assertIn("Query Failed", str(context.exception))

    def test_fetch_science_types_service_error(self):
        """Test exception when fetching science types fails due to service error."""
        with patch("app.alma_utils.query_alma", side_effect=DALServiceError("Fetch Failed")):
            with self.assertRaises(DALServiceError) as context:
                fetch_science_types()
            self.assertIn("Fetch Failed", str(context.exception))

    def test_estimate_alma_beam_size(self):
        """Test estimate_alma_beam_size with valid input."""
        result = estimate_alma_beam_size(230 * u.GHz, 1.5 * u.km)  # Central frequency in GHz, max baseline in km
        self.assertGreater(result, 0)

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
        mock_read_csv.assert_called_once_with(
            os.path.join("test_master_path", "antenna_config", "antenna_coordinates.csv"))

    @patch("app.alma_utils.pd.read_csv", side_effect=FileNotFoundError("File not found"))
    @patch("builtins.open", new_callable=mock_open)
    def test_generate_antenna_config_file_file_not_found(self, mock_open, mock_read_csv):
        """Test generate_antenna_config_file_from_antenna_array with missing file."""
        mock_open.side_effect = FileNotFoundError("File not found")
        with self.assertRaises(FileNotFoundError):
            generate_antenna_config_file_from_antenna_array("A1 A2", "test_master_path", "test_output_dir")

    @patch("app.alma_utils.np.loadtxt", return_value=np.array(
        [[-33.8941259635723, -712.751648379266, -2.33008949622916],
         [-17.4539088663487, -709.608697236419, -2.32867141985557],
         [25.2459389865028, -744.431844526921, -2.33668425597284]]))
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
        mock_read_csv.assert_called_once_with(
            os.path.join("test_master_path", "antenna_config", "antenna_coordinates.csv"))

    @patch("app.alma_utils.pd.read_csv", side_effect=FileNotFoundError("File not found"))
    def test_get_max_baseline_from_antenna_array_file_not_found(self, mock_read_csv):
        """Test get_max_baseline_from_antenna_array with missing file."""
        with self.assertRaises(FileNotFoundError):
            get_max_baseline_from_antenna_array("A1 A2", "missing_master_path")


if __name__ == "__main__":
    unittest.main()
