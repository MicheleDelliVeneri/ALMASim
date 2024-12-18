import unittest
from unittest.mock import patch, mock_open, Mock
from app.alma_utils import (
    estimate_alma_beam_size,
    get_fov_from_band,
    generate_antenna_config_file_from_antenna_array,
    generate_query_for_science,
    get_max_baseline_from_antenna_config,
    get_max_baseline_from_antenna_array,
    query_by_targets,
    get_tap_service,
    query_by_science,
    query_alma,
    fetch_science_types,
    validate_science_filters,
    compute_distance,
    TAP_URLS
)
from astropy import units as u
from pyvo.dal import DALServiceError
from tenacity import RetryError
from pyvo.dal import TAPService
import pandas as pd
import os
import numpy as np

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

    @patch("app.alma_utils.TAPService")
    @patch("app.alma_utils.logger")
    def test_tap_service_query_failed(self, mock_logger, mock_tap_service):
        """Test exception when TAP service query fails for a URL."""
        # Mock TAPService to simulate query failure for the first URL
        mock_service = Mock()
        mock_service.search.side_effect = [Exception("Query failed"), Mock()]
        mock_tap_service.side_effect = [mock_service, mock_service]

        result = get_tap_service()

        # Ensure the TAPService was called for both URLs
        self.assertEqual(mock_tap_service.call_count, 2)
        self.assertIsNotNone(result)

        # Verify the logger was called with the appropriate error message
        mock_logger.error.assert_any_call(
            "TAP service query failed for https://almascience.eso.org/tap: Query failed"
        )

    @patch("app.alma_utils.TAPService")
    def test_query_alma_success(self, mock_tap_service):
        """Test query_alma with successful query."""
        # Mock the behavior of service.search
        mock_service = mock_tap_service.return_value
        mock_search_result = Mock()
        mock_search_result.to_table.return_value.to_pandas.return_value = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        )
        mock_service.search.return_value = mock_search_result

        query = "SELECT * FROM ivoa.obscore"
        result = query_alma(mock_service, query)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ["col1", "col2"])
        mock_service.search.assert_called_once_with(query, response_timeout=120)

    def test_query_alma_failure(self):
        """Test query_alma with a query failure."""
        # Mock the behavior of service.search to raise an exception
        with patch("app.alma_utils.TAPService") as mock_tap_service:
            mock_service = mock_tap_service.return_value
            mock_service.search.side_effect = Exception("Query failed")

            query = "SELECT * FROM ivoa.obscore"
            with self.assertRaises(RetryError) as context:
                query_alma(mock_service, query)

            # Check the cause of the RetryError
            self.assertIn("Query failed", str(context.exception.last_attempt.exception()))
            mock_service.search.assert_called_with(query, response_timeout=120)

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

    def test_scientific_categories_filter(self):
        """Test query generation with scientific_categories filter."""
        science_filters = {
            "scientific_categories": ["Category1", "Category2"]
        }
        query = generate_query_for_science(science_filters)
        self.assertIn("scientific_category IN ('Category1', 'Category2')", query)

    def test_bands_filter(self):
        """Test query generation with bands filter."""
        science_filters = {
            "bands": [3, 7]
        }
        query = generate_query_for_science(science_filters)
        self.assertIn("band_list IN ('3', '7')", query)

    def test_frequency_filter(self):
        """Test query generation with frequency filter."""
        science_filters = {
            "frequency": (100.0, 200.0)
        }
        query = generate_query_for_science(science_filters)
        self.assertIn("frequency BETWEEN 100.0 AND 200.0", query)

    def test_spatial_resolution_filter(self):
        """Test query generation with spatial_resolution filter."""
        science_filters = {
            "spatial_resolution": (0.5, 1.5)
        }
        query = generate_query_for_science(science_filters)
        self.assertIn("spatial_resolution BETWEEN 0.5 AND 1.5", query)

    def test_velocity_resolution_filter(self):
        """Test query generation with velocity_resolution filter."""
        science_filters = {
            "velocity_resolution": (0.1, 1.0)
        }
        query = generate_query_for_science(science_filters)
        self.assertIn("velocity_resolution BETWEEN 0.1 AND 1.0", query)

    def test_combined_filters(self):
        """Test query generation with multiple filters."""
        science_filters = {
            "science_keywords": ["Keyword1"],
            "scientific_categories": ["Category1"],
            "bands": [3],
            "frequency": (100.0, 200.0),
            "spatial_resolution": (0.5, 1.5),
            "velocity_resolution": (0.1, 1.0)
        }
        query = generate_query_for_science(science_filters)
        self.assertIn("science_keyword IN ('Keyword1')", query)
        self.assertIn("scientific_category IN ('Category1')", query)
        self.assertIn("band_list IN ('3')", query)
        self.assertIn("frequency BETWEEN 100.0 AND 200.0", query)
        self.assertIn("spatial_resolution BETWEEN 0.5 AND 1.5", query)
        self.assertIn("velocity_resolution BETWEEN 0.1 AND 1.0", query)

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
