from unittest.mock import patch
import unittest
import pandas as pd
from app.alma_utils import (
    get_tap_service,
    query_by_targets,
    query_by_science,
    fetch_science_types,
    validate_science_filters,
)
from pyvo.dal import DALServiceError
from pytest import raises

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
