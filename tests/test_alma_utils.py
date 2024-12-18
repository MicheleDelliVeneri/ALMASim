from unittest.mock import patch
import unittest
import pandas as pd
from app.alma_utils import (
    get_tap_service,
    query_by_targets,
    query_by_science,
    fetch_science_types,
)


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

