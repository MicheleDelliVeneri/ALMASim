import unittest
from unittest.mock import patch
import requests


class TestFrontend(unittest.TestCase):
    def test_fetch_science_keywords(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "science_keywords": ["AGN"],
                "scientific_categories": ["Galaxy Dynamics"],
            }
            response = requests.get("http://localhost:8000/science_keywords/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("AGN", response.json()["science_keywords"])

    def test_query_targets_frontend(self):
        targets = [
            {"target_name": "NGC253", "member_ous_uid": "uid://A001/X122/X1"}
        ]
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = [
                {"target_name": "NGC253", "frequency": 230.5}
            ]
            response = requests.post("http://localhost:8000/query_targets/", 
                                     json=targets)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["target_name"], "NGC253")

    def test_query_science_frontend(self):
        filters = {"science_keywords": ["AGN"], "bands": [3, 7]}
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = [
                {"target_name": "NGC253", "frequency": 230.5}
            ]
            response = requests.post("http://localhost:8000/query_science/", 
                                     json=filters)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["target_name"], "NGC253")


if __name__ == "__main__":
    unittest.main()
