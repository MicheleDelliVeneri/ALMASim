from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
import pandas as pd

client = TestClient(app)


def test_get_science_keywords():
    with patch("app.main.fetch_science_types") as mock_fetch:
        mock_fetch.return_value = (["AGN"], ["Galaxy Dynamics"])
        response = client.get("/science_keywords/")
        assert response.status_code == 200
        assert response.json() == {
            "science_keywords": ["AGN"],
            "scientific_categories": ["Galaxy Dynamics"],
        }


def test_query_targets():
    targets = [
        {"target_name": "NGC253", "member_ous_uid": "uid://A001/X122/X1"}
    ]
    with patch("app.main.query_by_targets") as mock_query:
        mock_query.return_value = pd.DataFrame(
            {"target_name": ["NGC253"], "frequency": [230.5]}
        )
        response = client.post("/query_targets/", json=targets)
        assert response.status_code == 200
        assert "target_name" in response.json()[0]


def test_query_science():
    filters = {"science_keywords": ["AGN"], "bands": [3, 7]}
    with patch("app.main.query_by_science") as mock_query:
        mock_query.return_value = pd.DataFrame(
            {"target_name": ["NGC253"], "frequency": [230.5]}
        )
        response = client.post("/query_science/", json=filters)
        assert response.status_code == 200
        assert "target_name" in response.json()[0]
