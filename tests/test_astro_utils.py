import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
from app.astro_utils import (
    compute_redshift,
    redshift_to_snapshot,
    get_data_from_hdf,
    get_subhaloids_from_db,
    read_line_emission_csv,
    get_line_info,
    compute_rest_frequency_from_redshift,
    write_sim_parameters
)
from astropy import units as u

class TestAstroUtils(unittest.TestCase):
    def test_compute_redshift(self):
        """Test compute_redshift with valid input."""
        rest_frequency = 1420 * u.MHz
        observed_frequency = 710 * u.MHz
        result = compute_redshift(rest_frequency, observed_frequency)
        self.assertAlmostEqual(result, 1.0)

    def test_compute_redshift_invalid_input(self):
        """Test compute_redshift with invalid input."""
        with self.assertRaises(ValueError):
            compute_redshift(-1420 * u.MHz, 710 * u.MHz)
        with self.assertRaises(ValueError):
            compute_redshift(1420 * u.MHz, 1500 * u.MHz)

    def test_redshift_to_snapshot(self):
        """Test redshift_to_snapshot."""
        # Mock snap_db logic to ensure proper testing
        result = redshift_to_snapshot(0)
        self.assertEqual(result, 99)

    @patch("app.astro_utils.h5py.File")
    def test_get_data_from_hdf(self, mock_h5py):
        """Test get_data_from_hdf."""
        mock_group = {
            "Column1": np.array([1, 2, 3]),
            "Column2": np.array([4, 5, 6])
        }
        mock_h5py.return_value.__enter__.return_value = {"Snapshot_99": mock_group}

        result = get_data_from_hdf("test.hdf5", 99)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ["Column1", "Column2"])

    @patch("app.astro_utils.get_data_from_hdf")
    def test_get_subhaloids_from_db(self, mock_get_data):
        """Test get_subhaloids_from_db."""
        mock_catalogue = pd.DataFrame({
            "SubhaloID": [1, 2, 3, 4, 5],
            "P_Late": [0.7, 0.2, 0.3, 0.8, 0.1],
            "P_S0": [0.1, 0.8, 0.1, 0.1, 0.7],
            "P_Sab": [0.1, 0.0, 0.8, 0.1, 0.1]
        })
        mock_get_data.return_value = mock_catalogue

        result = get_subhaloids_from_db(3, "test_path", 99)
        self.assertEqual(len(result), 3)

    @patch("app.astro_utils.pd.read_csv")
    def test_read_line_emission_csv(self, mock_read_csv):
        """Test read_line_emission_csv."""
        mock_read_csv.return_value = pd.DataFrame({"Line": ["CO", "H2O"], "freq(GHz)": [115.27, 183.31]})
        result = read_line_emission_csv("test.csv")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(result.columns.tolist(), ["Line", "freq(GHz)"])

    @patch("app.astro_utils.read_line_emission_csv")
    def test_get_line_info(self, mock_read_csv):
        """Test get_line_info."""
        mock_read_csv.return_value = pd.DataFrame({
            "Line": ["CO", "H2O"],
            "freq(GHz)": [115.27, 183.31]
        })
        result = get_line_info("test_path")
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    @patch("app.astro_utils.read_line_emission_csv")
    def test_compute_rest_frequency_from_redshift(self, mock_read_csv):
        """Test compute_rest_frequency_from_redshift."""
        mock_read_csv.return_value = pd.DataFrame({
            "Line": ["CO", "H2O"],
            "freq(GHz)": [115.27, 183.31]
        })
        result = compute_rest_frequency_from_redshift("test_path", 115.0, 0.1)
        self.assertAlmostEqual(result, 115.27, places=2)

    @patch("builtins.open", new_callable=mock_open)
    def test_write_sim_parameters(self, mock_file):
        """Test write_sim_parameters."""
        write_sim_parameters("test.txt", param1=1, param2=2)
        mock_file.assert_called_once_with("test.txt", "w")
        mock_file().write.assert_any_call("Simulation Parameters:\n")
        mock_file().write.assert_any_call("Param1: 1\n")
        mock_file().write.assert_any_call("Param2: 2\n")

if __name__ == "__main__":
    unittest.main()
