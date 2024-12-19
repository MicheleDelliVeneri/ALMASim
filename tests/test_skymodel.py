import unittest
from unittest.mock import patch, mock_open
import numpy as np
from app.skymodel_utils import (
    interpolate_array,
    gaussian,
    diffuse_signal,
    molecular_cloud,
    make_extended,
    delayed_model_insertion,
    insert_model,
    generate_datacube
)
from martini import DataCube
from astropy import units as u

class TestSkyModelUtils(unittest.TestCase):

    def test_interpolate_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = interpolate_array(arr, 4)
        self.assertEqual(result.shape, (4, 4))

    def test_gaussian(self):
        x = np.linspace(-5, 5, 100)
        amp, cen, fwhm = 1.0, 0.0, 2.0
        result = gaussian(x, amp, cen, fwhm)
        self.assertAlmostEqual(np.sum(result), amp, places=5)

    def test_diffuse_signal(self):
        n_px = 10
        result = diffuse_signal(n_px)
        self.assertEqual(result.shape, (10, 10))
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_molecular_cloud(self):
        n_px = 10
        result = molecular_cloud(n_px)
        self.assertEqual(result.shape, (10, 10))
        self.assertFalse(np.isnan(result).any())  # Ensure no NaN values
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_make_extended(self):
        imsize = 10
        result = make_extended(imsize)
        self.assertEqual(result.shape, (10, 10))

    def test_delayed_model_insertion(self):
        slice_data = np.ones((10, 10))
        template = np.ones((10, 10))
        line_flux = 2.0
        continum = 1.0
        result = delayed_model_insertion(slice_data, template, line_flux, continum).compute()
        self.assertTrue((result == 3.0).all())

    def test_insert_model_pointlike(self):
        datacube = generate_datacube(10, 10)
        model_type = "pointlike"
        line_fluxes = np.ones(10)
        continum = np.ones(10)
        pos_x, pos_y = 5, 5

        result = insert_model(
            datacube, model_type, line_fluxes, continum, pos_x=pos_x, pos_y=pos_y
        )
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array[:, pos_x, pos_y].sum(), 0)

    def test_insert_model_gaussian(self):
        datacube = generate_datacube(10, 10)
        model_type = "gaussian"
        line_fluxes = np.ones(10)
        continum = np.ones(10)
        pos_x, pos_y = 5, 5
        fwhm_x, fwhm_y = 2, 2

        result = insert_model(
            datacube,
            model_type,
            line_fluxes,
            continum,
            pos_x=pos_x,
            pos_y=pos_y,
            fwhm_x=fwhm_x,
            fwhm_y=fwhm_y,
        )
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array.sum(), 0)

    def test_insert_model_diffuse(self):
        datacube = generate_datacube(10, 10)
        model_type = "diffuse"
        line_fluxes = np.ones(10)
        continum = np.ones(10)

        result = insert_model(datacube, model_type, line_fluxes, continum)
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array.sum(), 0)

    def test_insert_model_molecular_cloud(self):
        datacube = generate_datacube(10, 10)
        model_type = "molecular_cloud"
        line_fluxes = np.ones(10)
        continum = np.ones(10)

        result = insert_model(datacube, model_type, line_fluxes, continum)
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array.sum(), 0)

    def test_insert_model_hubble(self):
        datacube = generate_datacube(10, 10)
        model_type = "hubble"
        line_fluxes = np.ones(10)
        continum = np.ones(10)
        data_path = "test_data"

        with patch("os.listdir", return_value=["test.fits"]), patch(
                "app.skymodel_utils.io.imread", return_value=np.zeros((10, 10, 3))  # Uniform image
        ):
            result = insert_model(
                datacube,
                model_type,
                line_fluxes,
                continum,
                data_path=data_path,
            )
            self.assertIsInstance(result, DataCube)
            self.assertFalse(np.isnan(result._array).any())  # Ensure no NaN values
            self.assertTrue((result._array > 0).any())  # Ensure at least some values are greater than 0

    def test_generate_datacube(self):
        datacube = generate_datacube(10, 10)
        self.assertEqual(datacube._array.shape, (10, 10, 10))

if __name__ == "__main__":
    unittest.main()
