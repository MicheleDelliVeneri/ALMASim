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
    generate_datacube,
    insert_serendipitous,
    get_iou,
    get_iou_1d,
    get_pos,
    sample_positions
)
from martini import DataCube
from astropy import units as u

class TestSkyModelUtils(unittest.TestCase):

    def test_interpolate_array(self):
        arr = np.array([[1, 2], [3, 4]])
        result = interpolate_array(arr, 4)
        self.assertEqual(result.shape, (4, 4))

    def test_interpolate_array_edge_case(self):
        with self.assertRaises(ValueError):
            interpolate_array(np.zeros((0, 0)), 4)

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

    def test_diffuse_signal_edge_case(self):
        with self.assertRaises(ValueError):
            diffuse_signal(0)
        with self.assertRaises(ValueError):
            diffuse_signal(-1)
        result = diffuse_signal(2)
        self.assertEqual(result.shape, (2, 2))
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

    def test_make_extended_invalid_ellip(self):
        imsize = 10
        with self.assertRaises(ValueError):
            make_extended(imsize, ellip=-0.1)


    def test_delayed_model_insertion(self):
        slice_data = np.ones((10, 10))
        template = np.ones((10, 10))
        line_flux = 2.0
        continuum = 1.0
        result = delayed_model_insertion(template, line_flux, continuum).compute()
        self.assertTrue((result == 3.0).all())

    def test_insert_model_pointlike(self):
        datacube = generate_datacube(10, 10)
        model_type = "pointlike"
        line_fluxes = np.ones(10)
        continuum = np.ones(10)
        pos_x, pos_y = 5, 5

        result = insert_model(
            datacube, model_type, line_fluxes, continuum, pos_x=pos_x, pos_y=pos_y
        )
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array[:, pos_x, pos_y].sum(), 0)

    def test_insert_model_gaussian(self):
        datacube = generate_datacube(10, 10)
        model_type = "gaussian"
        line_fluxes = np.ones(10)
        continuum = np.ones(10)
        pos_x, pos_y = 5, 5
        fwhm_x, fwhm_y = 2, 2

        result = insert_model(
            datacube,
            model_type,
            line_fluxes,
            continuum,
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
        continuum = np.ones(10)

        result = insert_model(datacube, model_type, line_fluxes, continuum)
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array.sum(), 0)

    def test_insert_model_molecular_cloud(self):
        datacube = generate_datacube(10, 10)
        model_type = "molecular_cloud"
        line_fluxes = np.ones(10)
        continuum = np.ones(10)

        result = insert_model(datacube, model_type, line_fluxes, continuum)
        self.assertIsInstance(result, DataCube)
        self.assertGreater(result._array.sum(), 0)

    def test_insert_model_hubble(self):
        datacube = generate_datacube(10, 10)
        model_type = "hubble"
        line_fluxes = np.ones(10)
        continuum = np.ones(10)
        data_path = "test_data"

        with patch("os.listdir", return_value=["test.fits"]), patch(
                "app.skymodel_utils.io.imread", return_value=np.zeros((10, 10, 3))  # Uniform image
        ):
            result = insert_model(
                datacube,
                model_type,
                line_fluxes,
                continuum,
                data_path=data_path,
            )
            self.assertIsInstance(result, DataCube)
            self.assertFalse(np.isnan(result._array).any())  # Ensure no NaN values
            self.assertTrue((result._array > 0).any())  # Ensure at least some values are greater than 0

    def test_insert_model_invalid_type(self):
        datacube = generate_datacube(10, 10)
        model_type = "invalid_type"
        line_fluxes = np.ones(10)
        continuum = np.ones(10)

        with self.assertRaises(ValueError):
            insert_model(datacube, model_type, line_fluxes, continuum)


    def test_generate_datacube(self):
        datacube = generate_datacube(10, 10)
        self.assertEqual(datacube._array.shape, (10, 10, 10))

    def setUp(self):
        self.datacube = DataCube(10, 10, 10)


    def test_insert_model_galaxy_zoo(self):
        line_fluxes = np.ones(10)
        continuum = np.ones(10)
        data_path = "test_data"

        with patch("os.listdir", return_value=["galaxy.fits"]), patch(
            "app.skymodel_utils.io.imread", return_value=np.ones((10, 10, 3))
        ):
            result = insert_model(
                self.datacube,
                "galaxy_zoo",
                line_fluxes,
                continuum,
                data_path=data_path,
            )
            self.assertIsInstance(result, DataCube)
            self.assertFalse(np.isnan(result._array).any())
            self.assertTrue((result._array > 0).any())

    def test_insert_serendipitous(self):
        line_fluxes = [1.0]
        continuum = np.ones(10)
        line_names = ["CO"]
        line_frequencies = [115.271]
        freq_sup = 0.1
        pos_zs = [5]
        fwhm_x, fwhm_y = 2, 2
        fwhm_zs = [5]
        sim_params_path = "params.txt"
        xy_radius, z_radius = 10, 10
        sep_xy, sep_z = 3, 3
        datacube = generate_datacube(10, 10)
        with patch("app.skymodel_utils.insert_model", return_value=self.datacube), patch(
            "builtins.open", mock_open()
        ):
            result = insert_serendipitous(
                datacube,
                continuum,
                line_fluxes,
                line_names,
                line_frequencies,
                pos_zs,
                fwhm_x,
                fwhm_y,
                fwhm_zs,
                10,
                10,
                xy_radius,
                z_radius,
                sep_xy,
                sep_z,
                freq_sup,
                sim_params_path,
            )
            self.assertIsInstance(result, DataCube)

    def test_get_iou(self):
        bb1 = {"x1": 0, "x2": 2, "y1": 0, "y2": 2}
        bb2 = {"x1": 1, "x2": 3, "y1": 1, "y2": 3}
        result = get_iou(bb1, bb2)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 1)

    def test_get_iou_no_overlap(self):
        bb1 = {"x1": 0, "x2": 1, "y1": 0, "y2": 1}
        bb2 = {"x1": 2, "x2": 3, "y1": 2, "y2": 3}
        result = get_iou(bb1, bb2)
        self.assertEqual(result, 0.0)

    def test_get_iou_1d(self):
        bb1 = {"z1": 0, "z2": 2}
        bb2 = {"z1": 1, "z2": 3}
        result = get_iou_1d(bb1, bb2)
        self.assertGreater(result, 0)
        self.assertLessEqual(result, 1)

    def test_get_iou_1d_no_overlap(self):
        bb1 = {"z1": 0, "z2": 1}
        bb2 = {"z1": 2, "z2": 3}
        result = get_iou_1d(bb1, bb2)
        self.assertEqual(result, 0.0)

    def test_get_pos(self):
        x_radius, y_radius, z_radius = 5, 5, 5
        pos = get_pos(x_radius, y_radius, z_radius)
        self.assertTrue(-x_radius <= pos[0] <= x_radius)
        self.assertTrue(-y_radius <= pos[1] <= y_radius)
        self.assertTrue(-z_radius <= pos[2] <= z_radius)

    def test_sample_positions(self):
        pos_x, pos_y, pos_z = 5, 5, 5
        n_components = 3
        xy_radius, z_radius = 10, 10
        sep_xy, sep_z = 3, 3

        result = sample_positions(
            pos_x,
            pos_y,
            pos_z,
            n_components,
            xy_radius,
            z_radius,
            sep_xy,
            sep_z,
        )
        self.assertEqual(len(result), n_components)
        for pos in result:
            self.assertTrue(isinstance(pos, tuple))
            self.assertTrue(len(pos), 3)


if __name__ == "__main__":
    unittest.main()
