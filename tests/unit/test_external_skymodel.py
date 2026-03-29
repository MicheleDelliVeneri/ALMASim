"""Unit tests for external skymodel ingestion and native MS model building."""

import numpy as np
from astropy.io import fits
from astropy.table import Table

from almasim.services.external_skymodel import (
    infer_external_cube_geometry,
    is_external_source_type,
    load_external_sky_model,
)
from almasim.services.products.ms_model import build_measurement_set_model


def test_is_external_source_type():
    assert is_external_source_type("external-fits-image") is True
    assert is_external_source_type("external-fits-cube") is True
    assert is_external_source_type("external-components") is True
    assert is_external_source_type("point") is False


def test_infer_external_cube_geometry_from_fits_cube(tmp_path):
    cube = np.ones((4, 8, 8), dtype=np.float32)
    path = tmp_path / "input_cube.fits"
    fits.PrimaryHDU(data=cube).writeto(path)

    geometry = infer_external_cube_geometry(
        source_type="external-fits-cube",
        skymodel_path=str(path),
    )

    assert geometry == {
        "n_channels": 4,
        "n_pix_y": 8,
        "n_pix_x": 8,
        "n_pix": 8,
    }


def test_load_external_fits_image_aligns_to_observation_shape(tmp_path):
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    path = tmp_path / "input_image.fits"
    fits.PrimaryHDU(data=image).writeto(path)

    payload = load_external_sky_model(
        source_type="external-fits-image",
        skymodel_path=str(path),
        target_npix=8,
        target_nchannels=3,
        alignment_mode="observation",
        header_mode="observation",
        target_header=fits.Header(),
    )

    assert payload.cube.shape == (3, 8, 8)
    assert payload.metadata["input_kind"] == "fits"
    assert payload.metadata["original_shape"] == [1, 4, 4]


def test_load_external_component_table_builds_channel_cube(tmp_path):
    table_path = tmp_path / "components.ecsv"
    table = Table(
        {
            "x_pix": [2, 4],
            "y_pix": [3, 1],
            "flux_jy": [1.5, 2.0],
            "spectral_index": [0.0, 1.0],
            "reference_freq_hz": [100.0, 100.0],
        }
    )
    table.write(table_path, format="ascii.ecsv")

    payload = load_external_sky_model(
        source_type="external-components",
        component_table_path=str(table_path),
        target_npix=8,
        target_nchannels=4,
        channel_frequencies_hz=np.array([100.0, 110.0, 120.0, 130.0]),
        target_header=fits.Header(),
    )

    assert payload.cube.shape == (4, 8, 8)
    assert np.all(payload.cube[:, 3, 2] > 0.0)
    assert payload.cube[0, 1, 4] < payload.cube[-1, 1, 4]


def test_build_measurement_set_model_from_visibility_rows():
    vis = {
        "uvw_m": np.array([[0.0, 0.0, 0.0], [10.0, 5.0, 1.0]], dtype=np.float64),
        "antenna1": np.array([0, 0], dtype=np.int32),
        "antenna2": np.array([1, 1], dtype=np.int32),
        "time_mjd_s": np.array([5.0e9, 5.0e9 + 6.0], dtype=np.float64),
        "interval_s": np.array([6.0, 6.0], dtype=np.float64),
        "exposure_s": np.array([6.0, 6.0], dtype=np.float64),
        "data": np.zeros((2, 1, 4), dtype=np.complex64),
        "model_data": np.zeros((2, 1, 4), dtype=np.complex64),
        "flag": np.zeros((2, 1, 4), dtype=bool),
        "weight": np.ones((2, 1), dtype=np.float32),
        "sigma": np.ones((2, 1), dtype=np.float32),
        "channel_freq_hz": np.array([1.0e11, 1.001e11, 1.002e11, 1.003e11], dtype=np.float64),
        "antenna_names": ["DA01", "DA02"],
        "antenna_positions_m": np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float64),
        "field_ra_rad": 1.0,
        "field_dec_rad": 0.5,
        "observation_date": "2020-01-01",
    }

    model = build_measurement_set_model(
        visibility_table=vis,
        project_name="demo_project",
        source_name="demo_source",
    )

    assert model.main_keywords["MS_VERSION"] == 2.0
    assert model.main.nrows == 2
    assert model.main.columns["DATA"].shape == (2, 1, 4)
    assert "ANTENNA" in model.subtables
    assert model.subtables["ANTENNA"].nrows == 2
    assert model.subtables["FIELD"].columns["NAME"] == "demo_source"
