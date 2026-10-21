"""One instrument model for the imaging planner and the simulator (MADIV-28 T2.11)."""

import math

import numpy as np
import pytest

from almasim.services.instrument import array, geometry
from almasim.services.instrument.constants import ARCSEC_PER_RAD, SPEED_OF_LIGHT_M_S

pytestmark = pytest.mark.unit


def test_max_baseline_is_the_longest_antenna_separation():
    positions = np.array([[0.0, 0.0, 0.0], [300.0, 400.0, 0.0], [10.0, 0.0, 0.0]])
    assert geometry.max_baseline_m(positions) == pytest.approx(500.0)


def test_synthesized_beam_is_lambda_over_the_longest_baseline():
    beam = geometry.synthesized_beam_arcsec(100e9, 1000.0)
    assert beam == pytest.approx(SPEED_OF_LIGHT_M_S / 100e9 / 1000.0 * ARCSEC_PER_RAD)
    np.testing.assert_allclose(
        geometry.synthesized_beam_arcsec(np.array([100e9, 200e9]), 1000.0), [beam, beam / 2]
    )


def test_field_of_view_is_the_primary_beam_half_power_width():
    fov_12m = geometry.primary_beam_fov_arcsec(12.0, 100e9)
    assert 55 < fov_12m < 65, "about 58 arcsec at 100 GHz for a 12 m dish"
    assert geometry.primary_beam_fov_arcsec(7.0, 100e9) > fov_12m


def test_image_size_rounds_up_or_to_a_power_of_two():
    assert geometry.cell_arcsec(1.0, 8) == 0.125
    assert geometry.image_size(10.0, 0.3) == math.ceil(10.0 / 0.3)
    assert geometry.image_size(1.0, 0.5) == geometry.MIN_IMAGE_PIXELS
    assert geometry.image_size(10.0, 0.3, min_pixels=40.2) == 41
    assert geometry.image_size(100.0, 1.0, rounding="pow2") == 128


def test_antenna_names_give_array_types_and_dish_sizes():
    assert [array.antenna_array_type(n) for n in ("DA41", "DV02", "CM03", "PM04")] == [
        "12m",
        "12m",
        "7m",
        "TP",
    ]
    assert array.dish_diameter_m("7m") == 7.0
    assert array.dish_diameter_m("TP") == 12.0
    assert array.dish_diameter_m("") == 12.0
