"""Hour-angle sampling and coverage in seconds and radians (MADIV-28 T1.3)."""

from __future__ import annotations

import math

import astropy.units as U
import pytest
from astropy.coordinates import EarthLocation

from almasim.services.instrument.constants import (
    ALMA_ELEVATION_M,
    ALMA_LATITUDE_DEG,
    ALMA_LONGITUDE_DEG,
)
from almasim.services.interferometry.observing import hour_angle_coverage, hour_angle_samples

pytestmark = pytest.mark.unit

ALMA = EarthLocation.from_geodetic(
    lon=ALMA_LONGITUDE_DEG * U.deg, lat=ALMA_LATITUDE_DEG * U.deg, height=ALMA_ELEVATION_M * U.m
)
SIDEREAL_RATE = 2 * math.pi / 86164.0905  # rad per second of time


@pytest.mark.parametrize(
    ("integration_s", "expected"),
    [
        (450.0, (75, 6.0)),
        (1200.0, (200, 6.0)),
        (1500.0, (186, 8.064)),
        (13970.88, (462, 30.24)),
        (1.0, (1, 6.0)),
    ],
)
def test_one_epoch_per_scan_in_seconds(integration_s, expected):
    assert hour_angle_samples(integration_s) == expected


@pytest.mark.parametrize("ra_deg", [0.0, 36.6, 180.0, 359.9])
def test_a_one_hour_track_spans_about_a_quarter_radian(ra_deg):
    start, end = hour_angle_coverage("2017-03-14", 3600.0, math.radians(ra_deg), ALMA)
    assert end - start == pytest.approx(3600.0 * SIDEREAL_RATE, rel=1e-3)
    assert start == pytest.approx(-end, rel=1e-3), "centred on the middle of the track"
    assert abs(start) < math.pi and abs(end) < math.pi
