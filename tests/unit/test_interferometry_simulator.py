"""The pieces Interferometer is built from (MADIV-28 T5.11)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from almasim.services.compute.sync import SyncBackend
from almasim.services.interferometry import simulator
from almasim.services.interferometry.frequency import channel_wavelength_edges
from almasim.services.interferometry.observing import alma_location, clip_to_horizon

pytestmark = pytest.mark.unit


def test_picklable_header_keeps_basic_values_and_strings_the_rest():
    header = fits.Header()
    header["NAXIS"] = 3
    header["OBJECT"] = "NGC 1068"
    header["BMAJ"] = 1.5
    assert simulator.picklable_header(header) == {"NAXIS": 3, "OBJECT": "NGC 1068", "BMAJ": 1.5}
    assert simulator.picklable_header({"A": [1, 2], "B": None}) == {"A": "[1, 2]", "B": None}
    assert simulator.picklable_header(object()) == {}


def test_channel_noise_per_channel_or_shared():
    assert simulator.channel_noise([0.1, 0.2], 1) == 0.2
    assert simulator.channel_noise(np.array([0.3, 0.4]), 0) == pytest.approx(0.3)
    assert simulator.channel_noise(0.5, 7) == 0.5


def test_image_channels_runs_every_channel_with_the_shared_config(monkeypatch):
    seen = []

    def fake_image_channel(image, wavelength, *args):
        seen.append((float(image.sum()), wavelength, args[4]))  # args[4] is the noise
        return tuple(np.full((4, 4), float(image.sum())) for _ in range(8)) + ([image.sum()],)

    monkeypatch.setattr(simulator, "image_channel", fake_image_channel)
    config = simulator.ChannelImagingConfig(
        *[MagicMock()] * 16, header={}, robust=0.0, scan_time_s=6.0
    )
    progress = []
    results = simulator.image_channels(
        SyncBackend(),
        config,
        np.ones((3, 2, 2)),
        ["w0", "w1", "w2"],
        [0.1, 0.2, 0.3],
        progress.append,
    )
    assert [s[1:] for s in seen] == [("w0", 0.1), ("w1", 0.2), ("w2", 0.3)]
    assert len(results) == 3 and len(progress) == 1

    cubes, rows = simulator.stack_channels(results, start=1, size=2)
    assert cubes["model"].shape == (3, 2, 2) and cubes["u"].shape == (3, 4, 4)
    assert cubes["uv_mask"].shape == (3, 2, 2) and rows == [[4.0], [4.0], [4.0]]


def test_clip_to_horizon_limits_the_track():
    lat = np.radians(-23.0)
    coverage = clip_to_horizon([-3.0, 3.0], lat, np.radians(40.0))  # low northern source
    assert -3.0 < coverage[0] < 0 < coverage[1] < 3.0
    assert clip_to_horizon([-1.0, 1.0], lat, np.radians(-23.0)) == [-1.0, 1.0]
    assert clip_to_horizon([-3.0, 3.0], lat, np.radians(-89.0)) == [-3.0, 3.0]  # circumpolar


def test_channel_wavelength_edges_split_the_band():
    w_min, w_max, edges = channel_wavelength_edges(100e9, 2e9, 4, 3e8)
    assert w_min == pytest.approx(3e8 / 101e9) and w_max == pytest.approx(3e8 / 99e9)
    assert edges.shape == (4, 2) and edges[0, 0] == w_min and edges[-1, 1] == pytest.approx(w_max)


def test_alma_location_is_the_site():
    assert alma_location().lat.deg == pytest.approx(-23.03, abs=0.01)
