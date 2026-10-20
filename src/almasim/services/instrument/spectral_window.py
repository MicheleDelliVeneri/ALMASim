"""One channel grid for the cube, the uv sampling and the exported MS (MADIV-28 T5.4).

Maintainer decision (2026-10-20): channel ``i`` of ``N`` is centred on
``f0 + (i - (N - 1) / 2) * delta``, in increasing frequency, everywhere. The
simulator used to sample the uv plane on a split of the band in wavelength
(shortest first, so channel 0 was observed at the band's highest frequency
while the cube's channel 0 is its lowest) and to export CHAN_FREQ edge to
edge with step BW / (N - 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralWindow:
    """``n_channels`` channels of ``channel_width_hz`` centred on ``centre_hz``."""

    centre_hz: float
    channel_width_hz: float
    n_channels: int

    @classmethod
    def centred(cls, centre_hz: float, channel_width_hz: float, n_channels: int) -> SpectralWindow:
        return cls(float(centre_hz), float(channel_width_hz), int(n_channels))

    @classmethod
    def from_band(cls, centre_hz: float, bandwidth_hz: float, n_channels: int) -> SpectralWindow:
        """The band split into ``n_channels`` equal channels."""
        return cls.centred(centre_hz, float(bandwidth_hz) / int(n_channels), n_channels)

    def frequencies_hz(self) -> np.ndarray:
        """Each channel's centre frequency, increasing."""
        offsets = (
            np.arange(self.n_channels, dtype=float) - (float(self.n_channels) - 1.0) / 2.0
        ) * (self.channel_width_hz)
        return self.centre_hz + offsets

    def wavelength_edges_m(self, speed_of_light: float) -> np.ndarray:
        """``[[shorter, longer], ...]``: each channel's wavelength range, channel by channel."""
        frequencies = self.frequencies_hz()
        half = abs(self.channel_width_hz) / 2.0
        return np.stack(
            [speed_of_light / (frequencies + half), speed_of_light / (frequencies - half)], axis=1
        )
