"""Channel-by-channel interferometric imaging of a model cube (MADIV-28 T5.11).

:class:`ChannelImagingConfig` holds what every channel shares, so a channel
task is the config, the channel's image, its wavelength and its noise; the
22 positional arguments of :func:`imaging.image_channel` are filled from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from astropy.io import fits

from .imaging import image_channel
from .utils import sampling_to_uv_mask

# The per-channel outputs of image_channel, in order (the ninth is the raw
# visibility rows, kept as a list).
CHANNEL_OUTPUTS = ("model", "dirty", "model_vis", "dirty_vis", "u", "v", "beam", "totsampling")


def picklable_header(header: Any) -> dict:
    """A FITS header as a plain dict of basic values (Headers hold thread locks)."""
    try:
        if isinstance(header, (fits.Header, dict)):
            return {
                str(k): (v if isinstance(v, (int, float, bool, type(None))) else str(v))
                for k, v in header.items()
            }
    except Exception:
        pass
    return {}


@dataclass(frozen=True)
class ChannelImagingConfig:
    """Everything :func:`image_channel` needs that is the same for every channel."""

    Npix: int
    Nant: int
    Hcov: list[float]
    nH: int
    antPos: list
    robfac: float
    trlat: list[float]
    trdec: list[float]
    Diameters: list[float]
    imsize: float
    Xmax: float
    lfac: float
    distmat: np.ndarray
    Nphf: int
    Np4: int
    zooming: int
    header: dict
    robust: float
    scan_time_s: float
    # Seeds each channel's noise (MADIV-28 T5.1); None draws fresh entropy.
    noise_entropy: tuple[int, ...] | None = None

    def channel_rng(self, channel: int) -> np.random.Generator:
        """The generator of ``channel``: the same whichever worker images it."""
        if self.noise_entropy is None:
            return np.random.default_rng()
        return np.random.default_rng((*self.noise_entropy, int(channel)))


def image_configured_channel(
    config: ChannelImagingConfig, image: Any, wavelength: Any, noise: float, channel: int = 0
) -> tuple:
    """:func:`image_channel` of one channel with the shared ``config``."""
    return image_channel(
        image,
        wavelength,
        config.Npix,
        config.Nant,
        config.Hcov,
        config.nH,
        noise,
        config.antPos,
        config.robfac,
        config.trlat,
        config.trdec,
        config.Diameters,
        config.imsize,
        config.Xmax,
        config.lfac,
        config.distmat,
        config.Nphf,
        config.Np4,
        config.zooming,
        config.header,
        config.robust,
        config.scan_time_s,
        rng=config.channel_rng(channel),
    )


def channel_noise(noise: Any, channel: int) -> float:
    """The noise of ``channel``: one value per channel, or one for all."""
    if isinstance(noise, (list, tuple, np.ndarray)):
        return float(noise[channel])
    return float(noise)


def image_channels(
    backend: Any,
    config: ChannelImagingConfig,
    skymodel: np.ndarray,
    wavelengths: Sequence[Any],
    noise: Any,
    track_progress: Callable[[list], None],
) -> list:
    """Image every channel of ``skymodel`` on ``backend``; the per-channel results in order."""
    scattered = [backend.scatter(skymodel[i]) for i in range(skymodel.shape[0])]
    delayed = backend.delayed(image_configured_channel)
    tasks = [
        delayed(config, scattered[i], wavelengths[i], channel_noise(noise, i), i)
        for i in range(skymodel.shape[0])
    ]
    futures = [backend.compute(task, sync=False) for task in tasks]
    track_progress(futures)
    return backend.gather(futures)


def stack_channels(results: list, *, start: int, size: int) -> tuple[dict, list]:
    """The channel results as cubes cropped to the science field, and the raw visibility rows.

    The model sits in the central ``size`` box of the zero-padded FFT grid in
    both domains (the source is centred in the image, the sampled
    visibilities are DC-centred), so the ring around it is the guard band.
    """
    stop = start + size
    cubes = {
        name: np.stack([res[index] for res in results], axis=0)
        for index, name in enumerate(CHANNEL_OUTPUTS)
    }
    for name in ("model", "dirty", "beam", "model_vis", "dirty_vis", "totsampling"):
        cubes[name] = cubes[name][:, start:stop, start:stop]
    cubes["uv_mask"] = sampling_to_uv_mask(cubes["totsampling"])
    return cubes, [res[8] for res in results]
