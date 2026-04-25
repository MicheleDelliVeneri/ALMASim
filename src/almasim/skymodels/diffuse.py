"""Diffuse sky model implementation."""

import random
import numpy as np
import astropy.units as U
from typing import Any
from dask import delayed
import nifty8 as ift

from .base import SkyModel
from .utils import track_progress, gaussian


def diffuse_signal(n_px: int) -> np.ndarray:
    """Generate diffuse signal using NIFTY."""
    ift.random.push_sseq(random.randint(1, 1000))
    space = ift.RGSpace((2 * n_px, 2 * n_px))
    args = {
        "offset_mean": 24,
        "offset_std": (1, 0.1),
        "fluctuations": (5.0, 1.0),
        "loglogavgslope": (-3.5, 0.5),
        "flexibility": (1.2, 0.4),
        "asperity": (0.2, 0.2),
    }

    cf = ift.SimpleCorrelatedField(space, **args)
    exp_cf = ift.exp(cf)
    random_pos = ift.from_random(exp_cf.domain)
    sample = np.log(exp_cf(random_pos))
    data = sample.val[0:n_px, 0:n_px]
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data


@delayed
def diffuse_image(diffuse_signal: np.ndarray, amp: float) -> np.ndarray:
    """Scale diffuse signal by amplitude."""
    return diffuse_signal * amp


class DiffuseSkyModel(SkyModel):
    """Diffuse sky model using NIFTY."""

    def insert(self) -> Any:
        """Insert diffuse source into datacube."""
        ts = diffuse_signal(self.n_px)
        ts = np.nan_to_num(ts)
        ts -= np.min(ts)
        ts *= 1 / np.max(ts)
        z_idxs = np.arange(0, self.n_chan)
        gs = np.zeros(self.n_chan)
        skymodel = []
        for i in range(len(self.line_fluxes)):
            gs += gaussian(z_idxs, self.line_fluxes[i], self.pos_z[i], self.fwhm_z[i])
        for z in range(0, self.n_chan):
            delayed_result = diffuse_image(ts, gs[z] + self.continuum[z])
            skymodel.append(delayed_result)
        delayed_skymodel = delayed(np.stack)(skymodel, axis=0)
        futures = self.client.compute([delayed_skymodel])
        track_progress(self.update_progress, futures)
        skymodel = self.client.gather(futures)
        self.datacube._array = skymodel * U.Jy * U.pix**-2
        return self.datacube
