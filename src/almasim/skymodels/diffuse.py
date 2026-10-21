"""Diffuse sky model implementation."""

from typing import Any, Optional

import nifty8 as ift
import numpy as np

from .base import SkyModel


def diffuse_signal(n_px: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate diffuse signal using NIFTY; ``rng`` seeds it (fresh entropy when None)."""
    rng = rng if rng is not None else np.random.default_rng()
    ift.random.push_sseq(int(rng.integers(1, 1001)))
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


class DiffuseSkyModel(SkyModel):
    """Diffuse sky model using NIFTY."""

    def insert(self) -> Any:
        """Insert diffuse source into datacube."""
        ts = diffuse_signal(self.n_px, self.rng)
        ts = np.nan_to_num(ts)
        ts -= np.min(ts)
        ts *= 1 / np.max(ts)
        return self._add_template(ts)
