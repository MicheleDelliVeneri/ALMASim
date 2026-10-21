"""Sky models whose morphology is a real galaxy image (Galaxy Zoo, Hubble Top-100)."""

import os
from typing import Any

import numpy as np

from .base import SkyModel
from .utils import interpolate_array


class ImageTemplateSkyModel(SkyModel):
    """A source shaped like a randomly chosen image from ``data_path``.

    The image (its first two colour bands averaged) is rescaled to the cube's
    grid and normalised to unit sum; each channel holds the spectrum's flux
    spread over it. Subclasses say how to list and read the images.
    """

    def __init__(self, *, data_path: str, client: Any | None = None, **inputs: Any):
        super().__init__(client=client, **inputs)
        self.data_path = data_path

    def _image_files(self) -> np.ndarray:
        # Sorted: the listing order depends on the filesystem, and the choice
        # below must be the same for the same seed.
        return np.array(sorted(os.listdir(self.data_path)))

    def _read_image(self, path: str) -> np.ndarray:
        raise NotImplementedError

    def insert(self) -> Any:
        """Insert the image-shaped source into the datacube."""
        imfile = os.path.join(self.data_path, self.rng.choice(self._image_files()))
        img = self._read_image(imfile).astype(np.float32)
        d3 = min(2, np.shape(img)[2])
        avimg = np.average(img[:, :, :d3], axis=2)
        avimg -= np.min(avimg)
        avimg *= 1 / np.max(avimg)
        avimg = interpolate_array(avimg, self.n_px)
        avimg /= np.sum(avimg)
        return self._add_template(avimg)
