"""Hubble sky model implementation."""

import os

import numpy as np
from skimage import io

from .image_template import ImageTemplateSkyModel


class HubbleSkyModel(ImageTemplateSkyModel):
    """Hubble Top-100 morphology sky model (``data_path``: the top100 directory)."""

    def _image_files(self) -> np.ndarray:
        return np.array([file for file in os.listdir(self.data_path) if not file.startswith(".")])

    def _read_image(self, path: str) -> np.ndarray:
        return io.imread(path)
