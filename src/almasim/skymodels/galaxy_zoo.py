"""Galaxy Zoo sky model implementation."""

import matplotlib.image as plimg
import numpy as np

from .image_template import ImageTemplateSkyModel


class GalaxyZooSkyModel(ImageTemplateSkyModel):
    """Galaxy Zoo 2 morphology sky model (``data_path``: the images directory)."""

    def _read_image(self, path: str) -> np.ndarray:
        return plimg.imread(path)
