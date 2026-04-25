"""Sky model classes for ALMA simulations."""

from martini import DataCube
from .base import SkyModel
from .pointlike import PointlikeSkyModel
from .gaussian import GaussianSkyModel
from .galaxy_zoo import GalaxyZooSkyModel
from .diffuse import DiffuseSkyModel
from .molecular import MolecularCloudSkyModel
from .hubble import HubbleSkyModel
from .extended import ExtendedSkyModel
from .utils import (
    interpolate_array,
    track_progress,
    gaussian,
    get_datacube_header,
)
from .serendipitous import insert_serendipitous
from .datasets import (
    download_galaxy_zoo,
    download_hubble_top100,
    download_tng_structure,
    RemoteMachine,
)

__all__ = [
    "DataCube",
    "SkyModel",
    "PointlikeSkyModel",
    "GaussianSkyModel",
    "GalaxyZooSkyModel",
    "DiffuseSkyModel",
    "MolecularCloudSkyModel",
    "HubbleSkyModel",
    "ExtendedSkyModel",
    "interpolate_array",
    "track_progress",
    "gaussian",
    "get_datacube_header",
    "insert_serendipitous",
    # Dataset download functions
    "download_galaxy_zoo",
    "download_hubble_top100",
    "download_tng_structure",
    "RemoteMachine",
]
