"""Sky model classes for ALMA simulations."""

from martini import DataCube

from .base import SkyModel
from .datasets import (
    RemoteMachine,
    download_galaxy_zoo,
    download_hubble_top100,
    download_tng_structure,
)
from .diffuse import DiffuseSkyModel
from .extended import ExtendedSkyModel
from .galaxy_zoo import GalaxyZooSkyModel
from .gaussian import GaussianSkyModel
from .hubble import HubbleSkyModel
from .molecular import MolecularCloudSkyModel
from .pointlike import PointlikeSkyModel
from .serendipitous import insert_serendipitous
from .utils import (
    gaussian,
    get_datacube_header,
    interpolate_array,
    track_progress,
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
