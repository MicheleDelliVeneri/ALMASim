"""Dataset download utilities for sky models."""

from .galaxy_zoo import download_galaxy_zoo
from .hubble import download_hubble_top100
from .tng import download_tng_structure, RemoteMachine

__all__ = [
    "download_galaxy_zoo",
    "download_hubble_top100",
    "download_tng_structure",
    "RemoteMachine",
]
