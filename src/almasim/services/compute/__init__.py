"""Computation backend abstraction layer.

This module provides a backend-agnostic interface for parallel computation,
supporting local execution, Dask, Slurm, and Kubernetes backends.
"""

from .base import ComputationBackend
from .sync import SyncBackend
from .local import LocalBackend
from .dask_backend import DaskBackend
from .slurm import SlurmBackend
from .kubernetes import KubernetesBackend
from .factory import create_backend

__all__ = [
    "ComputationBackend",
    "SyncBackend",
    "LocalBackend",
    "DaskBackend",
    "SlurmBackend",
    "KubernetesBackend",
    "create_backend",
]
