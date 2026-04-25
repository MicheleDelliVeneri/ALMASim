"""Computation backend abstraction layer.

This module provides a backend-agnostic interface for parallel computation,
supporting local execution, Dask, Slurm, and Kubernetes backends.
"""

from .base import ComputationBackend
from .dask_backend import DaskBackend
from .factory import create_backend
from .kubernetes import KubernetesBackend
from .local import LocalBackend
from .slurm import SlurmBackend
from .sync import SyncBackend

__all__ = [
    "ComputationBackend",
    "SyncBackend",
    "LocalBackend",
    "DaskBackend",
    "SlurmBackend",
    "KubernetesBackend",
    "create_backend",
]
