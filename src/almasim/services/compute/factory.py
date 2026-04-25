"""Factory for creating computation backends."""

from typing import Any, Dict

from .base import ComputationBackend
from .dask_backend import DaskBackend
from .kubernetes import KubernetesBackend
from .local import LocalBackend
from .slurm import SlurmBackend
from .sync import SyncBackend


def create_backend(
    backend_type: str = "local",
    **kwargs: Dict[str, Any],
) -> ComputationBackend:
    """Create a computation backend.

    Parameters
    ----------
    backend_type : str
        Backend type: "sync", "local", "dask", "slurm", or "kubernetes"
    **kwargs
        Backend-specific configuration parameters

    Returns
    -------
    ComputationBackend
        Configured computation backend

    Examples
    --------
    >>> # Synchronous backend
    >>> backend = create_backend("sync")

    >>> # Local backend
    >>> backend = create_backend("local", n_workers=4)

    >>> # Dask backend
    >>> backend = create_backend("dask", scheduler="tcp://localhost:8786")

    >>> # Slurm backend
    >>> backend = create_backend("slurm", queue="normal", n_workers=8)

    >>> # Kubernetes backend
    >>> backend = create_backend("kubernetes", n_workers=4, namespace="default")
    """
    backend_type = backend_type.lower()

    if backend_type == "sync":
        return SyncBackend()
    elif backend_type == "local":
        return LocalBackend(**kwargs)
    elif backend_type == "dask":
        return DaskBackend(**kwargs)
    elif backend_type == "slurm":
        return SlurmBackend(**kwargs)
    elif backend_type == "kubernetes":
        return KubernetesBackend(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported types: sync, local, dask, slurm, kubernetes"
        )
