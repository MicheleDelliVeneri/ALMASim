"""FastAPI dependencies."""

import logging
import threading
from typing import Optional

from almasim.services.compute.base import ComputationBackend
from almasim.services.compute.factory import create_backend
from app.core.config import settings

logger = logging.getLogger(__name__)

_default_backend: Optional[ComputationBackend] = None
_default_lock = threading.Lock()


def _create_default_backend() -> ComputationBackend:
    """The backend the settings ask for, or the local one when that cannot start."""
    backend_type = settings.COMPUTE_BACKEND.lower()
    backend_config = settings.COMPUTE_BACKEND_CONFIG.copy()

    # Backward compatibility: if using legacy DASK_SCHEDULER, convert to backend config
    # Only apply if backend_type is explicitly "dask"
    if backend_type == "dask" and not backend_config:
        if settings.DASK_SCHEDULER and settings.DASK_SCHEDULER != "threads":
            backend_config["scheduler"] = settings.DASK_SCHEDULER
        if settings.DASK_N_WORKERS:
            backend_config["n_workers"] = settings.DASK_N_WORKERS

    if not backend_type or backend_type not in ("local", "dask", "slurm", "kubernetes"):
        if backend_type:
            logger.warning("Unknown COMPUTE_BACKEND %r; using the local backend", backend_type)
        return create_backend("local")

    try:
        return create_backend(backend_type, **backend_config)
    except Exception as exc:
        logger.warning(
            "Could not start the %s compute backend (%s: %s); using the local backend",
            backend_type,
            type(exc).__name__,
            exc,
        )
        return create_backend("local")


def get_compute_backend() -> ComputationBackend:
    """The application's default backend, created once (MADIV-28 T6.5).

    It used to be created on every request and never closed, leaking a
    process pool or a Dask client per simulation.
    """
    global _default_backend
    with _default_lock:
        if _default_backend is None:
            _default_backend = _create_default_backend()
        return _default_backend


def close_compute_backend() -> None:
    """Close the default backend (application shutdown)."""
    global _default_backend
    with _default_lock:
        backend, _default_backend = _default_backend, None
    if backend is not None:
        try:
            backend.close()
        except Exception as exc:
            logger.warning("Closing the compute backend failed: %s", exc)
