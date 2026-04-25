"""FastAPI dependencies."""

from almasim.services.compute.factory import create_backend
from almasim.services.compute.base import ComputationBackend

from app.core.config import settings


def get_compute_backend() -> ComputationBackend:
    """Get or create computation backend."""
    # Determine backend type from settings
    backend_type = settings.COMPUTE_BACKEND.lower()

    # Build backend configuration
    backend_config = settings.COMPUTE_BACKEND_CONFIG.copy()

    # Backward compatibility: if using legacy DASK_SCHEDULER, convert to backend config
    # Only apply if backend_type is explicitly "dask"
    if backend_type == "dask" and not backend_config:
        if settings.DASK_SCHEDULER and settings.DASK_SCHEDULER != "threads":
            backend_config["scheduler"] = settings.DASK_SCHEDULER
        if settings.DASK_N_WORKERS:
            backend_config["n_workers"] = settings.DASK_N_WORKERS

    # Ensure we default to "local" if backend_type is empty or invalid
    if not backend_type or backend_type not in ("local", "dask", "slurm", "kubernetes"):
        backend_type = "local"
        backend_config = {}

    # Create backend
    try:
        backend = create_backend(backend_type, **backend_config)
        return backend
    except Exception:
        # Fallback to local backend on any error
        return create_backend("local")
