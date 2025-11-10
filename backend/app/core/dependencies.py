"""FastAPI dependencies."""
from dask.distributed import Client
from fastapi import Depends

from app.core.config import settings


def get_dask_client() -> Client:
    """Get or create Dask client."""
    # In a real implementation, you might want to use a singleton pattern
    # or connection pool for the Dask client
    try:
        if settings.DASK_SCHEDULER and settings.DASK_SCHEDULER != "threads":
            client = Client(settings.DASK_SCHEDULER, n_workers=settings.DASK_N_WORKERS)
            return client
        else:
            # Use threads scheduler
            return Client(threads=True)
    except Exception:
        # Fallback to threads scheduler
        return Client(threads=True)

