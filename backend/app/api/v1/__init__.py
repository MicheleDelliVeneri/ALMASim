"""API v1 package."""

from fastapi import APIRouter

from app.api.v1.routers import download, imaging, metadata, simulation, visualizer

api_router = APIRouter()
api_router.include_router(
    simulation.router, prefix="/simulations", tags=["simulations"]
)
api_router.include_router(metadata.router, prefix="/metadata", tags=["metadata"])
api_router.include_router(visualizer.router, prefix="/visualizer", tags=["visualizer"])
api_router.include_router(imaging.router, prefix="/imaging", tags=["imaging"])
api_router.include_router(download.router, prefix="/downloads", tags=["downloads"])

__all__ = ["api_router"]
