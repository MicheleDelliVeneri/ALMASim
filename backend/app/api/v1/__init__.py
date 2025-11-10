"""API v1 package."""
from fastapi import APIRouter

from app.api.v1.routers import metadata, simulation

api_router = APIRouter()
api_router.include_router(simulation.router, prefix="/simulations", tags=["simulations"])
api_router.include_router(metadata.router, prefix="/metadata", tags=["metadata"])

__all__ = ["api_router"]

