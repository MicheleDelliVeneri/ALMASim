"""Simulation API endpoints."""
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from dask.distributed import Client

from app.core.config import settings
from app.core.dependencies import get_dask_client
from app.schemas.simulation import (
    SimulationParamsCreate,
    SimulationResponse,
    SimulationStatus,
)
from app.services.simulation_service import SimulationService

router = APIRouter()


@router.post("/", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    params: SimulationParamsCreate,
    background_tasks: BackgroundTasks,
    dask_client: Client = Depends(get_dask_client),
) -> SimulationResponse:
    """Create and start a new simulation."""
    try:
        simulation_id = str(uuid.uuid4())
        service = SimulationService(
            main_dir=Path(params.main_dir),
            output_dir=Path(params.output_dir),
            tng_dir=Path(params.tng_dir),
            galaxy_zoo_dir=Path(params.galaxy_zoo_dir),
            hubble_dir=Path(params.hubble_dir),
            dask_client=dask_client,
        )

        # Start simulation in background
        background_tasks.add_task(
            service.run_simulation,
            simulation_id=simulation_id,
            params=params,
        )

        return SimulationResponse(
            simulation_id=simulation_id,
            status="queued",
            message="Simulation queued successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create simulation: {str(e)}",
        )


@router.get("/{simulation_id}/status", response_model=SimulationStatus)
async def get_simulation_status(simulation_id: str) -> SimulationStatus:
    """Get simulation status."""
    # In a real implementation, you would check the status from a database or cache
    # For now, return a placeholder
    return SimulationStatus(
        simulation_id=simulation_id,
        status="running",
        progress=50.0,
        message="Simulation in progress",
    )


