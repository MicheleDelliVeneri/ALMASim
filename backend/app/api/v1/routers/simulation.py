"""Simulation API endpoints."""
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, WebSocket, WebSocketDisconnect, status

from almasim.services.compute.base import ComputationBackend
from almasim.services.compute.factory import create_backend

from app.core.config import settings
from app.core.dependencies import get_compute_backend
from app.schemas.simulation import (
    SimulationParamsCreate,
    SimulationResponse,
    SimulationStatus as SimulationStatusSchema,
)
from app.services.simulation_service import SimulationService
from app.services.status_store import status_store

router = APIRouter()


@router.post("/", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    params: SimulationParamsCreate,
    background_tasks: BackgroundTasks,
    default_backend: ComputationBackend = Depends(get_compute_backend),
) -> SimulationResponse:
    """Create and start a new simulation."""
    try:
        simulation_id = str(uuid.uuid4())
        
        # Use backend from params if provided, otherwise use default
        if params.compute_backend:
            backend_config = params.compute_backend_config or {}
            compute_backend = create_backend(params.compute_backend, **backend_config)
        else:
            compute_backend = default_backend
        
        # Use settings values for paths (frontend doesn't need to know Docker paths)
        service = SimulationService(
            main_dir=settings.MAIN_DIR,
            output_dir=settings.OUTPUT_DIR,
            tng_dir=settings.TNG_DIR,
            galaxy_zoo_dir=settings.GALAXY_ZOO_DIR,
            hubble_dir=settings.HUBBLE_DIR,
            compute_backend=compute_backend,
        )

        # Create status entry
        status_store.create(simulation_id)
        status_store.update(simulation_id, status="queued", message="Simulation queued successfully")
        
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


@router.get("/{simulation_id}/status", response_model=SimulationStatusSchema)
async def get_simulation_status(simulation_id: str) -> SimulationStatusSchema:
    """Get simulation status."""
    sim_status = status_store.get(simulation_id)
    if not sim_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found",
        )
    
    return SimulationStatusSchema(
        simulation_id=sim_status.simulation_id,
        status=sim_status.status,
        progress=sim_status.progress,
        message=sim_status.message,
        current_step=sim_status.current_step,
        logs=sim_status.logs[-100:],  # Return last 100 log entries
        error=sim_status.error,
    )


@router.websocket("/{simulation_id}/ws")
async def websocket_status(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation status updates."""
    await websocket.accept()
    
    try:
        while True:
            sim_status = status_store.get(simulation_id)
            if not sim_status:
                await websocket.send_json({
                    "error": f"Simulation {simulation_id} not found"
                })
                break
            
            # Send current status
            await websocket.send_json({
                "simulation_id": sim_status.simulation_id,
                "status": sim_status.status,
                "progress": sim_status.progress,
                "current_step": sim_status.current_step,
                "message": sim_status.message,
                "logs": sim_status.logs[-50:],  # Send last 50 logs
                "error": sim_status.error,
            })
            
            # If simulation is completed or failed, close connection
            if sim_status.status in ("completed", "failed"):
                break
            
            # Wait a bit before next update
            await asyncio.sleep(0.5)  # Update every 500ms
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


