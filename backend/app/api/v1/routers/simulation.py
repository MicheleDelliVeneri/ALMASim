"""Simulation API endpoints."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from almasim.services.compute.base import ComputationBackend
from almasim.services.compute.factory import create_backend
from app.core.config import settings
from app.core.dependencies import get_compute_backend
from app.schemas.simulation import (
    SimulationListResponse,
    SimulationParamsCreate,
    SimulationResponse,
)
from app.schemas.simulation import (
    SimulationStatus as SimulationStatusSchema,
)
from app.services.simulation_service import SimulationService
from app.services.status_store import status_store
from database.config import get_db
from database.service import DatabaseService

router = APIRouter()


class DaskTestRequest(BaseModel):
    scheduler: str


class DaskTestResponse(BaseModel):
    ok: bool
    scheduler: str
    dashboard_port: int
    workers: int
    total_threads: int
    total_memory_gb: float
    error: str | None = None


@router.get("/", response_model=SimulationListResponse)
async def list_simulations(db: Session = Depends(get_db)) -> SimulationListResponse:
    """List all simulations."""
    from app.schemas.simulation import SimulationSummary

    summaries_by_id: dict[str, SimulationSummary] = {}

    try:
        for sim in DatabaseService(db).list_simulations(limit=500):
            simulation_id = str(sim.simulation_id)
            summaries_by_id[simulation_id] = SimulationSummary(
                simulation_id=simulation_id,
                status=sim.status,
                progress=float(sim.progress or 0.0),
                message=sim.message or "",
                created_at=sim.created_at,
                updated_at=sim.updated_at,
                error=sim.error,
                output_dir=sim.output_path,
            )
    except Exception:
        pass

    for sim in status_store.list_all():
        summaries_by_id[sim.simulation_id] = SimulationSummary(
            simulation_id=sim.simulation_id,
            status=sim.status,
            progress=sim.progress,
            message=sim.message,
            created_at=sim.created_at,
            updated_at=sim.updated_at,
            error=sim.error,
            output_dir=sim.output_dir,
        )

    summaries = list(summaries_by_id.values())

    # Sort by updated_at descending (most recent first)
    summaries.sort(key=lambda x: x.updated_at, reverse=True)

    return SimulationListResponse(simulations=summaries, total=len(summaries))


@router.post(
    "/", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED
)
async def create_simulation(
    params: SimulationParamsCreate,
    background_tasks: BackgroundTasks,
    default_backend: ComputationBackend = Depends(get_compute_backend),
    db: Session = Depends(get_db),
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

        try:
            DatabaseService(db).create_simulation_job(
                simulation_id=uuid.UUID(simulation_id),
                source_name=params.source_name,
                idx=params.idx,
                project_name=params.project_name,
                source_type=params.source_type,
                n_pix=params.n_pix,
                n_channels=params.n_channels,
                rest_frequency=params.rest_frequency,
                redshift=params.redshift,
                lum_infrared=params.lum_infrared,
                snr=params.snr,
                n_lines=params.n_lines,
                line_names=params.line_names,
                save_mode=params.save_mode,
                inject_serendipitous=params.inject_serendipitous,
                ncpu=params.ncpu,
                remote=False,
                status="queued",
                progress=0.0,
                current_step="Initializing",
                message="Simulation queued successfully",
                output_path=params.output_dir or str(settings.OUTPUT_DIR),
            )
        except Exception:
            pass

        # Create status entry
        status_store.create(simulation_id)
        status_store.update(
            simulation_id, status="queued", message="Simulation queued successfully"
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


@router.get("/{simulation_id}/status", response_model=SimulationStatusSchema)
async def get_simulation_status(
    simulation_id: str,
    db: Session = Depends(get_db),
) -> SimulationStatusSchema:
    """Get simulation status."""
    sim_status = status_store.get(simulation_id)
    if sim_status:
        return SimulationStatusSchema(
            simulation_id=sim_status.simulation_id,
            status=sim_status.status,
            progress=sim_status.progress,
            message=sim_status.message,
            current_step=sim_status.current_step,
            logs=sim_status.logs[-100:],
            error=sim_status.error,
        )

    try:
        db_service = DatabaseService(db)
        job = db_service.get_simulation_by_id(uuid.UUID(simulation_id))
        if job:
            logs = [
                f"[{log.timestamp.isoformat()}] {log.message}"
                for log in reversed(db_service.get_simulation_logs(job.simulation_id, limit=100))
            ]
            return SimulationStatusSchema(
                simulation_id=str(job.simulation_id),
                status=job.status,
                progress=float(job.progress or 0.0),
                message=job.message,
                current_step=job.current_step,
                logs=logs,
                error=job.error,
            )
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Simulation {simulation_id} not found",
    )


@router.websocket("/{simulation_id}/ws")
async def websocket_status(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation status updates."""
    await websocket.accept()

    try:
        while True:
            sim_status = status_store.get(simulation_id)
            if not sim_status:
                await websocket.send_json(
                    {"error": f"Simulation {simulation_id} not found"}
                )
                break

            # Send current status
            await websocket.send_json(
                {
                    "simulation_id": sim_status.simulation_id,
                    "status": sim_status.status,
                    "progress": sim_status.progress,
                    "current_step": sim_status.current_step,
                    "message": sim_status.message,
                    "logs": sim_status.logs[-50:],  # Send last 50 logs
                    "error": sim_status.error,
                }
            )

            # If simulation is completed or failed, close connection
            if sim_status.status in ("completed", "failed"):
                break

            # Wait a bit before next update
            await asyncio.sleep(0.5)  # Update every 500ms

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})


@router.post("/test-dask", response_model=DaskTestResponse)
async def test_dask_connection(body: DaskTestRequest):
    """Test connectivity to a Dask scheduler and return cluster info."""
    from dask.distributed import Client

    try:
        client = Client(body.scheduler, timeout=10)
        info = client.scheduler_info()
        workers = info.get("workers", {})
        total_threads = sum(w.get("nthreads", 0) for w in workers.values())
        total_memory = sum(w.get("memory_limit", 0) for w in workers.values())
        dashboard_port = info.get("services", {}).get("dashboard", 8787)

        result = DaskTestResponse(
            ok=True,
            scheduler=info["address"],
            dashboard_port=dashboard_port,
            workers=len(workers),
            total_threads=total_threads,
            total_memory_gb=round(total_memory / 1e9, 2),
        )
        client.close()
        return result
    except Exception as e:
        return DaskTestResponse(
            ok=False,
            scheduler=body.scheduler,
            dashboard_port=0,
            workers=0,
            total_threads=0,
            total_memory_gb=0,
            error=str(e),
        )
