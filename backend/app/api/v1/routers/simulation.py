"""Simulation API endpoints."""

import asyncio
import uuid

from database.config import get_db
from database.service import DatabaseService
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

import almasim.services.simulation as sim_service
from almasim.services.compute.base import ComputationBackend
from almasim.services.compute.factory import create_backend
from app.core.config import settings
from app.core.dependencies import get_compute_backend
from app.schemas.simulation import (
    SimulationEstimate,
    SimulationListResponse,
    SimulationParamsCreate,
    SimulationResponse,
)
from app.schemas.simulation import (
    SimulationStatus as SimulationStatusSchema,
)
from app.services.simulation_service import SimulationService
from app.services.status_store import status_store

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


@router.post("/", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
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


@router.post("/estimate", response_model=SimulationEstimate)
async def estimate_simulation(
    params: SimulationParamsCreate,
) -> SimulationEstimate:
    """Estimate simulation cube dimensions and raw storage footprint."""
    try:
        sim_params = sim_service.SimulationParams(
            idx=params.idx,
            source_name=params.source_name,
            member_ouid=params.member_ouid,
            main_dir=str(settings.MAIN_DIR),
            output_dir=params.output_dir or str(settings.OUTPUT_DIR),
            tng_dir=str(settings.TNG_DIR),
            galaxy_zoo_dir=str(settings.GALAXY_ZOO_DIR),
            hubble_dir=str(settings.HUBBLE_DIR),
            project_name=params.project_name,
            ra=params.ra,
            dec=params.dec,
            band=params.band,
            ang_res=params.ang_res,
            vel_res=params.vel_res,
            fov=params.fov,
            obs_date=params.obs_date,
            pwv=params.pwv,
            int_time=params.int_time,
            bandwidth=params.bandwidth,
            freq=params.freq,
            freq_support=params.freq_support,
            cont_sens=params.cont_sens,
            line_sens_10kms=params.line_sens_10kms,
            antenna_array=params.antenna_array,
            n_pix=params.n_pix,
            n_channels=params.n_channels,
            source_type=params.source_type,
            tng_api_key=params.tng_api_key,
            ncpu=params.ncpu,
            rest_frequency=params.rest_frequency,
            redshift=params.redshift,
            lum_infrared=params.lum_infrared,
            snr=params.snr,
            n_lines=params.n_lines,
            line_names=params.line_names,
            save_mode=params.save_mode,
            persist=params.persist,
            ml_dataset_path=params.ml_dataset_path,
            inject_serendipitous=params.inject_serendipitous,
            remote=False,
            observation_configs=params.observation_configs,
            ground_temperature_k=params.ground_temperature_k,
            correlator=params.correlator,
            elevation_deg=params.elevation_deg,
            source_offset_x_arcsec=params.source_offset_x_arcsec,
            source_offset_y_arcsec=params.source_offset_y_arcsec,
            background_mode=params.background_mode,
            background_level=params.background_level,
            background_seed=params.background_seed,
            external_skymodel_path=params.external_skymodel_path,
            external_component_table_path=params.external_component_table_path,
            external_alignment_mode=params.external_alignment_mode,
            external_header_mode=params.external_header_mode,
            external_header_overrides=params.external_header_overrides,
            ms_export=params.ms_export,
            ms_export_dir=params.ms_export_dir,
        )
        return SimulationEstimate(**sim_service.estimate_simulation_footprint(sim_params))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to estimate simulation footprint: {str(e)}",
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


@router.post("/{simulation_id}/cancel", response_model=SimulationResponse)
async def cancel_simulation(
    simulation_id: str,
    db: Session = Depends(get_db),
) -> SimulationResponse:
    """Request cancellation of a queued or running simulation."""
    sim_status = status_store.get(simulation_id)
    if sim_status is not None:
        if sim_status.status in ("completed", "failed", "cancelled"):
            return SimulationResponse(
                simulation_id=simulation_id,
                status=sim_status.status,
                message=f"Simulation already {sim_status.status}",
            )
        status_store.cancel(simulation_id)
        try:
            DatabaseService(db).update_simulation_status(
                uuid.UUID(simulation_id),
                status="cancelled",
                progress=sim_status.progress,
                current_step="Cancelled",
                message="Cancellation requested",
            )
        except Exception:
            pass
        return SimulationResponse(
            simulation_id=simulation_id,
            status="cancelled",
            message="Cancellation requested",
        )

    try:
        job = DatabaseService(db).get_simulation_by_id(uuid.UUID(simulation_id))
    except Exception:
        job = None

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Simulation {simulation_id} not found",
        )

    if job.status in ("completed", "failed", "cancelled"):
        return SimulationResponse(
            simulation_id=simulation_id,
            status=job.status,
            message=f"Simulation already {job.status}",
        )

    DatabaseService(db).update_simulation_status(
        job.simulation_id,
        status="cancelled",
        progress=job.progress,
        current_step="Cancelled",
        message="Cancellation requested",
    )
    return SimulationResponse(
        simulation_id=simulation_id,
        status="cancelled",
        message="Cancellation requested",
    )


@router.websocket("/{simulation_id}/ws")
async def websocket_status(websocket: WebSocket, simulation_id: str):
    """WebSocket endpoint for real-time simulation status updates."""
    await websocket.accept()

    try:
        while True:
            sim_status = status_store.get(simulation_id)
            if not sim_status:
                await websocket.send_json({"error": f"Simulation {simulation_id} not found"})
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
            if sim_status.status in ("completed", "failed", "cancelled"):
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
