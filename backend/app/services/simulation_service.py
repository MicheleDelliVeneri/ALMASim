"""Simulation business logic service."""

import uuid
from pathlib import Path
from typing import Any, Optional

import almasim.services.simulation as sim_service
from almasim.services.compute.base import ComputationBackend
from almasim.services.simulation import SimulationParams
from app.schemas.simulation import SimulationParamsCreate
from app.services.status_store import status_store
from database.config import get_db_context
from database.service import DatabaseService


class SimulationService:
    """Service for managing simulations."""

    def __init__(
        self,
        main_dir: Path,
        output_dir: Path,
        tng_dir: Path,
        galaxy_zoo_dir: Path,
        hubble_dir: Path,
        compute_backend: Optional[ComputationBackend] = None,
    ):
        """Initialize simulation service."""
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.tng_dir = tng_dir
        self.galaxy_zoo_dir = galaxy_zoo_dir
        self.hubble_dir = hubble_dir
        self.compute_backend = compute_backend

    def _persist_status(
        self,
        simulation_id: str,
        *,
        status: str,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Persist simulation status in the database when available."""
        try:
            with get_db_context() as db:
                DatabaseService(db).update_simulation_status(
                    uuid.UUID(simulation_id),
                    status=status,
                    progress=progress,
                    current_step=current_step,
                    message=message,
                    error=error,
                )
        except Exception:
            # Keep the in-memory status path working even if the DB is unavailable.
            return

    def _persist_log(
        self,
        simulation_id: str,
        message: str,
        *,
        level: str = "INFO",
    ) -> None:
        """Persist a simulation log entry in the database when available."""
        try:
            with get_db_context() as db:
                DatabaseService(db).add_simulation_log(
                    uuid.UUID(simulation_id),
                    message=message,
                    level=level,
                )
        except Exception:
            return

    def run_simulation(
        self,
        simulation_id: str,
        params: SimulationParamsCreate,
    ) -> None:
        """Run a simulation."""
        class SimulationCancelledError(RuntimeError):
            pass

        # Define simulation steps for progress tracking
        simulation_steps = [
            "Initializing",
            "Generating clean cube",
            "Running interferometric simulation",
            "Reconstructing image products",
            "Exporting results",
        ]

        # Track current progress state
        current_progress = {"value": 0.0, "step_index": 0}

        def check_cancelled() -> None:
            if status_store.is_cancelled(simulation_id):
                raise SimulationCancelledError("Simulation cancelled")

        def status_callback(message: str):
            """Callback to update simulation status."""
            check_cancelled()
            # Try to match message to a simulation step
            step_index = next(
                (
                    i
                    for i, s in enumerate(simulation_steps)
                    if s.lower() in message.lower()
                ),
                None,
            )

            if step_index is not None:
                # Update step index and calculate base progress
                current_progress["step_index"] = step_index
                if step_index == 0:
                    base_progress = 0.0
                elif step_index == 1:
                    base_progress = 20.0
                elif step_index == 2:
                    base_progress = 50.0
                elif step_index == 3:
                    base_progress = 85.0
                else:
                    base_progress = 95.0

                current_progress["value"] = base_progress

            # Extract step name from message
            current_step = next(
                (s for s in simulation_steps if s.lower() in message.lower()), message
            )

            status_store.update(
                simulation_id,
                status="running",
                progress=current_progress["value"],
                current_step=current_step,
                message=message,
            )
            self._persist_status(
                simulation_id,
                status="running",
                progress=current_progress["value"],
                current_step=current_step,
                message=message,
            )

        def log_callback(message: str):
            """Callback to log messages."""
            check_cancelled()
            status_store.update(simulation_id, log=message)
            self._persist_log(simulation_id, message)

        def progress_callback(progress: int):
            """Callback for fine-grained progress updates during simulation step."""
            check_cancelled()
            # Only update progress if we're in the "Running interferometric simulation" step
            if current_progress["step_index"] == 2:
                # Map 0-100 progress to 50-85% range
                overall_progress = 50 + (progress * 0.35)
                current_progress["value"] = overall_progress
                status_store.update(simulation_id, progress=overall_progress)
                self._persist_status(
                    simulation_id,
                    status="running",
                    progress=overall_progress,
                )

        try:
            # Convert API params to internal SimulationParams
            sim_params = SimulationParams(
                idx=params.idx,
                source_name=params.source_name,
                member_ouid=params.member_ouid,
                main_dir=str(self.main_dir),
                output_dir=params.output_dir or str(self.output_dir),
                tng_dir=str(self.tng_dir),
                galaxy_zoo_dir=str(self.galaxy_zoo_dir),
                hubble_dir=str(self.hubble_dir),
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
            )

            status_store.update(
                simulation_id,
                status="running",
                progress=0.0,
                current_step="Initializing",
                output_dir=sim_params.output_dir,
            )
            self._persist_status(
                simulation_id,
                status="running",
                progress=0.0,
                current_step="Initializing",
                message="Simulation started",
            )

            # Run simulation with callbacks
            sim_service.run_simulation(
                sim_params,
                compute_backend=self.compute_backend,
                robust=params.robust,
                status_callback=status_callback,
                progress_emitter=progress_callback,
                logger=log_callback,
                stop_requested=lambda: status_store.is_cancelled(simulation_id),
            )

            # Mark as completed
            status_store.update(
                simulation_id,
                status="completed",
                progress=100.0,
                current_step="Completed",
                message="Simulation completed successfully",
            )
            self._persist_status(
                simulation_id,
                status="completed",
                progress=100.0,
                current_step="Completed",
                message="Simulation completed successfully",
            )
        except SimulationCancelledError:
            status_store.update(
                simulation_id,
                status="cancelled",
                progress=current_progress["value"],
                current_step="Cancelled",
                message="Simulation cancelled",
            )
            self._persist_status(
                simulation_id,
                status="cancelled",
                progress=current_progress["value"],
                current_step="Cancelled",
                message="Simulation cancelled",
            )
            self._persist_log(simulation_id, "Simulation cancelled", level="WARNING")
        except Exception as e:
            # Mark as failed
            error_msg = str(e)
            status_store.update(
                simulation_id,
                status="failed",
                error=error_msg,
                message=f"Simulation failed: {error_msg}",
            )
            self._persist_status(
                simulation_id,
                status="failed",
                error=error_msg,
                message=f"Simulation failed: {error_msg}",
            )
            log_callback(f"ERROR: {error_msg}")
            raise
