"""Simulation business logic service."""

from pathlib import Path
from typing import Any, Optional

import almasim.services.simulation as sim_service
from almasim.services.compute.base import ComputationBackend
from almasim.services.simulation import SimulationParams
from app.schemas.simulation import SimulationParamsCreate
from app.services.status_store import status_store


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

    def run_simulation(
        self,
        simulation_id: str,
        params: SimulationParamsCreate,
    ) -> None:
        """Run a simulation."""
        # Define simulation steps for progress tracking
        simulation_steps = [
            "Initializing",
            "Generating antenna configuration",
            "Computing max baseline",
            "Creating sky model",
            "Running interferometric simulation",
            "Processing results",
            "Saving output",
        ]

        # Track current progress state
        current_progress = {"value": 0.0, "step_index": 0}

        def status_callback(message: str):
            """Callback to update simulation status."""
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
                # Each step gets equal weight except "Running interferometric simulation" which gets more
                if step_index < 4:  # Before running simulation
                    base_progress = (step_index / len(simulation_steps)) * 50  # 0-50%
                elif (
                    step_index == 4
                ):  # During running simulation (handled by progress_callback)
                    base_progress = 50  # Start of simulation
                else:  # After simulation (processing, saving)
                    base_progress = 70 + ((step_index - 5) / 2) * 30  # 70-100%

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

        def log_callback(message: str):
            """Callback to log messages."""
            status_store.update(simulation_id, log=message)

        def progress_callback(progress: int):
            """Callback for fine-grained progress updates during simulation step."""
            # Only update progress if we're in the "Running interferometric simulation" step
            if current_progress["step_index"] == 4:
                # Map 0-100 progress to 50-70% range
                overall_progress = 50 + (progress * 0.20)
                current_progress["value"] = overall_progress
                status_store.update(simulation_id, progress=overall_progress)

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
                inject_serendipitous=params.inject_serendipitous,
                remote=False,
            )

            status_store.update(
                simulation_id,
                status="running",
                progress=0.0,
                current_step="Initializing",
                output_dir=sim_params.output_dir,
            )

            # Run simulation with callbacks
            sim_service.run_simulation(
                sim_params,
                compute_backend=self.compute_backend,
                robust=params.robust,
                status_callback=status_callback,
                progress_emitter=progress_callback,
                logger=log_callback,
            )

            # Mark as completed
            status_store.update(
                simulation_id,
                status="completed",
                progress=100.0,
                current_step="Completed",
                message="Simulation completed successfully",
            )
        except Exception as e:
            # Mark as failed
            error_msg = str(e)
            status_store.update(
                simulation_id,
                status="failed",
                error=error_msg,
                message=f"Simulation failed: {error_msg}",
            )
            log_callback(f"ERROR: {error_msg}")
            raise
