"""Simulation business logic service."""

import dataclasses
import uuid
from pathlib import Path
from typing import Any

import almasim.services.simulation as sim_service
from almasim.services.compute.base import ComputationBackend
from almasim.services.paths import UnsafePathError, safe_join
from almasim.services.simulation import SimulationParams
from app.db.config import get_db_context
from app.db.service import DatabaseService
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
        compute_backend: ComputationBackend | None = None,
        owns_backend: bool = False,
    ):
        """Initialize simulation service.

        With ``owns_backend`` the backend was made for this one job and is
        closed when the job ends; the application's default one is shared.
        """
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.tng_dir = tng_dir
        self.galaxy_zoo_dir = galaxy_zoo_dir
        self.hubble_dir = hubble_dir
        self.compute_backend = compute_backend
        self.owns_backend = owns_backend

    def _persist_status(
        self,
        simulation_id: str,
        *,
        status: str,
        progress: float | None = None,
        current_step: str | None = None,
        message: str | None = None,
        error: str | None = None,
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
        except Exception:  # noqa: BLE001 - the database is optional here
            # Keep the in-memory status path working even if the DB is unavailable.
            return

    def _safe_output_dir(self, user_dir: str | None) -> Path:
        """Return a validated output directory that must reside within the configured base."""
        if not user_dir:
            return self.output_dir
        try:
            return safe_join(self.output_dir, user_dir)
        except UnsafePathError as exc:
            raise ValueError("output_dir must be within the configured output directory") from exc

    def _simulation_params(self, params: SimulationParamsCreate) -> SimulationParams:
        """The internal SimulationParams for an API request."""
        return to_sim_params(
            params,
            main_dir=self.main_dir,
            output_dir=self._safe_output_dir(params.output_dir),
            tng_dir=self.tng_dir,
            galaxy_zoo_dir=self.galaxy_zoo_dir,
            hubble_dir=self.hubble_dir,
        )

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
        except Exception:  # noqa: BLE001 - the database is optional here
            return

    def run_simulation(  # noqa: C901
        self,
        simulation_id: str,
        params: SimulationParamsCreate,
    ) -> None:
        """Run a simulation."""

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
                raise sim_service.SimulationCancelled("Simulation cancelled")

        def status_callback(message: str):
            """Callback to update simulation status."""
            check_cancelled()
            # Try to match message to a simulation step
            step_index = next(
                (i for i, s in enumerate(simulation_steps) if s.lower() in message.lower()),
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
            # Only update progress if we're in the
            # "Running interferometric simulation" step
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
            sim_params = self._simulation_params(params)

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
        # The library raises the same error when its own stop_requested check
        # fires; both mean the user cancelled, not that the run failed.
        except sim_service.SimulationCancelled:
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
            # Not log_callback: it checks for cancellation and would raise again
            # from inside this handler.
            status_store.update(simulation_id, log=f"ERROR: {error_msg}")
            self._persist_log(simulation_id, f"ERROR: {error_msg}", level="ERROR")
            raise
        finally:
            if self.owns_backend and self.compute_backend is not None:
                self.compute_backend.close()


# The request's own directory fields are not used: the server's configured
# directories apply (the output directory after validation).
_SERVER_SET_FIELDS = {"main_dir", "output_dir", "tng_dir", "galaxy_zoo_dir", "hubble_dir", "remote"}


def to_sim_params(
    params: SimulationParamsCreate,
    *,
    main_dir: Any,
    output_dir: Any,
    tng_dir: Any,
    galaxy_zoo_dir: Any,
    hubble_dir: Any,
) -> SimulationParams:
    """The library's SimulationParams for an API request, with the server's directories."""
    names = {spec.name for spec in dataclasses.fields(SimulationParams)} - _SERVER_SET_FIELDS
    values = {key: value for key, value in params.model_dump().items() if key in names}
    return SimulationParams(
        **values,
        main_dir=main_dir,
        output_dir=output_dir,
        tng_dir=tng_dir,
        galaxy_zoo_dir=galaxy_zoo_dir,
        hubble_dir=hubble_dir,
        remote=False,
    )
