"""Simulation business logic service."""
from pathlib import Path
from typing import Any, Optional
from dask.distributed import Client

import almasim.services.simulation as sim_service
from almasim.services.simulation import SimulationParams

from app.schemas.simulation import SimulationParamsCreate


class SimulationService:
    """Service for managing simulations."""

    def __init__(
        self,
        main_dir: Path,
        output_dir: Path,
        tng_dir: Path,
        galaxy_zoo_dir: Path,
        hubble_dir: Path,
        dask_client: Optional[Client] = None,
    ):
        """Initialize simulation service."""
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.tng_dir = tng_dir
        self.galaxy_zoo_dir = galaxy_zoo_dir
        self.hubble_dir = hubble_dir
        self.dask_client = dask_client

    def run_simulation(
        self,
        simulation_id: str,
        params: SimulationParamsCreate,
    ) -> None:
        """Run a simulation."""
        # Convert API params to internal SimulationParams
        sim_params = SimulationParams(
            idx=params.idx,
            source_name=params.source_name,
            member_ouid=params.member_ouid,
            main_dir=str(self.main_dir),
            output_dir=str(self.output_dir),
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

        # Run simulation
        sim_service.run_simulation(
            sim_params,
            dask_client=self.dask_client,
            robust=params.robust,
        )


