"""Simulation-related schemas."""
from typing import Any, Optional
from pydantic import BaseModel, Field


class SimulationParamsBase(BaseModel):
    """Base simulation parameters."""

    source_name: str = Field(..., description="Source name")
    member_ouid: str = Field(..., description="Member OUS UID")
    project_name: str = Field(..., description="Project name")
    ra: float = Field(..., description="Right ascension in degrees")
    dec: float = Field(..., description="Declination in degrees")
    band: float = Field(..., description="Band number")
    ang_res: float = Field(..., description="Angular resolution in arcsec")
    vel_res: float = Field(..., description="Velocity resolution in km/s")
    fov: float = Field(..., description="Field of view in arcsec")
    obs_date: str = Field(..., description="Observation date")
    pwv: float = Field(..., description="Precipitable water vapor")
    int_time: float = Field(..., description="Integration time in seconds")
    bandwidth: float = Field(..., description="Bandwidth in GHz")
    freq: float = Field(..., description="Frequency in GHz")
    freq_support: str = Field(..., description="Frequency support string")
    cont_sens: float = Field(..., description="Continuum sensitivity in mJy/beam")
    antenna_array: str = Field(..., description="Antenna array configuration")
    source_type: str = Field(default="point", description="Source type")
    n_pix: Optional[int] = Field(None, description="Number of pixels")
    n_channels: Optional[int] = Field(None, description="Number of channels")
    tng_api_key: Optional[str] = Field(None, description="TNG API key")
    ncpu: int = Field(default=4, description="Number of CPUs")
    rest_frequency: Optional[Any] = Field(None, description="Rest frequency")
    redshift: Optional[float] = Field(None, description="Redshift")
    lum_infrared: Optional[float] = Field(None, description="Infrared luminosity")
    snr: float = Field(default=1.3, description="Signal-to-noise ratio")
    n_lines: Optional[int] = Field(None, description="Number of spectral lines")
    line_names: Optional[Any] = Field(None, description="Line names")
    save_mode: str = Field(default="npz", description="Save mode")
    inject_serendipitous: bool = Field(default=False, description="Inject serendipitous sources")
    robust: float = Field(default=0.0, description="Robustness parameter")
    compute_backend: Optional[str] = Field(default="local", description="Computation backend type: local, dask, slurm, or kubernetes")
    compute_backend_config: Optional[dict] = Field(default_factory=dict, description="Backend-specific configuration")


class SimulationParamsCreate(SimulationParamsBase):
    """Simulation parameters for creation."""

    idx: int = Field(..., description="Simulation index")
    main_dir: Optional[str] = Field(None, description="Main directory path (uses settings if not provided)")
    output_dir: Optional[str] = Field(None, description="Output directory path (uses settings if not provided)")
    tng_dir: Optional[str] = Field(None, description="TNG directory path (uses settings if not provided)")
    galaxy_zoo_dir: Optional[str] = Field(None, description="Galaxy Zoo directory path (uses settings if not provided)")
    hubble_dir: Optional[str] = Field(None, description="Hubble directory path (uses settings if not provided)")


class SimulationResponse(BaseModel):
    """Simulation response."""

    simulation_id: str = Field(..., description="Simulation ID")
    status: str = Field(..., description="Simulation status")
    output_path: Optional[str] = Field(None, description="Output file path")
    message: Optional[str] = Field(None, description="Status message")


class SimulationStatus(BaseModel):
    """Simulation status."""

    simulation_id: str = Field(..., description="Simulation ID")
    status: str = Field(..., description="Status")
    progress: Optional[float] = Field(None, description="Progress percentage")
    message: Optional[str] = Field(None, description="Status message")
    current_step: Optional[str] = Field(None, description="Current simulation step")
    logs: Optional[list[str]] = Field(None, description="Recent log entries")
    error: Optional[str] = Field(None, description="Error message if failed")


