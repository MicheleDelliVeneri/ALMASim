"""Simulation-related schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObservationConfigInput(BaseModel):
    """One configuration of a multi-configuration observation plan.

    Any field left out takes the observation's value (``observation_plan``);
    other keys the library accepts (``antennalist``) pass through.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    array_type: Literal["12m", "7m", "TP"] | None = None
    antenna_array: str | None = None
    total_time_s: float | None = None
    correlator: str | None = None
    antenna_diameter_m: float | None = None


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
    line_sens_10kms: float | None = Field(
        default=None,
        description="Optional 10 km/s line sensitivity in mJy/beam",
    )
    antenna_array: str = Field(..., description="Antenna array configuration")
    observation_configs: list[ObservationConfigInput] | None = Field(
        default=None,
        description="Optional multi-configuration single-pointing observation plan",
    )
    source_type: str = Field(default="point", description="Source type (see almasim SKYMODELS)")
    n_pix: int | None = Field(None, description="Number of pixels")
    n_channels: int | None = Field(None, description="Number of channels")
    tng_api_key: str | None = Field(None, description="TNG API key")
    ncpu: int = Field(default=4, description="Number of CPUs")
    rest_frequency: Any | None = Field(None, description="Rest frequency")
    redshift: float | None = Field(None, description="Redshift")
    lum_infrared: float | None = Field(None, description="Infrared luminosity")
    snr: float | None = Field(
        default=None,
        description=("Optional manual signal-to-noise ratio override; leave null to auto-derive"),
    )
    n_lines: int | None = Field(None, description="Number of spectral lines")
    line_names: Any | None = Field(None, description="Line names")
    save_mode: str = Field(
        default="fits", description="Image save mode: npz, fits, both, h5, or memory"
    )
    ms_save_mode: str = Field(
        default="msv2", description="MS/visibility save mode: npz, msv2, or both"
    )
    persist: bool = Field(
        default=True,
        description="Whether to persist standard simulation outputs to disk",
    )
    ml_dataset_path: str | None = Field(
        default=None,
        description="Optional HDF5 shard path for ML dataset export",
    )
    inject_serendipitous: bool = Field(default=False, description="Inject serendipitous sources")
    robust: float = Field(default=0.0, description="Robustness parameter")
    imaging_algorithm: Literal["legacy"] = Field(
        default="legacy",
        description="Imaging implementation (ducc0 is disabled until its uv geometry is fixed)",
    )
    compute_backend: str | None = Field(
        default=None,
        description=(
            "Computation backend type: local, dask, slurm, or kubernetes. "
            "Leave it out to use the server's configured backend (COMPUTE_BACKEND)."
        ),
    )
    compute_backend_config: dict | None = Field(
        default_factory=dict, description="Backend-specific configuration"
    )
    ground_temperature_k: float = Field(
        default=270.0,
        description="Ground temperature used by the PWV-aware noise model",
    )
    correlator: str | None = Field(
        default=None,
        description="Optional correlator label stored in the observation plan",
    )
    elevation_deg: float | None = Field(
        default=None,
        description="Optional source elevation override for the noise model",
    )
    source_offset_x_arcsec: float = Field(
        default=0.0,
        description=("Optional source offset along x in arcsec relative to phase center"),
    )
    source_offset_y_arcsec: float = Field(
        default=0.0,
        description=("Optional source offset along y in arcsec relative to phase center"),
    )
    background_mode: str = Field(
        default="none",
        description=("Background sky mode: none, blank_field_dsfg, dusty_diffuse, or combined"),
    )
    background_level: float = Field(
        default=1.0,
        description="Relative amplitude for the simulated background sky",
    )
    background_seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducible background sky generation",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for the whole run: sky, noise and background "
            "(none: fresh randomness every run)"
        ),
    )
    external_skymodel_path: str | None = Field(
        default=None,
        description="Optional path to an external FITS image or FITS cube sky model",
    )
    external_component_table_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a component-list-like source table for external-components mode"
        ),
    )
    external_alignment_mode: str = Field(
        default="observation",
        description="External model alignment mode: observation or preserve",
    )
    external_header_mode: str = Field(
        default="observation",
        description="External FITS header mode: observation or preserve",
    )
    external_header_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Optional FITS header keyword overrides applied to external models",
    )
    ms_export: bool = Field(
        default=False,
        description="Whether to generate a native MeasurementSet `.ms` export",
    )
    ms_export_dir: str | None = Field(
        default=None,
        description=("Optional target directory for the native MeasurementSet `.ms` export"),
    )

    @field_validator("source_type")
    @classmethod
    def _known_source_type(cls, value: str) -> str:
        """Reject an unknown sky-model family up front (422) instead of mid-run."""
        from almasim.skymodels.registry import skymodel_spec

        skymodel_spec(value)
        return value.lower()


class SimulationParamsCreate(SimulationParamsBase):
    """Simulation parameters for creation."""

    idx: int = Field(..., description="Simulation index")
    main_dir: str | None = Field(
        None, description="Main directory path (uses settings if not provided)"
    )
    output_dir: str | None = Field(
        None, description="Output directory path (uses settings if not provided)"
    )
    tng_dir: str | None = Field(
        None, description="TNG directory path (uses settings if not provided)"
    )
    galaxy_zoo_dir: str | None = Field(
        None, description="Galaxy Zoo directory path (uses settings if not provided)"
    )
    hubble_dir: str | None = Field(
        None, description="Hubble directory path (uses settings if not provided)"
    )


class SimulationResponse(BaseModel):
    """Simulation response."""

    simulation_id: str = Field(..., description="Simulation ID")
    status: str = Field(..., description="Simulation status")
    output_path: str | None = Field(None, description="Output file path")
    message: str | None = Field(None, description="Status message")


class SimulationStatus(BaseModel):
    """Simulation status."""

    simulation_id: str = Field(..., description="Simulation ID")
    status: str = Field(..., description="Status")
    progress: float | None = Field(None, description="Progress percentage")
    message: str | None = Field(None, description="Status message")
    current_step: str | None = Field(None, description="Current simulation step")
    logs: list[str] | None = Field(None, description="Recent log entries")
    error: str | None = Field(None, description="Error message if failed")


class SimulationEstimate(BaseModel):
    """Preflight simulation size estimate."""

    n_pix: int = Field(..., description="Resolved spatial pixel count")
    n_channels: int = Field(..., description="Resolved spectral channel count")
    cube_shape: list[int] = Field(..., description="Cube shape [channels, y, x]")
    cube_voxels: int = Field(..., description="Total number of voxels")
    cell_size_arcsec: float = Field(..., description="Estimated cell size in arcsec")
    beam_size_arcsec: float = Field(..., description="Estimated synthesized beam size in arcsec")
    raw_single_cube_gb: float = Field(..., description="Raw float32 size of one cube in GiB")
    raw_complex_cube_gb: float = Field(..., description="Raw complex64 size of one cube in GiB")
    estimated_standard_output_gb: float = Field(
        ...,
        description="Approximate raw GiB footprint of the standard output products",
    )
    note: str = Field(..., description="Estimate note")


class SimulationSummary(BaseModel):
    """Simulation summary for list view."""

    simulation_id: str = Field(..., description="Simulation ID")
    status: str = Field(..., description="Status")
    progress: float = Field(..., description="Progress percentage")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    error: str | None = Field(None, description="Error message if failed")
    output_dir: str | None = Field(None, description="Output directory path")


class SimulationListResponse(BaseModel):
    """Response for listing simulations."""

    simulations: list[SimulationSummary] = Field(..., description="List of simulations")
    total: int = Field(..., description="Total number of simulations")
