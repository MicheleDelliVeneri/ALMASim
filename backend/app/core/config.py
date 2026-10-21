"""Application configuration."""

from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

import almasim
from almasim.__version__ import __version__ as almasim_version

# Relative DATA_DIR and OUTPUT_DIR are taken from here (backend/), as backend/run.sh
# starts the server in this directory.
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    PROJECT_NAME: str = "ALMASim API"
    VERSION: str = almasim_version
    API_V1_STR: str = "/api/v1"

    # CORS
    BACKEND_CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Paths
    MAIN_DIR: Path = Field(
        default=Path(almasim.__file__).resolve().parent,
        description="Main ALMASim package directory (default: the installed package)",
    )
    DATA_DIR: Path = Field(
        default=Path("data"),
        description="Data directory (metadata, etc.)",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("outputs"),
        description="Default output directory for simulations",
    )
    HOST_DIR: Path = Field(
        default=Path("/host_home"),
        description="Host filesystem mount point exposed for file browsing and downloads",
    )
    TNG_DIR: Path = Field(
        default=Path("/data/TNG100-1"),
        description="TNG simulation data directory",
    )
    GALAXY_ZOO_DIR: Path = Field(
        default=Path("/data/galaxy_zoo"),
        description="Galaxy Zoo data directory",
    )
    HUBBLE_DIR: Path = Field(
        default=Path("/data/hubble"),
        description="Hubble data directory",
    )

    # Computation backend configuration
    COMPUTE_BACKEND: str = Field(
        default="local",
        description="Computation backend type: local, dask, slurm, or kubernetes",
    )
    COMPUTE_BACKEND_CONFIG: dict = Field(
        default_factory=dict,
        description="Backend-specific configuration (JSON dict)",
    )

    # Legacy Dask configuration (for backward compatibility)
    DASK_SCHEDULER: str = Field(
        default="threads",
        description="Dask scheduler type (deprecated, use COMPUTE_BACKEND)",
    )
    DASK_N_WORKERS: int = Field(
        default=4,
        description="Number of Dask workers (deprecated, use COMPUTE_BACKEND_CONFIG)",
    )

    # Where request-supplied paths may point (MADIV-28 T1.9): HOST_DIR, OUTPUT_DIR,
    # DATA_DIR and these extra roots, e.g. ["/SKAO_data"].
    ALLOWED_ROOTS: list[Path] = Field(
        default_factory=list,
        validation_alias=AliasChoices("ALLOWED_ROOTS", "ALMASIM_ALLOWED_ROOTS"),
        description="Extra directories requests may write under (JSON list)",
    )
    # Dask scheduler addresses requests may use, besides DASK_SCHEDULER.
    ALLOWED_SCHEDULERS: list[str] = Field(
        default_factory=list,
        description="Extra Dask scheduler addresses requests may use (JSON list)",
    )

    @model_validator(mode="after")
    def _absolute_dirs(self) -> "Settings":
        """Resolve relative DATA_DIR and OUTPUT_DIR once, against backend/ (MADIV-28 W0.3).

        Startup used to resolve DATA_DIR against backend/ while every other use
        took it from the working directory.
        """
        for name in ("DATA_DIR", "OUTPUT_DIR"):
            value = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, BACKEND_DIR / value)
        return self

    def write_roots(self) -> list[Path]:
        """Directories request-supplied paths must lie under, in lookup order."""
        return [self.HOST_DIR, self.OUTPUT_DIR, self.DATA_DIR, *self.ALLOWED_ROOTS]

    # Simulation defaults
    DEFAULT_SOURCE_TYPE: str = "point"
    DEFAULT_SNR: float = 1.3
    DEFAULT_SAVE_MODE: str = "npz"
    DEFAULT_ROBUST: float = 0.0

    # Database configuration
    DATABASE_URL: str = Field(
        default="postgresql://almasim:almasim_dev_password@localhost:5432/almasim",
        description="PostgreSQL database URL",
    )


settings = Settings()
