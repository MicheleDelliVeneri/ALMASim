"""Application configuration."""
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    PROJECT_NAME: str = "ALMASim API"
    VERSION: str = "2.1.10"
    API_V1_STR: str = "/api/v1"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins",
    )

    # Paths
    MAIN_DIR: Path = Field(
        default=Path("almasim"),
        description="Main ALMASim data directory",
    )
    OUTPUT_DIR: Path = Field(
        default=Path("outputs"),
        description="Default output directory for simulations",
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

    # Dask configuration
    DASK_SCHEDULER: str = Field(
        default="threads",
        description="Dask scheduler type",
    )
    DASK_N_WORKERS: int = Field(
        default=4,
        description="Number of Dask workers",
    )

    # Simulation defaults
    DEFAULT_SOURCE_TYPE: str = "point"
    DEFAULT_SNR: float = 1.3
    DEFAULT_SAVE_MODE: str = "npz"
    DEFAULT_ROBUST: float = 0.0


settings = Settings()


