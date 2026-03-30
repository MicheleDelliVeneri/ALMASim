from .__version__ import __version__
from .services.simulation import (
    SimulationParams,
    export_results,
    generate_clean_cube,
    image_products,
    run_simulation,
    simulate_observation,
    write_ml_dataset_shard,
)

__all__ = [
    "__version__",
    "SimulationParams",
    "generate_clean_cube",
    "simulate_observation",
    "image_products",
    "export_results",
    "write_ml_dataset_shard",
    "run_simulation",
]

__doc__ = """
ALMASim provides headless building blocks for generating realistic ALMA
observations.  The legacy PyQt GUI has been removed in favour of reusable
services that can be orchestrated by CLIs, FastAPI backends, or notebooks.
"""
