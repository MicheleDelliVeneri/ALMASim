"""ALMA simulation: parameters, stages and the full pipeline (MADIV-28 T5.7).

``params`` holds :class:`SimulationParams`; ``geometry``, ``spectral_setup``,
``skymodel_stage``, ``background``, ``metadata`` and ``planning`` build the
clean cube that ``stages.generate_clean_cube`` returns; ``stages`` also runs
the observation and the image products, ``export`` writes them and
``pipeline.run_simulation`` chains everything.
"""

from ..products.cube_export import write_ml_dataset_shard
from .background import generate_background_cube
from .context import RunContext
from .export import export_results
from .geometry import (
    CubeGeometry,
    estimate_simulation_footprint,
    resolve_cube_geometry,
    resolve_source_pixel_position,
)
from .params import LogFn, SimulationParams, StopFn
from .pipeline import run_simulation
from .stages import CleanCubeStage, generate_clean_cube, image_products, simulate_observation

__all__ = [
    "CleanCubeStage",
    "CubeGeometry",
    "LogFn",
    "RunContext",
    "SimulationParams",
    "StopFn",
    "estimate_simulation_footprint",
    "export_results",
    "generate_background_cube",
    "generate_clean_cube",
    "image_products",
    "resolve_cube_geometry",
    "resolve_source_pixel_position",
    "run_simulation",
    "simulate_observation",
    "write_ml_dataset_shard",
]
