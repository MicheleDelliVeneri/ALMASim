from .__version__ import __version__
from .services.simulation import SimulationParams, run_simulation

__all__ = ["__version__", "SimulationParams", "run_simulation"]

__doc__ = """
ALMASim provides headless building blocks for generating realistic ALMA
observations.  The legacy PyQt GUI has been removed in favour of reusable
services that can be orchestrated by CLIs, FastAPI backends, or notebooks.
"""
