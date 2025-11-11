"""Service layer for ALMASim.

This package contains modules that expose headless functions
that used to live inside the PyQt6 UI. Each module should focus on a
single area (metadata queries, dataset downloads, simulation orchestration,…)
so that future backends (FastAPI, CLI, notebooks) can compose them
without pulling in GUI dependencies.
"""

from . import simulation, astro, interferometry  # re-export convenience

__all__ = ["simulation", "astro", "interferometry"]
