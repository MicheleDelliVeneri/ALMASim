"""Service layer for ALMASim.

This package contains modules that expose headless functions
that used to live inside the PyQt6 UI. Each module should focus on a
single area (metadata queries, dataset downloads, simulation orchestration,…)
so that future backends (FastAPI, CLI, notebooks) can compose them
without pulling in GUI dependencies.
"""

from importlib import import_module

_MODULES = {
    "simulation",
    "astro",
    "interferometry",
    "compute",
    "download",
    "metadata",
    "observation_plan",
    "imaging",
    "products",
    "external_skymodel",
}


def __getattr__(name):
    """Lazily import service submodules on first access."""
    if name in _MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "simulation",
    "astro",
    "interferometry",
    "compute",
    "download",
    "metadata",
    "observation_plan",
    "imaging",
    "products",
    "external_skymodel",
]
