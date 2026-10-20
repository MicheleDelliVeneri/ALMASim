"""CASA set-up shared by unpack, calibrate and self-calibration (MADIV-28 T2.5).

``runtime_data`` finds and populates the measures data; ``environment`` writes
the per-process site config that casatools reads at import time. ``tools()``
and ``tasks()`` import CASA on first use, with one clear error when it is
missing, and give tests one place to patch (or a ``sys.modules`` fake).
"""

import importlib
from types import ModuleType

from .environment import configure_casa_environment
from .runtime_data import (
    ensure_casa_runtime_data,
    find_existing_casa_data,
    find_input_casa_data,
    has_casa_runtime_data,
)

_INSTALL_HINT = "install ALMASim with its CASA extra: pip install 'almasim[casa]'"


def _casa_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(f"CASA's {name} is not available; {_INSTALL_HINT}") from exc


def tools() -> ModuleType:
    """The ``casatools`` module, imported on first use."""
    return _casa_module("casatools")


def tasks() -> ModuleType:
    """The ``casatasks`` module, imported on first use."""
    return _casa_module("casatasks")


__all__ = [
    "tasks",
    "tools",
    "configure_casa_environment",
    "ensure_casa_runtime_data",
    "find_existing_casa_data",
    "find_input_casa_data",
    "has_casa_runtime_data",
]
