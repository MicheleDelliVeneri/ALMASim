"""CASA set-up shared by unpack, calibrate and self-calibration (MADIV-28 T2.5).

``runtime_data`` finds and populates the measures data; ``environment`` writes
the per-process site config that casatools reads at import time.
"""

from .environment import configure_casa_environment
from .runtime_data import (
    ensure_casa_runtime_data,
    find_existing_casa_data,
    find_input_casa_data,
    has_casa_runtime_data,
)

__all__ = [
    "configure_casa_environment",
    "ensure_casa_runtime_data",
    "find_existing_casa_data",
    "find_input_casa_data",
    "has_casa_runtime_data",
]
