"""Turn nested metadata into JSON-serialisable values (MADIV-28 T2.8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as U
import numpy as np


def json_safe(value: Any) -> Any:
    """``value`` with dicts, sequences, numpy values, Quantities and paths made JSON-safe.

    Quantities are checked before ndarrays: a Quantity is an ndarray whose
    ``tolist()`` refuses to run. Array-likes such as astropy's StringArray go
    through ``tolist()``; anything else that JSON cannot hold becomes its str().
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, U.Quantity):
        return {"value": json_safe(value.value), "unit": str(value.unit)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if not isinstance(value, (str, int, float, bool, type(None))):
        return str(value)
    return value
