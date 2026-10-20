"""Instrument model shared by the real-data pipeline and the simulator (MADIV-28 T2.11).

One implementation per instrument fact (array types, and later geometry,
primary beam, spectral windows and constants), used by both sides instead of
a copy each.
"""

from .array import derive_array_type

__all__ = ["derive_array_type"]
