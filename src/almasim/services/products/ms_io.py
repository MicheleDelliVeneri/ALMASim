"""Compatibility re-exports: the MS writer and reader moved to ``services.ms`` (MADIV-28 T4.9).

Removed in T8.4.
"""

from __future__ import annotations

from ..ms.backends import _CasacoreBackend, _CasatoolsBackend, _get_backend
from ..ms.io import export_native_ms, read_native_ms
from ..ms.schema import _arr_desc, _make_tabledesc, _sca_desc, _vtype

__all__ = [
    "_CasacoreBackend",
    "_CasatoolsBackend",
    "_arr_desc",
    "_get_backend",
    "_make_tabledesc",
    "_sca_desc",
    "_vtype",
    "export_native_ms",
    "read_native_ms",
]
