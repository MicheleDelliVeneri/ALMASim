"""casatools and python-casacore table backends for the MS writer (MADIV-28 T4.9).

Both take and return arrays in python-casacore's row-major layout (see
:mod:`.schema`); the casatools backend reverses the axes on the way in and out.
"""

from __future__ import annotations

import logging

import numpy as np

from ..casa import tools as casa_tools
from .schema import _make_tabledesc

logger = logging.getLogger(__name__)


class _CasatoolsBackend:
    def __init__(self):
        self._ct = casa_tools()

    @staticmethod
    def put(tb, name: str, value) -> None:
        tb.putcol(name, np.asarray(value).transpose())

    @staticmethod
    def put_cell(tb, name: str, row: int, value) -> None:
        tb.putcell(name, row, np.asarray(value).transpose())

    @staticmethod
    def get(tb, name: str) -> np.ndarray:
        return np.asarray(tb.getcol(name)).transpose()

    def create(self, path: str, coldesc_list: list, nrow: int):
        desc = _make_tabledesc(coldesc_list)
        tb = self._ct.table()
        tb.create(path, desc, nrow=nrow)
        return tb

    def open(self, path: str, readonly: bool = True):
        tb = self._ct.table()
        tb.open(path, nomodify=readonly)
        return tb


class _CasacoreBackend:
    def __init__(self):
        from casacore import tables as _  # noqa: F401 — fail fast if not installed

    @staticmethod
    def put(tb, name: str, value) -> None:
        tb.putcol(name, np.asarray(value))

    @staticmethod
    def put_cell(tb, name: str, row: int, value) -> None:
        tb.putcell(name, row, np.asarray(value))

    @staticmethod
    def get(tb, name: str) -> np.ndarray:
        return np.asarray(tb.getcol(name))

    def create(self, path: str, coldesc_list: list, nrow: int):
        from casacore.tables import table

        desc = _make_tabledesc(coldesc_list)
        return table(path, tabledesc=desc, nrow=nrow, readonly=False)

    def open(self, path: str, readonly: bool = True):
        from casacore.tables import table

        return table(path, readonly=readonly)


def _get_backend():
    for cls in (_CasatoolsBackend, _CasacoreBackend):
        try:
            return cls()
        except ImportError as exc:
            logger.debug("MS backend %s not installed: %s", cls.__name__, exc)
        except Exception as exc:
            # Installed but failing to import (e.g. casatools' runtime libraries):
            # say why before falling back instead of hiding it.
            logger.warning(
                "MS backend %s failed to load (%s: %s); trying the next one",
                cls.__name__,
                type(exc).__name__,
                exc,
            )
    raise ImportError(
        "MS export requires casatools or python-casacore.\n"
        "  pip install casatools        (requires Python ≤ 3.12)\n"
        "  pip install python-casacore  (broader Python support)\n"
        "See https://casadocs.readthedocs.io/en/stable/api/casatools.html"
    )
