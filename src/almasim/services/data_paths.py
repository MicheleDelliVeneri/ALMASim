"""Where ALMASim's bundled data files live, and cached readers for them.

``DataLayout(main_dir)`` names the files under the package data directory
(``main_dir``, the ``almasim`` package by default): the antenna positions, the
calibrated line list and the SED templates. The readers parse each file once
per process and hand out copies, so callers may modify what they get.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataLayout:
    """Paths of the bundled data files under ``main_dir``."""

    main_dir: Path

    def __init__(self, main_dir: str | os.PathLike[str]):
        object.__setattr__(self, "main_dir", Path(main_dir))

    @property
    def antenna_coordinates(self) -> Path:
        return self.main_dir / "antenna_config" / "antenna_coordinates.csv"

    @property
    def brightness_dir(self) -> Path:
        return self.main_dir / "brightnes"

    @property
    def calibrated_lines(self) -> Path:
        return self.brightness_dir / "calibrated_lines.csv"

    def sed_template(self, name: str) -> Path:
        return self.brightness_dir / name


@lru_cache(maxsize=32)
def _read_table(path: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def read_table(path: str | os.PathLike[str], sep: str = ",") -> pd.DataFrame:
    """A copy of the table at ``path``, parsed once per process."""
    return _read_table(os.fspath(path), sep).copy()


def antenna_names(antenna_array: str) -> list[str]:
    """Antenna names of an array string such as ``"A001:DA41 A002:DA42"``."""
    return [antenna.split(":")[0] for antenna in antenna_array.split(" ")]


def antenna_positions(layout: DataLayout, antenna_array: str) -> pd.DataFrame:
    """Rows of the antenna table (name, x, y, z in metres) for the array, in table order."""
    table = read_table(layout.antenna_coordinates)
    return table[table["name"].isin(antenna_names(antenna_array))].reset_index(drop=True)


def max_baseline_km(layout: DataLayout, antenna_array: str) -> float:
    """Longest baseline (km) between two antennas of the array; 0 for fewer than two."""
    from scipy.spatial.distance import pdist

    xyz = antenna_positions(layout, antenna_array)[["x", "y", "z"]].to_numpy(dtype=float)
    if len(xyz) < 2:
        return 0.0
    return float(np.max(pdist(xyz))) / 1000


def calibrated_lines(layout: DataLayout) -> pd.DataFrame:
    """The calibrated emission-line list (columns include ``Line`` and ``freq(GHz)``)."""
    return read_table(layout.calibrated_lines)
