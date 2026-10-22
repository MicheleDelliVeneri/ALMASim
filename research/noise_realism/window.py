"""The research subset of MADIV-16 windows, and each one's imaging parameters.

    python window.py info I          shell variables for window I (MS, SPW, FIELD, TASK, NAME)
    python window.py params I MS     its imaging-parameters row, with MS as the data
"""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = "/SKAO_data/cycle11/MADIV-16"
WINDOWS = "/SKAO_data/cycle11/MADIV-28/noise/windows.csv"


def subset() -> pd.DataFrame:
    """The cheap windows (under 60 Mvoxels and 40 GB of MS), by frequency."""
    w = pd.read_csv(WINDOWS)
    cheap = w[(w.voxels_m < 60) & (w.ms_gb < 40)].sort_values("ghz").reset_index(drop=True)
    return cheap


def name(row) -> str:
    return f"{os.path.basename(row.task.split('/')[0]).removesuffix('.ms.split')}_spw{row.spw}"


if __name__ == "__main__":
    step, i = sys.argv[1], int(sys.argv[2])
    row = subset().iloc[i]
    if step == "info":
        print(f"MS={row.ms} SPW={row.spw} FIELD={row.field} TASK={ROOT}/cubes/{row.task} NAME={name(row)}")
    elif step == "params":
        params = pd.read_csv(f"{ROOT}/params.csv")
        real = params[(params.filename == row.ms) & (params.spectral_window_id == row.spw)].iloc[[0]].copy()
        real["filename"] = sys.argv[3]
        real["spectral_window_id"] = 0  # the split MS holds one window and one field
        real["target_field_ids"] = 0
        print(real.to_csv(index=False), end="")
