"""How big a simulation will be, before it runs (web console plan, U3.6).

The cube's geometry comes from one metadata row and the size options, by
the same code a simulation uses (``estimate_simulation_footprint``); it is
arithmetic on the row, quick enough for the console to answer while the
form is being filled in. Nothing is written.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .files import _existing

router = APIRouter(prefix="/api/v2/simulation", tags=["simulation"])

# Where `almasim simulation` finds its data by default (its --main-dir).
MAIN_DIR = Path(__file__).resolve().parents[2]


class EstimateRequest(BaseModel):
    metadata_path: str
    row_idx: int = Field(0, ge=0)
    n_pix: int | None = Field(None, ge=1)
    n_channels: int | None = Field(None, ge=1)
    source_type: str = "point"
    instrument_geometry: str = "legacy"


class Estimate(BaseModel):
    source_name: str
    band: str
    n_pix: int
    n_channels: int
    cube_shape: list[int]
    cell_size_arcsec: float
    beam_size_arcsec: float
    raw_single_cube_gb: float
    estimated_standard_output_gb: float
    note: str


def _row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i == index:
                return row
    raise HTTPException(status_code=400, detail=f"{path.name} has no row {index}")


@router.post("/estimate", response_model=Estimate)
def estimate(body: EstimateRequest, request: Request) -> dict[str, Any]:
    """The cube a simulation of one metadata row would make, and its raw size."""
    from almasim.services.simulation.geometry import estimate_simulation_footprint
    from almasim.services.simulation.params import SimulationParams

    row = _row(_existing(request.app.state.profile, body.metadata_path), body.row_idx)
    unused = "/nonexistent/almasim-estimate"  # dirs the estimate never touches
    try:
        params = SimulationParams.from_metadata_row(
            row,
            idx=body.row_idx,
            main_dir=str(MAIN_DIR),
            output_dir=unused,
            tng_dir=unused,
            galaxy_zoo_dir=unused,
            hubble_dir=unused,
            project_name="estimate",
            n_pix=body.n_pix,
            n_channels=body.n_channels,
            source_type=body.source_type,
            instrument_geometry=body.instrument_geometry,
            persist=False,
        )
        footprint = estimate_simulation_footprint(params)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"cannot estimate: {exc}") from exc
    return {
        **footprint,
        "source_name": str(params.source_name),
        "band": str(row.get("Band") or row.get("band") or ""),
    }
