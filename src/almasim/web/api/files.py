"""Browse, read and fetch files under the profile's roots (web console plan, S2.4).

Every path must resolve (symlinks followed) inside one of the profile's
``roots`` or its runs root; anything else is a 403. The server runs as the
user, so Unix permissions decide the rest. Reads are bounded: text and CSV
come in pages, and a FITS cube is never loaded whole: one plane is read
through a memory map and thinned to at most ``MAX_PLANE`` pixels a side.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from almasim.jobs.profile import Profile
from almasim.services.paths import UnsafePathError, safe_join

router = APIRouter(prefix="/api/v2/files", tags=["files"])

MAX_ENTRIES = 5000
MAX_TEXT = 1024 * 1024
MAX_JSON = 4 * 1024 * 1024
MAX_PLANE = 1024


class Root(BaseModel):
    path: str
    exists: bool
    runs: bool  # the runs root


class Roots(BaseModel):
    roots: list[Root]


class Entry(BaseModel):
    name: str
    kind: Literal["folder", "file"]
    size: int | None
    modified: float


class Listing(BaseModel):
    path: str
    parent: str | None
    entries: list[Entry]
    truncated: bool


class Folder(BaseModel):
    path: str


class TextSlice(BaseModel):
    path: str
    text: str
    offset: int
    next_offset: int
    size: int


def _roots(profile: Profile) -> list[Path]:
    roots = [Path(r).expanduser() for r in (*profile.roots, profile.runs_root)]
    seen: dict[str, Path] = {}
    for root in roots:
        seen.setdefault(os.path.realpath(root), root)
    return list(seen.values())


def resolve(profile: Profile, raw: str) -> Path:
    """``raw`` resolved, if it lies under one of the allowed roots; else HTTP 403."""
    if not raw:
        raise HTTPException(status_code=400, detail="give a path")
    for root in _roots(profile):
        try:
            return safe_join(root, raw)
        except UnsafePathError:
            continue
    raise HTTPException(status_code=403, detail=f"{raw} is outside the console's folders")


def _existing(profile: Profile, raw: str, *, folder: bool = False) -> Path:
    path = resolve(profile, raw)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{raw} does not exist")
    if folder and not path.is_dir():
        raise HTTPException(status_code=400, detail=f"{raw} is not a folder")
    if not folder and path.is_dir():
        raise HTTPException(status_code=400, detail=f"{raw} is a folder")
    return path


@router.get("/roots", response_model=Roots)
def roots(request: Request) -> dict[str, Any]:
    profile: Profile = request.app.state.profile
    runs_root = os.path.realpath(Path(profile.runs_root).expanduser())
    return {
        "roots": [
            {
                "path": str(root),
                "exists": root.is_dir(),
                "runs": os.path.realpath(root) == runs_root,
            }
            for root in _roots(profile)
        ]
    }


@router.get("/list", response_model=Listing)
def list_folder(
    request: Request,
    path: str = Query(...),
    hidden: bool = Query(False, description="Include names starting with a dot."),
) -> dict[str, Any]:
    folder = _existing(request.app.state.profile, path, folder=True)
    entries: list[dict[str, Any]] = []
    try:
        scan = list(os.scandir(folder))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"cannot read {folder}") from exc
    for entry in scan:
        if not hidden and entry.name.startswith("."):
            continue
        try:
            is_dir = entry.is_dir()
            st = entry.stat()
        except OSError:  # a broken link or a file that just vanished
            continue
        entries.append(
            {
                "name": entry.name,
                "kind": "folder" if is_dir else "file",
                "size": None if is_dir else st.st_size,
                "modified": st.st_mtime,
            }
        )
    entries.sort(key=lambda e: (e["kind"] != "folder", e["name"].lower()))
    return {
        "path": str(folder),
        "parent": str(folder.parent) if folder.parent != folder else None,
        "entries": entries[:MAX_ENTRIES],
        "truncated": len(entries) > MAX_ENTRIES,
    }


class FolderRequest(BaseModel):
    path: str


@router.post("/mkdir", status_code=201, response_model=Folder)
def make_folder(body: FolderRequest, request: Request) -> dict[str, str]:
    folder = resolve(request.app.state.profile, body.path)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=400, detail=f"cannot create {folder}: {exc.strerror}"
        ) from exc
    return {"path": str(folder)}


@router.get("/text", response_model=TextSlice)
def read_text(
    request: Request,
    path: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(64 * 1024, ge=1, le=MAX_TEXT),
) -> dict[str, Any]:
    file = _existing(request.app.state.profile, path)
    size = file.stat().st_size
    with file.open("rb") as f:
        f.seek(min(offset, size))
        chunk = f.read(limit)
    text = chunk.decode("utf-8", errors="replace")
    return {
        "path": str(file),
        "text": text,
        "offset": offset,
        "next_offset": offset + len(chunk),
        "size": size,
    }


@router.get("/csv")
def read_csv(
    request: Request,
    path: str = Query(...),
    start: int = Query(0, ge=0, description="First data row to return."),
    rows: int = Query(100, ge=1, le=5000),
) -> dict[str, Any]:
    """A page of rows, with the header; ``total`` counts the data rows."""
    file = _existing(request.app.state.profile, path)
    with file.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        page: list[list[str]] = []
        total = 0
        for i, row in enumerate(reader):
            if start <= i < start + rows:
                page.append(row)
            total = i + 1
    return {"path": str(file), "columns": header, "rows": page, "start": start, "total": total}


@router.get("/json")
def read_json(request: Request, path: str = Query(...)) -> Any:
    import json

    file = _existing(request.app.state.profile, path)
    if file.stat().st_size > MAX_JSON:
        raise HTTPException(status_code=413, detail=f"{file.name} is larger than 4 MB")
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{file.name} is not JSON: {exc}") from exc


def _open_fits(file: Path):
    from astropy.io import fits

    try:
        return fits.open(file, memmap=True, lazy_load_hdus=True)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail=f"{file.name} is not a FITS file: {exc}"
        ) from exc


@router.get("/fits")
def fits_header(
    request: Request, path: str = Query(...), hdu: int = Query(0, ge=0)
) -> dict[str, Any]:
    file = _existing(request.app.state.profile, path)
    with _open_fits(file) as hdul:
        if hdu >= len(hdul):
            raise HTTPException(status_code=400, detail=f"{file.name} has {len(hdul)} HDUs")
        header = hdul[hdu].header
        cards = [
            {
                "key": c.keyword,
                "value": c.value if _plain(c.value) else str(c.value),
                "comment": c.comment,
            }
            for c in header.cards
            if c.keyword
        ]
        shape = list(hdul[hdu].shape) if hdul[hdu].shape else []
        return {"path": str(file), "hdus": len(hdul), "hdu": hdu, "shape": shape, "cards": cards}


def _plain(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


@router.get("/fits/plane")
def fits_plane(
    request: Request,
    path: str = Query(...),
    hdu: int = Query(0, ge=0),
    channel: int = Query(0, ge=0),
    stokes: int = Query(0, ge=0),
) -> dict[str, Any]:
    """One image plane, thinned to at most ``MAX_PLANE`` pixels a side (NaN as null)."""
    import numpy as np

    file = _existing(request.app.state.profile, path)
    with _open_fits(file) as hdul:
        data = hdul[hdu].data
        if data is None or data.ndim < 2:
            raise HTTPException(status_code=400, detail="that HDU holds no image")
        index: list[int] = []
        lead = data.ndim - 2  # (STOKES, FREQ, Y, X) or (FREQ, Y, X) or (Y, X)
        wanted = ([0] * (lead - 2) + [stokes, channel])[-lead:] if lead else []
        for axis, want in enumerate(wanted):
            if want >= data.shape[axis]:
                raise HTTPException(
                    status_code=400, detail=f"axis {axis} has {data.shape[axis]} planes"
                )
            index.append(want)
        plane = np.asarray(data[tuple(index)], dtype=np.float64)
    step = max(1, math.ceil(max(plane.shape) / MAX_PLANE))
    thinned = plane[::step, ::step]
    finite = thinned[np.isfinite(thinned)]
    values = np.where(np.isfinite(thinned), thinned, np.nan)
    return {
        "path": str(file),
        "shape": list(plane.shape),
        "step": step,
        "channel": channel,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "data": [[None if math.isnan(v) else v for v in row] for row in values.tolist()],
    }


@router.get("/download")
def download(request: Request, path: str = Query(...)) -> FileResponse:
    file = _existing(request.app.state.profile, path)
    return FileResponse(file, filename=file.name)
