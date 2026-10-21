"""Visualizer API endpoints for product inspection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from almasim.services.products.preview import (
    FITS_SUFFIXES,
    MSV2_SUFFIXES,
    NPZ_SUFFIXES,
    classify_product_path,
    integrated_preview,
    iter_supported_products,
)
from app.core.config import settings
from app.core.path_utils import resolve_safe_path

router = APIRouter()


def _resolve_visualizer_path(path: str | Path | None, base_dir: Path) -> Path:
    if not path:
        return base_dir.expanduser().resolve()
    return resolve_safe_path(str(path), base_dir)


def _build_integrated_response(path: Path, method: str, **options) -> JSONResponse:
    return JSONResponse(integrated_preview(path, method, **options))


@router.get("/files")
def list_datacube_files(dir: str | None = None) -> JSONResponse:
    """List supported visualizer products in the given directory."""
    output_dir = _resolve_visualizer_path(dir, Path(settings.OUTPUT_DIR))

    if not output_dir.exists():  # lgtm[py/path-injection]
        return JSONResponse(
            {
                "files": [],
                "output_dir": str(output_dir),
                "message": "Output directory does not exist",
            }
        )

    files_info = []
    for file_path in iter_supported_products(output_dir):
        try:
            stat = file_path.stat()
            file_type = classify_product_path(file_path)
            if file_type is None:
                continue
            files_info.append(
                {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(output_dir)),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": file_type,
                }
            )
        except Exception:  # noqa: BLE001 - unreadable files are left out
            continue

    return JSONResponse(
        {
            "files": files_info,
            "output_dir": str(output_dir),
        }
    )


@router.get("/files/{file_path:path}")
def get_datacube_file(file_path: str, dir: str | None = None) -> FileResponse:
    """Get a visualizer file from the output directory."""
    sub = Path(dir) / file_path if dir else Path(file_path)
    resolved = _resolve_visualizer_path(sub, Path(settings.OUTPUT_DIR))

    if not resolved.exists() or not resolved.is_file():  # lgtm[py/path-injection]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )

    file_type = classify_product_path(resolved)
    if file_type not in {"npz", "fits"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .npz and FITS files can be downloaded directly",
        )

    return FileResponse(
        path=str(resolved),  # lgtm[py/path-injection]
        filename=resolved.name,
        media_type="application/octet-stream",
    )


@router.post("/integrate")
async def integrate_datacube(  # noqa: C901
    method: str = Form("sum"),
    integration_axis: str = Form("auto"),
    complex_component: str = Form("real"),
    channel_start: int | None = Form(default=None),
    channel_end: int | None = Form(default=None),
    file: UploadFile | None = File(default=None),
    server_path: str | None = Form(default=None),
    dir: str | None = Form(default=None),
) -> JSONResponse:
    """Load a supported visualizer product and produce a 2D inspection image."""
    if method not in ("sum", "mean"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Method must be 'sum' or 'mean'",
        )
    if file is None and not server_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either an uploaded file or a server-side file path",
        )
    if file is not None and server_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Choose either uploaded file or server-side file path, not both",
        )

    tmp_path: str | None = None
    try:
        if server_path:
            sub = Path(dir) / server_path if dir else Path(server_path)
            resolved = _resolve_visualizer_path(sub, Path(settings.OUTPUT_DIR))
            if not resolved.exists():  # lgtm[py/path-injection]
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found",
                )
            if classify_product_path(resolved) is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only .npz, FITS, and .ms files are supported",
                )
            # Loading, gridding and integrating a cube is CPU work: keep it off the event loop.
            return await run_in_threadpool(
                _build_integrated_response,
                resolved,
                method,
                integration_axis=integration_axis,
                complex_component=complex_component,
                channel_start=channel_start,
                channel_end=channel_end,
            )

        assert file is not None
        filename = file.filename or "uploaded-product"
        lower_name = filename.lower()
        if lower_name.endswith(MSV2_SUFFIXES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MSv2 directories must be selected from a server-side path",
            )
        if not lower_name.endswith(NPZ_SUFFIXES + FITS_SUFFIXES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .npz and FITS uploads are supported",
            )

        contents = await file.read()
        suffix = Path(filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        return await run_in_threadpool(
            _build_integrated_response,
            Path(tmp_path),
            method,
            integration_axis=integration_axis,
            complex_component=complex_component,
            channel_start=channel_start,
            channel_end=channel_end,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process datacube: {exc}",
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
