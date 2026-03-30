"""Visualizer API endpoints for datacube processing."""
import numpy as np
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, FileResponse

from app.core.config import settings

router = APIRouter()


@router.get("/files")
async def list_datacube_files(dir: Optional[str] = None) -> JSONResponse:
    """List available datacube files in the given directory (recursively).

    Parameters
    ----------
    dir : str, optional
        Absolute path to search. Defaults to settings.OUTPUT_DIR.

    Returns
    -------
    JSONResponse
        List of available .npz files with metadata
    """
    output_dir = Path(dir) if dir else Path(settings.OUTPUT_DIR)

    if not output_dir.exists():
        return JSONResponse({
            "files": [],
            "output_dir": str(output_dir),
            "message": "Output directory does not exist",
        })

    # Find all .npz files recursively
    files_info = []
    for file_path in sorted(output_dir.rglob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat = file_path.stat()
            files_info.append({
                "name": file_path.name,
                "path": str(file_path.relative_to(output_dir)),
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
        except Exception:
            continue

    return JSONResponse({
        "files": files_info,
        "output_dir": str(output_dir),
    })


@router.get("/files/{file_path:path}")
async def get_datacube_file(file_path: str, dir: Optional[str] = None) -> FileResponse:
    """Get a datacube file from the output directory.

    Parameters
    ----------
    file_path : str
        Relative path to the file within the output directory
    dir : str, optional
        Absolute path of the base directory. Defaults to settings.OUTPUT_DIR.

    Returns
    -------
    FileResponse
        The .npz file
    """
    output_dir = Path(dir) if dir else Path(settings.OUTPUT_DIR)
    file_path_obj = Path(file_path)
    
    # Security: ensure path is within output directory
    try:
        resolved = (output_dir / file_path_obj).resolve()
        resolved.relative_to(output_dir.resolve())
    except (ValueError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )
    
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )
    
    if not resolved.suffix.lower() == ".npz":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .npz files are supported",
        )
    
    return FileResponse(
        path=str(resolved),
        filename=resolved.name,
        media_type="application/octet-stream",
    )


@router.post("/integrate")
async def integrate_datacube(
    file: UploadFile = File(...),
    method: str = "sum",
) -> JSONResponse:
    """Load a datacube file and integrate over frequency axis.
    
    Parameters
    ----------
    file : UploadFile
        Datacube file (.npz format)
    method : str
        Integration method: 'sum' or 'mean' (default: 'sum')
    
    Returns
    -------
    JSONResponse
        Integrated 2D image data and metadata
    """
    if method not in ("sum", "mean"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Method must be 'sum' or 'mean'",
        )
    
    # Check file extension
    if not file.filename.endswith(".npz"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .npz files are supported",
        )
    
    try:
        # Read file content
        contents = await file.read()
        
        # Save to temporary location
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load .npz file
            data = np.load(tmp_path)
            
            # Find the datacube array (usually the first array or named 'arr_0')
            cube = None
            cube_name = None
            
            # Try common names first
            for name in ["modelCube", "dirtyCube", "clean_cube", "dirty_cube"]:
                if name in data:
                    cube = data[name]
                    cube_name = name
                    break
            
            # If not found, use the first array
            if cube is None:
                keys = list(data.keys())
                if len(keys) > 0:
                    cube_name = keys[0]
                    cube = data[cube_name]
                else:
                    raise ValueError("No arrays found in .npz file")
            
            # Check if it's a 3D cube (should be [n_pix_y, n_pix_x, n_channels] or [n_channels, n_pix_y, n_pix_x])
            if cube.ndim != 3:
                raise ValueError(f"Expected 3D datacube, got {cube.ndim}D array")
            
            # Determine axis order: assume frequency is the last axis
            # Common formats: [y, x, freq] or [freq, y, x]
            if cube.shape[0] < cube.shape[2]:
                # Likely [freq, y, x] - transpose to [y, x, freq]
                cube = np.transpose(cube, (1, 2, 0))
            
            # Integrate over frequency axis (last axis)
            if method == "sum":
                integrated = np.sum(cube, axis=2)
            else:  # mean
                integrated = np.nanmean(cube, axis=2)
            
            # Replace NaN and Inf with 0
            integrated = np.nan_to_num(integrated, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize for display (0-255 range)
            if integrated.max() > integrated.min():
                normalized = ((integrated - integrated.min()) / (integrated.max() - integrated.min()) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(integrated, dtype=np.uint8)
            
            # Get statistics
            stats = {
                "shape": cube.shape,
                "integrated_shape": integrated.shape,
                "min": float(integrated.min()),
                "max": float(integrated.max()),
                "mean": float(integrated.mean()),
                "std": float(integrated.std()),
                "cube_name": cube_name,
            }
            
            # Convert to list for JSON serialization
            image_data = normalized.tolist()
            
            return JSONResponse({
                "image": image_data,
                "stats": stats,
                "method": method,
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process datacube: {str(e)}",
        )

