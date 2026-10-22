"""Model visibilities of image planes with ducc0 (MADIV-28 T3.7).

One prediction for ``almasim predict`` (a FITS model into an MS's
MODEL_DATA) and the simulator's image-based path.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np

# Threads for ducc0 gridding: $ALMASIM_NTHREADS, else every CPU.
ALMASIM_NTHREADS = int(os.environ.get("ALMASIM_NTHREADS") or os.cpu_count() or 1)


def _ducc0_with_wgridder_fallback(func: Callable[..., Any], /, **kwargs: Any) -> Any:
    """``func(**kwargs)``, retried without w-gridding when ducc0 needs too many w planes."""
    try:
        return func(**kwargs)
    except RuntimeError as exc:
        if kwargs.get("do_wgridding") and "too many w planes" in str(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["do_wgridding"] = False
            return func(**retry_kwargs)
        raise


def ducc0_plane(plane: np.ndarray) -> np.ndarray:
    """A FITS-ordered image plane ``(ny, nx)`` (RA increasing to the left, as
    FITS images are stored and WSClean writes them) as ducc0's wgridder takes
    it: ducc0 reads its first axis the other way round, so a FITS plane handed
    over as is comes back from imaging rotated by 90 degrees, (y, x) -> (n - x, y)
    (checked by imaging predicted visibilities with WSClean). Flipped in y
    about the reference pixel (n // 2: ducc0 takes even sizes only) and
    transposed, it images where FITS has it.
    """
    plane = np.asarray(plane)
    return np.swapaxes(np.roll(plane[..., ::-1, :], 1, axis=-2), -1, -2)


def predict_visibilities(
    uvw_m: np.ndarray,
    freq_hz: np.ndarray,
    planes: np.ndarray,
    pixsize_x: float,
    pixsize_y: float,
    *,
    weights: np.ndarray | None = None,
    epsilon: float = 1e-6,
    nthreads: int | None = None,
) -> np.ndarray:
    """Visibilities ``(nrows, nchan)`` of ``planes`` at ``uvw_m`` (metres) and ``freq_hz``.

    ``planes`` is one image ``(ny, nx)`` for every channel, or a cube
    ``(nchan, ny, nx)`` with one plane per channel, in FITS order (as read
    from a FITS file; see :func:`ducc0_plane`); pixel sizes are in radians.
    ducc0's w-gridder runs with w-correction, falling back to 2D gridding
    when the w range needs too many planes.
    """
    from ducc0.wgridder import dirty2vis

    uvw_m = np.asarray(uvw_m, dtype=np.float64)
    freq_hz = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    planes = np.asarray(planes)
    nthreads = ALMASIM_NTHREADS if nthreads is None else int(nthreads)

    def predict(plane: np.ndarray, frequencies: np.ndarray, wgt: np.ndarray | None) -> Any:
        kwargs: dict[str, Any] = dict(
            uvw=uvw_m,
            freq=frequencies,
            dirty=np.ascontiguousarray(ducc0_plane(plane)),
            # ducc0's first axis is the FITS x (RA) axis once transposed.
            pixsize_x=pixsize_x,
            pixsize_y=pixsize_y,
            epsilon=epsilon,
            do_wgridding=True,
            nthreads=nthreads,
        )
        if wgt is not None:
            kwargs["wgt"] = wgt
        return _ducc0_with_wgridder_fallback(dirty2vis, **kwargs)

    if planes.ndim == 2 or (planes.ndim == 3 and len(planes) == 1):
        return predict(planes.reshape(planes.shape[-2:]), freq_hz, weights)
    if planes.ndim != 3 or len(planes) != len(freq_hz):
        raise ValueError(
            f"{planes.shape[0] if planes.ndim == 3 else planes.shape} model planes for "
            f"{len(freq_hz)} channels (need 1 or {len(freq_hz)})"
        )
    return np.stack(
        [
            predict(
                planes[c],
                freq_hz[c : c + 1],
                None if weights is None else weights[:, c : c + 1],
            )[:, 0]
            for c in range(len(freq_hz))
        ],
        axis=1,
    )
