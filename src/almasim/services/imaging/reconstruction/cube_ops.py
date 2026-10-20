"""Per-channel cube operations: regridding, beam convolution, padding (MADIV-28 T4.10)."""

from __future__ import annotations

import numpy as np


def regrid_cube_to_match(cube: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbor regrid of a cube onto a target `(chan, y, x)` shape."""
    cube = np.asarray(cube)
    if cube.shape == target_shape:
        return cube.copy()
    if cube.ndim != 3:
        raise ValueError("cube must have shape (n_channels, ny, nx)")

    n_chan, target_y, target_x = target_shape
    source_chan = np.linspace(0, cube.shape[0] - 1, n_chan).round().astype(int)
    source_y = np.linspace(0, cube.shape[1] - 1, target_y).round().astype(int)
    source_x = np.linspace(0, cube.shape[2] - 1, target_x).round().astype(int)
    return cube[source_chan][:, source_y][:, :, source_x]


def convolve_cube_with_beam(
    cube: np.ndarray,
    beam_cube: np.ndarray,
) -> np.ndarray:
    """Convolve a channel-first cube with a channel-matched beam cube."""
    cube = np.asarray(cube, dtype=np.float32)
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    if cube.shape != beam_cube.shape:
        raise ValueError("cube and beam_cube must have the same shape")

    convolved = np.zeros_like(cube, dtype=np.float32)
    for channel in range(cube.shape[0]):
        convolved[channel] = np.real(
            np.fft.ifft2(
                np.fft.fft2(cube[channel]) * np.fft.fft2(np.fft.ifftshift(beam_cube[channel]))
            )
        ).astype(np.float32)
    return convolved
