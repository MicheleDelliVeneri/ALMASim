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


def _insert_shifted_kernel(
    image: np.ndarray,
    kernel: np.ndarray,
    amplitude: float,
    target_y: int,
    target_x: int,
    center_y: int,
    center_x: int,
) -> None:
    """Accumulate a shifted kernel into an image without wrap-around."""
    y0 = target_y - center_y
    x0 = target_x - center_x

    image_y0 = max(0, y0)
    image_x0 = max(0, x0)
    image_y1 = min(image.shape[0], y0 + kernel.shape[0])
    image_x1 = min(image.shape[1], x0 + kernel.shape[1])
    if image_y0 >= image_y1 or image_x0 >= image_x1:
        return

    kernel_y0 = image_y0 - y0
    kernel_x0 = image_x0 - x0
    kernel_y1 = kernel_y0 + (image_y1 - image_y0)
    kernel_x1 = kernel_x0 + (image_x1 - image_x0)

    image[image_y0:image_y1, image_x0:image_x1] += (
        amplitude * kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1]
    )


def _estimate_psf_padding(psf: np.ndarray, *, threshold_ratio: float = 1e-3) -> tuple[int, int]:
    """Estimate a finite PSF support radius for edge-safe CLEAN padding."""
    peak_y, peak_x = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
    peak = float(np.max(np.abs(psf)))
    if peak < 1e-12:
        return 0, 0

    support = np.abs(psf) >= peak * float(threshold_ratio)
    if not np.any(support):
        return 0, 0

    ys, xs = np.where(support)
    pad_y = max(int(peak_y - ys.min()), int(ys.max() - peak_y))
    pad_x = max(int(peak_x - xs.min()), int(xs.max() - peak_x))
    return pad_y, pad_x


def _pad_channel(channel: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Zero-pad a single 2D channel symmetrically."""
    if pad_y <= 0 and pad_x <= 0:
        return channel.copy()
    return np.pad(channel, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant")


def _crop_channel(channel: np.ndarray, pad_y: int, pad_x: int) -> np.ndarray:
    """Crop a padded 2D channel back to its original shape."""
    if pad_y <= 0 and pad_x <= 0:
        return channel.copy()

    y_slice = slice(pad_y, -pad_y if pad_y > 0 else None)
    x_slice = slice(pad_x, -pad_x if pad_x > 0 else None)
    return channel[y_slice, x_slice].copy()
