"""Deterministic image reconstruction and TP/INT merge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def wiener_deconvolve_cube(
    dirty_cube: np.ndarray,
    beam_cube: np.ndarray,
    *,
    epsilon: float = 1e-3,
    clip_negative: bool = True,
) -> np.ndarray:
    """Apply a simple deterministic Wiener-style deconvolution per channel."""
    dirty_cube = np.asarray(dirty_cube, dtype=np.float32)
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    if dirty_cube.shape != beam_cube.shape:
        raise ValueError("dirty_cube and beam_cube must have the same shape")

    recon = np.zeros_like(dirty_cube, dtype=np.float32)
    for channel in range(dirty_cube.shape[0]):
        psf = beam_cube[channel]
        psf = psf / max(float(np.max(np.abs(psf))), 1e-12)
        psf_fft = np.fft.fft2(np.fft.ifftshift(psf))
        dirty_fft = np.fft.fft2(dirty_cube[channel])
        denom = np.abs(psf_fft) ** 2 + float(epsilon)
        recon_channel = np.real(np.fft.ifft2(np.conj(psf_fft) * dirty_fft / denom))
        if clip_negative:
            recon_channel = np.clip(recon_channel, 0.0, None)
        recon[channel] = recon_channel.astype(np.float32)
    return recon


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


def load_cube_from_npz(npz_path: str | Path) -> tuple[np.ndarray, str]:
    """Load a cube from an NPZ file and normalize to `(chan, y, x)` layout."""
    with np.load(npz_path) as data:
        cube = None
        cube_name = None
        for name in [
            "modelCube",
            "dirtyCube",
            "clean_cube",
            "dirty_cube",
            "beam_cube",
            "component_cube",
            "residual_cube",
            "restored_cube",
            "clean_beam_cube",
            "convolved_reference_cube",
        ]:
            if name in data:
                cube = data[name]
                cube_name = name
                break
        if cube is None:
            keys = list(data.keys())
            if not keys:
                raise ValueError("No arrays found in NPZ file")
            cube_name = keys[0]
            cube = data[cube_name]

    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D datacube, got {cube.ndim}D array")
    if cube.shape[2] <= min(cube.shape[0], cube.shape[1]):
        cube = np.transpose(cube, (2, 0, 1))
    return cube, str(cube_name)


def integrate_cube_preview(
    cube: np.ndarray,
    *,
    method: str = "sum",
    cube_name: str = "cube",
) -> dict[str, Any]:
    """Convert a channel-first cube into a normalized 2D preview payload."""
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D datacube, got {cube.ndim}D array")
    if method not in {"sum", "mean"}:
        raise ValueError("method must be 'sum' or 'mean'")

    if method == "sum":
        integrated = np.sum(cube, axis=0)
    else:
        integrated = np.nanmean(cube, axis=0)

    integrated = np.nan_to_num(integrated, nan=0.0, posinf=0.0, neginf=0.0)
    if integrated.max() > integrated.min():
        normalized = (
            (integrated - integrated.min()) / (integrated.max() - integrated.min()) * 255.0
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(integrated, dtype=np.uint8)

    return {
        "image": normalized.tolist(),
        "stats": {
            "shape": list(cube.shape),
            "integrated_shape": list(integrated.shape),
            "min": float(integrated.min()),
            "max": float(integrated.max()),
            "mean": float(integrated.mean()),
            "std": float(integrated.std()),
            "cube_name": cube_name,
        },
        "method": method,
    }


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


def _build_clean_beam(psf: np.ndarray) -> np.ndarray:
    """Approximate the central clean beam as a Gaussian from PSF moments."""
    peak_y, peak_x = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
    peak = float(psf[peak_y, peak_x])
    if abs(peak) < 1e-12:
        clean_beam = np.zeros_like(psf, dtype=np.float32)
        clean_beam[peak_y, peak_x] = 1.0
        return clean_beam

    normalized = np.asarray(psf, dtype=np.float32) / peak
    positive = np.clip(normalized, 0.0, None)

    window_radius_y = max(2, psf.shape[0] // 10)
    window_radius_x = max(2, psf.shape[1] // 10)
    y0 = max(0, peak_y - window_radius_y)
    y1 = min(psf.shape[0], peak_y + window_radius_y + 1)
    x0 = max(0, peak_x - window_radius_x)
    x1 = min(psf.shape[1], peak_x + window_radius_x + 1)
    cropped = positive[y0:y1, x0:x1]

    yy, xx = np.indices(cropped.shape, dtype=np.float32)
    yy += float(y0)
    xx += float(x0)
    weights = cropped / max(float(np.sum(cropped)), 1e-12)
    sigma_y = float(np.sqrt(np.sum(weights * (yy - peak_y) ** 2)))
    sigma_x = float(np.sqrt(np.sum(weights * (xx - peak_x) ** 2)))
    sigma_y = max(sigma_y, 1.0)
    sigma_x = max(sigma_x, 1.0)

    full_y, full_x = np.indices(psf.shape, dtype=np.float32)
    clean_beam = np.exp(
        -0.5 * (((full_y - peak_y) / sigma_y) ** 2 + ((full_x - peak_x) / sigma_x) ** 2)
    )
    clean_beam /= max(float(np.max(clean_beam)), 1e-12)
    return clean_beam.astype(np.float32)


def clean_deconvolve_cube(
    dirty_cube: np.ndarray,
    beam_cube: np.ndarray,
    *,
    n_cycles: int = 100,
    gain: float = 0.1,
    threshold: float | None = None,
    clip_negative: bool = False,
    initial_component_cube: np.ndarray | None = None,
    initial_residual_cube: np.ndarray | None = None,
    initial_clean_beam_cube: np.ndarray | None = None,
    initial_cycles_completed: int = 0,
) -> dict[str, Any]:
    """Run a simple Hogbom-style CLEAN loop per channel."""
    dirty_cube = np.asarray(dirty_cube, dtype=np.float32)
    beam_cube = np.asarray(beam_cube, dtype=np.float32)
    if dirty_cube.shape != beam_cube.shape:
        raise ValueError("dirty_cube and beam_cube must have the same shape")
    if dirty_cube.ndim != 3:
        raise ValueError("dirty_cube and beam_cube must have shape (n_channels, ny, nx)")
    if n_cycles < 0:
        raise ValueError("n_cycles must be non-negative")
    if not 0.0 < gain <= 1.0:
        raise ValueError("gain must be in the interval (0, 1]")

    component_cube = (
        np.zeros_like(dirty_cube, dtype=np.float32)
        if initial_component_cube is None
        else np.asarray(initial_component_cube, dtype=np.float32).copy()
    )
    residual_cube = (
        dirty_cube.copy()
        if initial_residual_cube is None
        else np.asarray(initial_residual_cube, dtype=np.float32).copy()
    )
    clean_beam_cube = (
        np.zeros_like(beam_cube, dtype=np.float32)
        if initial_clean_beam_cube is None
        else np.asarray(initial_clean_beam_cube, dtype=np.float32).copy()
    )
    if component_cube.shape != dirty_cube.shape:
        raise ValueError("initial_component_cube must match dirty_cube shape")
    if residual_cube.shape != dirty_cube.shape:
        raise ValueError("initial_residual_cube must match dirty_cube shape")
    if clean_beam_cube.shape != beam_cube.shape:
        raise ValueError("initial_clean_beam_cube must match beam_cube shape")

    restored_cube = np.zeros_like(dirty_cube, dtype=np.float32)
    cycles_added = 0

    for channel in range(dirty_cube.shape[0]):
        psf = np.asarray(beam_cube[channel], dtype=np.float32)
        pad_y, pad_x = _estimate_psf_padding(psf)
        psf = _pad_channel(psf, pad_y, pad_x)
        peak_y, peak_x = np.unravel_index(int(np.argmax(np.abs(psf))), psf.shape)
        psf_peak = float(psf[peak_y, peak_x])
        if abs(psf_peak) < 1e-12:
            psf_normalized = np.zeros_like(psf, dtype=np.float32)
            psf_normalized[peak_y, peak_x] = 1.0
        else:
            psf_normalized = (psf / psf_peak).astype(np.float32)

        existing_clean_beam = np.asarray(clean_beam_cube[channel], dtype=np.float32)
        if np.max(np.abs(existing_clean_beam)) > 0.0:
            clean_beam = _pad_channel(existing_clean_beam, pad_y, pad_x)
        else:
            clean_beam = _build_clean_beam(psf_normalized)

        residual = _pad_channel(residual_cube[channel], pad_y, pad_x)
        components = _pad_channel(component_cube[channel], pad_y, pad_x)
        channel_cycles = 0

        for _ in range(int(n_cycles)):
            peak_index = np.unravel_index(int(np.argmax(np.abs(residual))), residual.shape)
            peak_value = float(residual[peak_index])
            if threshold is not None and abs(peak_value) < float(threshold):
                break
            component_amplitude = gain * peak_value
            components[peak_index] += component_amplitude
            _insert_shifted_kernel(
                residual,
                psf_normalized,
                -component_amplitude,
                peak_index[0],
                peak_index[1],
                peak_y,
                peak_x,
            )
            channel_cycles += 1

        restored = np.real(
            np.fft.ifft2(np.fft.fft2(components) * np.fft.fft2(np.fft.ifftshift(clean_beam)))
        ).astype(np.float32)
        restored += residual
        if clip_negative:
            restored = np.clip(restored, 0.0, None)
        component_cube[channel] = _crop_channel(components, pad_y, pad_x)
        residual_cube[channel] = _crop_channel(residual, pad_y, pad_x)
        clean_beam_cube[channel] = _crop_channel(clean_beam, pad_y, pad_x)
        restored_cube[channel] = _crop_channel(restored, pad_y, pad_x)
        cycles_added = max(cycles_added, channel_cycles)

    return {
        "clean_cube": restored_cube,
        "restored_cube": restored_cube,
        "component_cube": component_cube,
        "residual_cube": residual_cube,
        "clean_beam_cube": clean_beam_cube,
        "cycles_completed": int(initial_cycles_completed) + cycles_added,
        "cycles_added": cycles_added,
        "gain": float(gain),
        "threshold": None if threshold is None else float(threshold),
    }


def match_tp_to_int_flux_scale(
    tp_cube: np.ndarray,
    int_cube: np.ndarray,
    beam_cube: np.ndarray | None = None,
) -> float:
    """Estimate a scalar TP-to-INT flux scale from overlapping weighted flux."""
    tp_cube = np.asarray(tp_cube, dtype=np.float32)
    int_cube = np.asarray(int_cube, dtype=np.float32)
    if tp_cube.shape != int_cube.shape:
        tp_cube = regrid_cube_to_match(tp_cube, int_cube.shape)

    if beam_cube is None:
        weights = np.ones_like(int_cube, dtype=np.float32)
    else:
        beam_cube = np.asarray(beam_cube, dtype=np.float32)
        weights = np.clip(beam_cube, 0.0, None)
        if weights.shape != int_cube.shape:
            weights = regrid_cube_to_match(weights, int_cube.shape)

    tp_flux = float(np.sum(weights * tp_cube))
    int_flux = float(np.sum(weights * int_cube))
    if abs(tp_flux) < 1e-12:
        return 1.0
    return int_flux / tp_flux


def feather_merge_cube(
    int_cube: np.ndarray,
    tp_cube: np.ndarray,
    tp_beam_cube: np.ndarray,
    *,
    int_primary_beam: np.ndarray | None = None,
    flux_scale: float | None = None,
) -> np.ndarray:
    """Merge TP and INT image cubes using a beam-shaped Fourier low-pass weight."""
    int_cube = np.asarray(int_cube, dtype=np.float32)
    tp_cube = np.asarray(tp_cube, dtype=np.float32)
    tp_beam_cube = np.asarray(tp_beam_cube, dtype=np.float32)

    if tp_cube.shape != int_cube.shape:
        tp_cube = regrid_cube_to_match(tp_cube, int_cube.shape)
    if tp_beam_cube.shape != int_cube.shape:
        tp_beam_cube = regrid_cube_to_match(tp_beam_cube, int_cube.shape)
    if int_primary_beam is not None:
        int_primary_beam = np.asarray(int_primary_beam, dtype=np.float32)
        if int_primary_beam.shape != int_cube.shape:
            int_primary_beam = regrid_cube_to_match(int_primary_beam, int_cube.shape)

    if flux_scale is None:
        flux_scale = match_tp_to_int_flux_scale(tp_cube, int_cube, int_primary_beam)

    merged = np.zeros_like(int_cube, dtype=np.float32)
    for channel in range(int_cube.shape[0]):
        tp_fft = np.fft.fft2(tp_cube[channel] * float(flux_scale))
        int_fft = np.fft.fft2(int_cube[channel])
        low_pass = np.abs(np.fft.fft2(np.fft.ifftshift(tp_beam_cube[channel])))
        low_pass /= max(float(np.max(low_pass)), 1e-12)
        low_pass = np.clip(low_pass, 0.0, 1.0)
        merged[channel] = np.real(
            np.fft.ifft2(low_pass * tp_fft + (1.0 - low_pass) * int_fft)
        ).astype(np.float32)
    return merged


def build_image_products(
    *,
    int_results: dict[str, Any] | None,
    tp_results: dict[str, Any] | None,
    reconstruction_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Build reproducible image-domain products from measurement-stage outputs."""
    products: dict[str, Any] = {
        "int_image_cube": None,
        "tp_image_cube": None,
        "tp_int_image_cube": None,
        "per_config_image_results": [],
    }

    if int_results is not None and "dirty_cube" in int_results and "beam_cube" in int_results:
        products["int_image_cube"] = wiener_deconvolve_cube(
            int_results["dirty_cube"],
            int_results["beam_cube"],
            epsilon=reconstruction_epsilon,
        )
        per_config_results = int_results.get("per_config_results", [])
        products["per_config_image_results"] = [
            {
                "config_name": config_result.get("config_name"),
                "array_type": config_result.get("array_type"),
                "image_cube": wiener_deconvolve_cube(
                    config_result["dirty_cube"],
                    config_result["beam_cube"],
                    epsilon=reconstruction_epsilon,
                ),
            }
            for config_result in per_config_results
            if "dirty_cube" in config_result and "beam_cube" in config_result
        ]

    if tp_results is not None:
        products["tp_image_cube"] = tp_results["tp_dirty_cube"]

    if products["int_image_cube"] is not None and products["tp_image_cube"] is not None:
        products["tp_int_image_cube"] = feather_merge_cube(
            products["int_image_cube"],
            products["tp_image_cube"],
            tp_results["tp_beam_cube"],
            int_primary_beam=int_results.get("beam_cube") if int_results else None,
        )
    elif products["int_image_cube"] is not None:
        products["tp_int_image_cube"] = products["int_image_cube"]
    elif products["tp_image_cube"] is not None:
        products["tp_int_image_cube"] = products["tp_image_cube"]

    return products
