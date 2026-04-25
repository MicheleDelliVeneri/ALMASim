"""CLI example for iterative imaging/deconvolution on saved cube products.

This script demonstrates:
1. building a synthetic clean cube
2. convolving it into a dirty cube with a known beam
3. saving `clean-cube_*`, `dirty-cube_*`, and `beam-cube_*` products
4. running the iterative CLEAN-style deconvolution
5. checking that the restored cube is closer to the clean cube than the dirty cube
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from almasim.services.imaging import clean_deconvolve_cube


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "examples" / "output" / "imaging_demo",
        help="Directory used for the saved demo products.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=4,
        help="Number of spectral channels in the synthetic cube.",
    )
    parser.add_argument(
        "--n-pix",
        type=int,
        default=33,
        help="Spatial size of the synthetic cube.",
    )
    parser.add_argument(
        "--beam-sigma-pix",
        type=float,
        default=1.8,
        help="Gaussian sigma of the dirty beam in pixels.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=180,
        help="Number of iterative CLEAN cycles to run.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.12,
        help="Loop gain for the iterative CLEAN run.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Optional additive Gaussian noise std in the dirty cube.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the synthetic demo.",
    )
    parser.add_argument(
        "--require-improvement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail with a non-zero exit code if the restored cube does not "
            "improve on the dirty cube."
        ),
    )
    return parser


def gaussian_beam(shape: tuple[int, int], sigma_pix: float) -> np.ndarray:
    """Return a centered Gaussian PSF normalized to unit peak."""
    ny, nx = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    beam = np.exp(-0.5 * (((yy - cy) / sigma_pix) ** 2 + ((xx - cx) / sigma_pix) ** 2))
    beam /= np.max(beam)
    return beam.astype(np.float32)


def build_clean_cube(channels: int, n_pix: int) -> np.ndarray:
    """Build a simple multi-channel clean cube with compact sources."""
    cube = np.zeros((channels, n_pix, n_pix), dtype=np.float32)
    center = n_pix // 2

    for channel in range(channels):
        cube[channel, center, center] = 1.0 + 0.15 * channel
        cube[channel, center - 5, center + 4] = 0.35 + 0.05 * channel

    return cube


def convolve_cube_with_beam(
    clean_cube: np.ndarray,
    beam_cube: np.ndarray,
    *,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convolve each channel with the provided PSF and add optional Gaussian noise."""
    dirty_cube = np.zeros_like(clean_cube, dtype=np.float32)
    for channel in range(clean_cube.shape[0]):
        dirty_fft = np.fft.fft2(clean_cube[channel]) * np.fft.fft2(
            np.fft.ifftshift(beam_cube[channel])
        )
        dirty = np.real(np.fft.ifft2(dirty_fft)).astype(np.float32)
        if noise_std > 0.0:
            dirty = dirty + rng.normal(0.0, noise_std, size=dirty.shape).astype(np.float32)
        dirty_cube[channel] = dirty
    return dirty_cube


def save_demo_products(
    output_dir: Path,
    *,
    clean_cube: np.ndarray,
    dirty_cube: np.ndarray,
    beam_cube: np.ndarray,
    restored_cube: np.ndarray,
    residual_cube: np.ndarray,
) -> None:
    """Persist the same product names the imaging page expects."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "clean-cube_0.npz", clean_cube=clean_cube)
    np.savez(output_dir / "dirty-cube_0.npz", dirty_cube=dirty_cube)
    np.savez(output_dir / "beam-cube_0.npz", beam_cube=beam_cube)
    np.savez(output_dir / "deconvolved-clean-cube_0.npz", clean_cube=restored_cube)
    np.savez(output_dir / "residual-cube_0.npz", residual_cube=residual_cube)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    clean_cube = build_clean_cube(args.channels, args.n_pix)
    beam = gaussian_beam((args.n_pix, args.n_pix), args.beam_sigma_pix)
    beam_cube = np.repeat(beam[None, ...], args.channels, axis=0)
    dirty_cube = convolve_cube_with_beam(
        clean_cube,
        beam_cube,
        noise_std=args.noise_std,
        rng=rng,
    )

    result = clean_deconvolve_cube(
        dirty_cube,
        beam_cube,
        n_cycles=args.cycles,
        gain=args.gain,
    )
    restored_cube = result["clean_cube"]
    residual_cube = result["residual_cube"]

    dirty_mse = float(np.mean((dirty_cube - clean_cube) ** 2))
    restored_mse = float(np.mean((restored_cube - clean_cube) ** 2))
    improvement_ratio = dirty_mse / max(restored_mse, 1e-12)

    save_demo_products(
        args.output_dir,
        clean_cube=clean_cube,
        dirty_cube=dirty_cube,
        beam_cube=beam_cube,
        restored_cube=restored_cube,
        residual_cube=residual_cube,
    )

    print(f"Saved demo products to: {args.output_dir.resolve()}")
    print(f"Clean cube shape: {clean_cube.shape}")
    print(f"Dirty MSE vs clean:      {dirty_mse:.6e}")
    print(f"Restored MSE vs clean:   {restored_mse:.6e}")
    print(f"Improvement ratio:       {improvement_ratio:.3f}x")
    print(f"Cycles completed:        {result['cycles_completed']}")
    print(
        "Products: clean-cube_0.npz, dirty-cube_0.npz, beam-cube_0.npz, "
        "deconvolved-clean-cube_0.npz, residual-cube_0.npz"
    )

    if args.require_improvement and not restored_mse < dirty_mse:
        raise SystemExit(
            "Deconvolution did not improve the reconstruction: "
            f"dirty_mse={dirty_mse:.6e}, restored_mse={restored_mse:.6e}"
        )


if __name__ == "__main__":
    main()
