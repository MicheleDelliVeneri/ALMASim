"""Previews that need computing, as FITS for the console's viewer (web console plan, U3.7).

``preview integrate`` sums or averages a cube along an axis (a moment-0
style map of a line cube, or the dirty image of a MeasurementSet gridded on
the fly); ``preview wiener`` reconstructs a simulated observation from its
dirty and beam cubes. The console runs them as short Slurm jobs and opens
what they write.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import typer
from astropy.io import fits

from ._shared import fail

preview_app = typer.Typer(
    help="Previews of products, written as FITS (moment-style maps, Wiener reconstructions).",
    no_args_is_help=True,
)


class Integration(StrEnum):
    sum = "sum"
    mean = "mean"


class ComplexPart(StrEnum):
    real = "real"
    imag = "imag"
    magnitude = "magnitude"
    phase = "phase"


def _write(path: Path, image: np.ndarray, **cards: object) -> None:
    from almasim.services.imaging.fits_io import write_fits_atomic

    header = fits.Header()
    for key, value in cards.items():
        if value is not None:
            header[key[:8].upper()] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    write_fits_atomic(path, [fits.PrimaryHDU(np.asarray(image, dtype=np.float32), header)])


@preview_app.command("integrate")
def integrate(
    product: Path = typer.Argument(
        ..., help="A cube to integrate: FITS, NPZ, or a MeasurementSet (gridded first)."
    ),
    output: Path = typer.Argument(..., help="The 2D FITS image to write."),
    method: Integration = typer.Option(
        Integration.sum, "--method", help="Sum (moment-0 style) or mean along the axis."
    ),
    channel_start: int | None = typer.Option(
        None, "--channel-start", min=0, help="First channel to include (default: all)."
    ),
    channel_end: int | None = typer.Option(
        None, "--channel-end", min=0, help="Last channel to include (default: all)."
    ),
    complex_part: ComplexPart = typer.Option(
        ComplexPart.real, "--complex-part", help="For visibility cubes: the part to show."
    ),
) -> None:
    """Integrate a cube along its spectral axis into one image."""
    from almasim.services.products.preview import integrate_product

    if not product.exists():
        fail(f"{product} does not exist")
    try:
        image, stats = integrate_product(
            product,
            method.value,
            complex_component=complex_part.value,
            channel_start=channel_start,
            channel_end=channel_end,
        )
    except (ValueError, OSError) as exc:
        fail(f"cannot integrate {product}: {exc}")
    _write(
        output,
        image,
        ORIGIN="almasim preview integrate",
        SOURCE=product.name[:68],
        METHOD=method.value,
        CHANSTRT=stats["channel_start"],
        CHANEND=stats["channel_end"],
    )
    typer.echo(
        f"{output}: {image.shape[1]} x {image.shape[0]} from {stats['format']} "
        f"{stats['cube_name']} (channels {stats['channel_start']}-{stats['channel_end']}, "
        f"{method.value}; min {stats['min']:.4g}, max {stats['max']:.4g})"
    )


@preview_app.command("wiener")
def wiener(
    dirty: Path = typer.Argument(..., help="The dirty cube (FITS or NPZ)."),
    beam: Path = typer.Argument(..., help="The beam (PSF) cube, the dirty cube's shape."),
    output_dir: Path = typer.Argument(..., help="Folder for the model, restored and residual."),
    epsilon: float = typer.Option(
        1e-3, "--epsilon", min=1e-12, help="Wiener regularisation (noise-to-signal power)."
    ),
) -> None:
    """Reconstruct a simulated observation from its dirty and beam cubes (Wiener filter).

    Writes wiener-model.fits, wiener-restored.fits (the model convolved with a
    clean beam fitted to each channel's PSF, plus the residual) and
    wiener-residual.fits. A quick look, not the reference imaging: that is
    WSClean on the simulated MeasurementSet.
    """
    from almasim.services.imaging.reconstruction.deconvolution import wiener_preview
    from almasim.services.products.preview import load_preview_cube

    for path in (dirty, beam):
        if not path.exists():
            fail(f"{path} does not exist")
    try:
        dirty_cube, _, _ = load_preview_cube(dirty)
        beam_cube, _, _ = load_preview_cube(beam)
        result = wiener_preview(np.real(dirty_cube), np.real(beam_cube), epsilon=epsilon)
    except (ValueError, OSError) as exc:
        fail(f"cannot reconstruct: {exc}")
    for name in ("model", "restored", "residual"):
        _write(
            output_dir / f"wiener-{name}.fits",
            result[name],
            ORIGIN="almasim preview wiener",
            SOURCE=dirty.name[:68],
            EPSILON=epsilon,
        )
    typer.echo(f"{output_dir}: wiener-model, -restored and -residual ({dirty_cube.shape})")
