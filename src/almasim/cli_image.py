"""Batch imaging commands for the ALMASim CLI."""

from __future__ import annotations

import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from scipy.constants import speed_of_light
from tqdm import tqdm

ALMA_FOV_FACTOR = 1.22  # Standard primary-beam/FOV approximation: 1.22 * λ / D
RAD_TO_ARCSEC = 180 / np.pi * 3600
MIN_IMAGE_PIXELS = 16


def import_casacore_tables() -> Any:
    try:
        from casacore.tables import table

        return table
    except ImportError:
        typer.echo(
            "Missing optional dependency `python-casacore` required to read metadata from "
            "the measurement set. Install it with the `ms-casacore` extra.",
            err=True,
        )
        raise typer.Exit(code=1)


image_app = typer.Typer(
    help="Data product batch imaging.",
    no_args_is_help=True,
)


def compute_imaging_parameters(input_ms: Path) -> pd.DataFrame:
    casacore_table = import_casacore_tables()
    spectral_windows = casacore_table(f"{input_ms}::SPECTRAL_WINDOW", ack=False)
    antennas = casacore_table(f"{input_ms}::ANTENNA", ack=False)
    reference_frequencies = spectral_windows.getcol("REF_FREQUENCY")
    min_dish_diameter = np.min(antennas.getcol("DISH_DIAMETER"))
    antenna_pos = antennas.getcol("POSITION")
    i, j = np.triu_indices(antenna_pos.shape[0], k=1)
    distance = np.linalg.norm(antenna_pos[j, :] - antenna_pos[i, :], axis=1)
    max_baseline_size = max(distance)

    fov_per_frequency = (
        ALMA_FOV_FACTOR * speed_of_light * RAD_TO_ARCSEC / reference_frequencies / min_dish_diameter
    )
    synthetized_beam_size = (
        speed_of_light * RAD_TO_ARCSEC / reference_frequencies / max_baseline_size
    )
    spectral_window_id = np.arange(reference_frequencies.size, dtype=int)

    derived_parameters = pd.DataFrame(
        {
            "filename": [str(input_ms.resolve())] * reference_frequencies.size,
            "spectral_window_id": spectral_window_id,
            "reference_frequency": reference_frequencies,
            "fov_per_frequency": fov_per_frequency,
            "max_baseline_size": [max_baseline_size] * reference_frequencies.size,
            "synthetized_beam_size": synthetized_beam_size,
        }
    )
    return derived_parameters


def imaging_parameter_to_command_arg(
    imaging_parameters: pd.Series, fov_fraction: float, beam_sampling: float
) -> list[str]:
    spw = imaging_parameters["spectral_window_id"]
    fov = imaging_parameters["fov_per_frequency"]
    synthetized_beam_size = imaging_parameters["synthetized_beam_size"]
    synthetized_beam_size /= beam_sampling
    fov *= fov_fraction
    n_pixels = max(MIN_IMAGE_PIXELS, int(math.ceil(fov / synthetized_beam_size)))
    cmd_args = [
        "-scale",
        f"{synthetized_beam_size}asec",
        "-size",
        str(n_pixels),
        str(n_pixels),
        "-spws",
        str(spw),
        "-mgain",
        "0.85",
        "-niter",
        "100000",
        "-pol",
        "I",
        "-make-psf",
        "-weight",
        "briggs",
        "0.5",
        "-update-model-required",
    ]
    return cmd_args


@image_app.command("ms-overview")
def derive_parameters(
    input_ms: Path = typer.Argument(
        ...,
        help="Source of the measurement set",
    ),
):

    derived_parameters = compute_imaging_parameters(input_ms)

    typer.echo(derived_parameters.to_string(index=False))


@image_app.command("compute-parameters")
def compute_parameters(
    archive_folder: Path = typer.Argument(..., help="Processed MSs folder"),
    output_metadata_file: Path = typer.Argument(..., help="Parameters csv file"),
):
    mss = list(archive_folder.glob("*.cal"))
    if len(mss) == 0:
        typer.echo(f"Cannot find any MS in {archive_folder}")
        raise typer.Exit(code=1)

    main_output = compute_imaging_parameters(mss[0])
    for ms in tqdm(mss[1:]):
        single_ms_output = compute_imaging_parameters(ms)
        main_output = pd.concat([main_output, single_ms_output], axis=0, ignore_index=True)
    main_output.to_csv(output_metadata_file, index=False)


@image_app.command("batch-image")
def image_set(
    imaging_parameters: Path = typer.Argument(help="Imaging parameter file"),
    output_directory: Path = typer.Argument(help="Output directory path"),
    fov_fraction: float = typer.Option(
        help="Fraction of the FOV to image",
        default=1.5,
        min=1e-6,
    ),
    beam_sampling: float = typer.Option(
        help="Number of pixels to use to sample the synthetized beam. Could be fractional.",
        default=8,
        min=1e-6,
    ),
    num_cores: int = typer.Option(help="Number of cores per imaging task", default=10, min=1),
    max_cores_per_node: int = typer.Option(
        help="Number of cores per node. [Used to scale the memory usage of wsclean]",
        default=95,
        min=1,
    ),
):

    parameters = pd.read_csv(str(imaging_parameters))
    for _, dset_parameter in tqdm(parameters.iterrows(), total=len(parameters)):
        input_filename = Path(dset_parameter["filename"])
        command_args = imaging_parameter_to_command_arg(dset_parameter, fov_fraction, beam_sampling)

        outdir = (
            output_directory / input_filename.stem / f"SPW-{dset_parameter['spectral_window_id']}"
        )
        outdir.mkdir(exist_ok=True, parents=True)
        mem_fraction = min(1.0, num_cores / max_cores_per_node)

        wsclean_cmd = [
            "wsclean",
            "-name",
            str(outdir / "wsclean"),
            *command_args,
            "-mem",
            str(mem_fraction),
            str(input_filename),
        ]
        wrap_text = shlex.join(wsclean_cmd)
        sbatch_cmd = [
            "sbatch",
            "-J",
            input_filename.stem + f"_{dset_parameter['spectral_window_id']}",
            "--wrap",
            wrap_text,
            "-o",
            str(output_directory.absolute() / r"%x-std%j.out"),
            "-e",
            str(output_directory.absolute() / r"%x-std%j.err"),
            "-c",
            str(num_cores),
        ]
        subprocess.run(sbatch_cmd, check=True)
