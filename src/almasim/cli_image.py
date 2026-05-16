"""Batch imaging commands for the ALMASim CLI."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

ALMA_FOV_FACTOR = 1.12  # Standard primary-beam/FOV approximation: 1.12 * λ / D
RAD_TO_ARCSEC = 180 / np.pi * 3600
MIN_IMAGE_PIXELS = 16
SPEED_OF_LIGHT_M_S = 299_792_458.0


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


def _run_commands_with_slurm_cluster(
    commands: list[tuple[str, list[str]]],
    *,
    cores_per_task: int,
    node_cores: int,
    queue: str,
    project: str | None,
    walltime: str,
    memory: str,
    n_jobs: int,
    scheduler_host: str | None,
    scheduler_interface: str | None,
    task_timeout: float | None,
) -> None:
    from almasim.services.compute import create_backend

    if not commands:
        return

    with create_backend(
        "slurm",
        queue=queue,
        node_cores=node_cores,
        memory=memory,
        walltime=walltime,
        n_workers=n_jobs,
        project=project,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
    ) as backend:
        futures: list[tuple[str, Any]] = []
        for label, cmd in commands:
            future = backend.submit_subcommand(
                command=cmd,
                cores=cores_per_task,
                timeout=task_timeout,
            )
            futures.append((label, future))

        for label, future in tqdm(futures, total=len(futures), desc="SLURM tasks"):
            result = future.result()
            if result.returncode != 0:
                typer.echo(f"Task failed: {label}", err=True)
                typer.echo(result.stderr.rstrip(), err=True)
                raise typer.Exit(code=result.returncode)


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
        ALMA_FOV_FACTOR
        * SPEED_OF_LIGHT_M_S
        * RAD_TO_ARCSEC
        / reference_frequencies
        / min_dish_diameter
    )
    synthetized_beam_size = (
        SPEED_OF_LIGHT_M_S * RAD_TO_ARCSEC / reference_frequencies / max_baseline_size
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
    slurm_queue: str = typer.Option(default="normal", help="SLURM queue/partition"),
    slurm_project: str | None = typer.Option(default=None, help="SLURM project/account"),
    slurm_walltime: str = typer.Option(default="02:00:00", help="SLURM walltime HH:MM:SS"),
    slurm_memory: str = typer.Option(default="16GB", help="SLURM memory per worker"),
    slurm_n_jobs: int = typer.Option(default=1, min=1, help="Number of SLURM workers/jobs"),
    scheduler_host: str | None = typer.Option(
        default=None,
        help="Scheduler host advertised to workers (defaults to submit HOSTNAME)",
    ),
    scheduler_interface: str | None = typer.Option(
        default=None,
        help="Scheduler/worker network interface (e.g. ib0, eth0)",
    ),
    task_timeout: float = typer.Option(
        default=3600,
        min=1,
        help="Timeout in seconds for each worker-side command",
    ),
):

    parameters = pd.read_csv(str(imaging_parameters))
    commands: list[tuple[str, list[str]]] = []
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
        label = f"{input_filename.stem}_{dset_parameter['spectral_window_id']}"
        commands.append((label, wsclean_cmd))

    _run_commands_with_slurm_cluster(
        commands,
        cores_per_task=num_cores,
        node_cores=max_cores_per_node,
        queue=slurm_queue,
        project=slurm_project,
        walltime=slurm_walltime,
        memory=slurm_memory,
        n_jobs=slurm_n_jobs,
        scheduler_host=scheduler_host,
        scheduler_interface=scheduler_interface,
        task_timeout=task_timeout,
    )


@image_app.command("predict-single")
def predict_single(
    input_ms: Path = typer.Argument(..., help="Input MS"),
    model: Path = typer.Argument(..., help="FITS model path"),
    output_ms: Path = typer.Argument(..., help="Output MS"),
):
    predict_from_model(input_ms=input_ms, model=model, output_ms=output_ms)


def predict_from_model(input_ms: Path, model: Path, output_ms: Path) -> None:
    from astropy.io import fits
    from ducc0.wgridder import dirty2vis

    typer.echo(f"Predicting visibilities from model: {model}")
    typer.echo(f"Copying measurement set: {input_ms} -> {output_ms}")

    if output_ms.exists():
        typer.echo(f"Output MS already exists: {output_ms}", err=True)
        raise typer.Exit(code=1)

    casacore_table = import_casacore_tables()
    shutil.copytree(input_ms, output_ms)
    main_table = casacore_table(str(output_ms), readonly=False)
    try:
        nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK") or (os.cpu_count() or 1))
        uvw = main_table.getcol("UVW")  # (nrows, 3) metres
        with fits.open(model) as hdul:
            main_hdu = cast(fits.PrimaryHDU, hdul[0])
            header = main_hdu.header

            # Pixel sizes: FITS CDELT is in degrees, ducc0 expects radians
            deg2rad = np.pi / 180.0
            pixsize_x = abs(cast(float, header["CDELT1"])) * deg2rad
            pixsize_y = abs(cast(float, header["CDELT2"])) * deg2rad

            # Channel frequencies from the FITS WCS axis 3 (CRVAL3/CDELT3/CRPIX3/NAXIS3)
            nchans = int(cast(int, header["NAXIS3"]))
            crval3 = cast(float, header["CRVAL3"])  # reference frequency [Hz]
            cdelt3 = cast(float, header["CDELT3"])  # frequency increment [Hz]
            crpix3 = cast(float, header["CRPIX3"])  # reference pixel (1-based)
            freq = (crval3 + (np.arange(nchans) - (crpix3 - 1)) * cdelt3).astype(np.float64)
            typer.echo(f"Computed frequency grid from FITS header: nchans={nchans}")

            # Model image: FITS axes are [stokes, freq, y, x]; take first stokes & channel
            assert main_hdu.data is not None, "FITS primary HDU contains no image data"
            dirty = np.ascontiguousarray(main_hdu.data[0, 0, :, :].astype(np.float64))
            typer.echo(f"Input arrays: uvw={uvw.shape}, dirty={dirty.shape}, freq={freq.shape}")

            # Predict visibilities from the model image → shape (nrows, nchan)
            model_vis = dirty2vis(
                uvw=uvw.astype(np.float64),
                dirty=dirty,
                freq=freq,
                pixsize_x=pixsize_x,
                pixsize_y=pixsize_y,
                do_wgridding=True,
                epsilon=1e-4,
                nthreads=nthreads,
            )
            typer.echo(f"Predicted visibilities with shape: {model_vis.shape}")

            # Write predicted visibilities to MODEL_DATA (nrows, 4, nchan): XX, XY, YX, YY
            zeros = np.zeros_like(model_vis)
            half = model_vis / 2
            model_vis_col = np.stack(
                [half, zeros, zeros, half],
                axis=1,
            )  # XX=I/2, XY=0, YX=0, YY=I/2
            main_table.putcol("MODEL_DATA", model_vis_col)
    finally:
        main_table.close()

    typer.echo(f"Wrote MODEL_DATA to: {output_ms}")


@image_app.command("predict-batch")
def predict_batch(
    imaging_parameters: Path = typer.Argument(..., help="Imaging parameter file"),
    output_directory: Path = typer.Argument(..., help="Output directory path"),
    num_cores: int = typer.Option(help="Number of cores per predict task", default=1, min=1),
    use_slurm: bool = typer.Option(help="Whether or not to use slurm or not", default=True),
    max_cores_per_node: int = typer.Option(
        help="Number of cores per node for SLURM worker resource accounting",
        default=95,
        min=1,
    ),
    slurm_queue: str = typer.Option(default="normal", help="SLURM queue/partition"),
    slurm_project: str | None = typer.Option(default=None, help="SLURM project/account"),
    slurm_walltime: str = typer.Option(default="02:00:00", help="SLURM walltime HH:MM:SS"),
    slurm_memory: str = typer.Option(default="16GB", help="SLURM memory per worker"),
    slurm_n_jobs: int = typer.Option(default=1, min=1, help="Number of SLURM workers/jobs"),
    scheduler_host: str | None = typer.Option(
        default=None,
        help="Scheduler host advertised to workers (defaults to submit HOSTNAME)",
    ),
    scheduler_interface: str | None = typer.Option(
        default=None,
        help="Scheduler/worker network interface (e.g. ib0, eth0)",
    ),
    task_timeout: float = typer.Option(
        default=3600,
        min=1,
        help="Timeout in seconds for each worker-side command",
    ),
):
    parameters = pd.read_csv(str(imaging_parameters))
    slurm_commands: list[tuple[str, list[str]]] = []
    for _, dset_parameter in tqdm(parameters.iterrows(), total=len(parameters)):
        input_filename = Path(dset_parameter["filename"])
        spw_id = int(dset_parameter["spectral_window_id"])
        model_path = output_directory / input_filename.stem / f"SPW-{spw_id}" / "wsclean-model.fits"
        if not model_path.exists():
            typer.echo(
                "[debug] missing model FITS, skipping row:"
                "ms={input_filename} spw={spw_id} path={model_path}",
                err=False,
            )
            continue

        output_ms = model_path.parent / f"{input_filename.name}.predicted"
        predict_cmd = [
            "almasim",
            "image",
            "predict-single",
            str(input_filename.absolute()),
            str(model_path),
            str(output_ms),
        ]
        if use_slurm:
            label = input_filename.stem + f"_{spw_id}_predict"
            slurm_commands.append((label, predict_cmd))

        else:
            cmd = predict_cmd
            subprocess.run(cmd, check=True)

    if use_slurm:
        _run_commands_with_slurm_cluster(
            slurm_commands,
            cores_per_task=num_cores,
            node_cores=max_cores_per_node,
            queue=slurm_queue,
            project=slurm_project,
            walltime=slurm_walltime,
            memory=slurm_memory,
            n_jobs=slurm_n_jobs,
            scheduler_host=scheduler_host,
            scheduler_interface=scheduler_interface,
            task_timeout=task_timeout,
        )
