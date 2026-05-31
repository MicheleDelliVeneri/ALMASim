"""Prediction commands for the ALMASim CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli_image import _iter_ms_inputs, _predict_all_models_for_ms

predict_app = typer.Typer(
    help="Model-to-visibility prediction commands.",
    no_args_is_help=True,
)


@predict_app.command("ms-from-image")
def ms_from_image(
    input_ms_or_folder: Path = typer.Argument(help="Single MS directory or folder of MSs"),
    output_directory: Path = typer.Argument(help="Directory containing model FITS products"),
    use_slurm: bool = typer.Option(help="Whether or not to use slurm or not", default=True),
    num_cores: int = typer.Option(help="Number of cores per predict task", default=1, min=1),
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
) -> None:
    """Predict visibilities from one MS or a folder of MSs."""
    ms_inputs = _iter_ms_inputs(input_ms_or_folder)

    if use_slurm:
        from .cli_image import _run_commands_with_slurm_cluster

        commands: list[tuple[str, list[str]]] = []
        for input_ms in ms_inputs:
            label = f"{input_ms.stem}_predict"
            commands.append(
                (
                    label,
                    [
                        "almasim",
                        "predict",
                        "ms-from-image",
                        str(input_ms),
                        str(output_directory),
                        "--no-use-slurm",
                    ],
                )
            )

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
        return

    for input_ms in ms_inputs:
        _predict_all_models_for_ms(input_ms, output_directory)
