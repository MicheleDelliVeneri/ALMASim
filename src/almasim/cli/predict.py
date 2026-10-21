"""Prediction commands for the ALMASim CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from ._options import (
    MaxCoresPerNode,
    SchedulerHost,
    SchedulerInterface,
    SlurmMemory,
    SlurmProject,
    SlurmQueue,
    SlurmWalltime,
    TaskTimeout,
)
from .image import _iter_ms_inputs, _predict_all_models_for_ms

predict_app = typer.Typer(
    help="Model-to-visibility prediction commands.",
    no_args_is_help=True,
)


@predict_app.command("ms-from-image")
def ms_from_image(
    input_ms_or_folder: Path = typer.Argument(help="Single MS directory or folder of MSs"),
    output_directory: Path = typer.Argument(help="Directory containing model FITS products"),
    use_slurm: bool = typer.Option(
        help="Whether or not to use slurm or not", default=True, rich_help_panel="Cluster"
    ),
    num_cores: int = typer.Option(
        help="Number of cores per predict task", default=1, min=1, rich_help_panel="Cluster"
    ),
    max_cores_per_node: MaxCoresPerNode = 95,
    slurm_queue: SlurmQueue = "normal",
    slurm_project: SlurmProject = None,
    slurm_walltime: SlurmWalltime = "02:00:00",
    slurm_memory: SlurmMemory = "16GB",
    slurm_n_jobs: int = typer.Option(
        1,
        "--slurm-workers",
        "--slurm-n-jobs",
        min=1,
        help="Number of Slurm workers (jobs).",
        rich_help_panel="Cluster",
    ),
    scheduler_host: SchedulerHost = None,
    scheduler_interface: SchedulerInterface = None,
    task_timeout: TaskTimeout = 3600,
) -> None:
    """Predict visibilities from one MS or a folder of MSs."""
    ms_inputs = _iter_ms_inputs(input_ms_or_folder)

    if use_slurm:
        from .image import _run_commands_with_slurm_cluster

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
