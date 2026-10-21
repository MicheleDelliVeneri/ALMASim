"""Options that several commands share (MADIV-28 T3.2).

Each alias is an ``Annotated`` type carrying its ``typer.Option``; a command
declares ``slurm_queue: SlurmQueue = "normal"`` and keeps its own default.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Optional

import typer


class Backend(StrEnum):
    """Where a command's tasks run."""

    sync = "sync"
    slurm = "slurm"


PostprocessBackend = Annotated[
    Backend,
    typer.Option(
        "--postprocess-backend",
        case_sensitive=False,
        help="Where the tasks run: sync (this process, one at a time) or slurm (Slurm jobs).",
    ),
]
SlurmQueue = Annotated[str, typer.Option("--slurm-queue", help="Slurm queue/partition.")]
SlurmProject = Annotated[
    Optional[str], typer.Option("--slurm-project", help="Slurm project/account.")
]
SlurmWalltime = Annotated[
    str, typer.Option("--slurm-walltime", help="Slurm walltime per worker job (HH:MM:SS).")
]
SlurmMemory = Annotated[str, typer.Option("--slurm-memory", help="Memory per Slurm worker.")]
SlurmCores = Annotated[int, typer.Option("--slurm-cores", min=1, help="Cores per Slurm worker.")]
SchedulerHost = Annotated[
    Optional[str],
    typer.Option(
        "--scheduler-host",
        help="Scheduler host advertised to the workers (default: this host).",
    ),
]
SchedulerInterface = Annotated[
    Optional[str],
    typer.Option(
        "--scheduler-interface", help="Scheduler/worker network interface (e.g. ib0, eth0)."
    ),
]
TaskTimeout = Annotated[
    float,
    typer.Option("--task-timeout", min=1, help="Timeout in seconds for each worker-side command."),
]
MaxCoresPerNode = Annotated[
    int,
    typer.Option(
        "--max-cores-per-node",
        min=1,
        help="Cores per node, for the Slurm workers' resource accounting.",
    ),
]


@dataclass(frozen=True)
class SlurmOptions:
    """The Slurm worker settings of a command, as compute-backend keyword arguments."""

    queue: str = "normal"
    project: Optional[str] = None
    walltime: str = "02:00:00"
    memory: str = "16GB"
    n_workers: int = 1
    cores: Optional[int] = None
    scheduler_host: Optional[str] = None
    scheduler_interface: Optional[str] = None

    def backend_kwargs(self) -> dict[str, Any]:
        """``create_backend("slurm", **kwargs)`` arguments; unset optional ones are left out."""
        kwargs: dict[str, Any] = {
            "queue": self.queue,
            "project": self.project,
            "walltime": self.walltime,
            "memory": self.memory,
            "n_workers": self.n_workers,
        }
        for name in ("cores", "scheduler_host", "scheduler_interface"):
            value = getattr(self, name)
            if value not in (None, ""):
                kwargs[name] = value
        return kwargs
