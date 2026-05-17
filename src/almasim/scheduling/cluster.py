"""Singleton utilities for running Dask on top of Slurm.

This module provides:
1. A singleton wrapper around a ``dask_jobqueue.SLURMCluster`` and
   ``dask.distributed.Client``.
2. A top-level helper function that can be submitted with ``client.submit``
   to execute shell subcommands and collect outputs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

try:
    from dask.distributed import Client
    from dask_jobqueue.slurm import SLURMCluster

    SLURM_DASK_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard
    Client = None
    SLURMCluster = None
    SLURM_DASK_AVAILABLE = False


@dataclass(frozen=True)
class SubcommandResult:
    """Result produced by ``run_subcommand``."""

    stdout: str
    stderr: str
    returncode: int


def run_subcommand(
    command: str | Sequence[str],
    cores: int,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
) -> SubcommandResult:
    """Execute a subcommand on a worker and return stdout/stderr/return code.

    Parameters
    ----------
    command : str | Sequence[str]
        Command to execute.
    cores : int
        Number of cores requested by this task. Must be less than the number
        of cores available on the worker node.
    cwd : str | None
        Optional working directory.
    env : Mapping[str, str] | None
        Optional environment variables to merge with the current environment.
    timeout : float | None
        Optional timeout in seconds.
    shell : bool
        Whether to execute through the shell.
    """
    if cores <= 0:
        raise ValueError("cores must be greater than 0")

    node_cores = os.cpu_count() or 1
    if cores >= node_cores:
        raise ValueError(f"cores ({cores}) must be less than node cores ({node_cores})")

    command_env = os.environ.copy()
    if env:
        command_env.update(dict(env))

    # Keep threaded libraries aligned with requested task cores.
    for var_name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        command_env[var_name] = str(cores)

    run_command: str | Sequence[str]
    if shell:
        run_command = command if isinstance(command, str) else " ".join(command)
    else:
        run_command = shlex.split(command) if isinstance(command, str) else command

    try:
        completed = subprocess.run(
            run_command,
            cwd=cwd,
            env=command_env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            shell=shell,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or "")
        stderr = f"{stderr}\nCommand timed out after {timeout} seconds".strip()
        return SubcommandResult(
            stdout=stdout,
            stderr=stderr,
            returncode=124,
        )
    return SubcommandResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


class SlurmDaskClusterSingleton:
    """Singleton manager for a Slurm-backed Dask cluster."""

    _instance: ClassVar[SlurmDaskClusterSingleton | None] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    def _resolve_safe_log_directory(raw_log_directory: Any) -> str:
        """Resolve and validate that worker logs stay under the user's home directory."""
        if not isinstance(raw_log_directory, str):
            raise ValueError("log_directory must be a string")
        if not raw_log_directory.strip():
            raise ValueError("log_directory must not be empty")

        safe_log_root = os.path.realpath(os.path.expanduser("~"))
        log_directory = os.path.realpath(
            os.path.abspath(os.path.expanduser(raw_log_directory))
        )

        try:
            within_safe_root = os.path.commonpath([safe_log_root, log_directory]) == safe_log_root
        except ValueError as exc:
            raise ValueError("log_directory is invalid") from exc

        if not within_safe_root:
            raise ValueError("log_directory must be within the user's home directory")
        return log_directory

    def __init__(
        self,
        queue: str,
        node_cores: int,
        memory: str,
        walltime: str,
        n_jobs: int,
        project: str | None = None,
        scheduler_host: str | None = None,
        scheduler_interface: str | None = None,
        **cluster_kwargs: Any,
    ) -> None:
        if not SLURM_DASK_AVAILABLE:
            raise ImportError(
                "dask-jobqueue is not installed. Install it with: pip install dask-jobqueue"
            )
        if node_cores <= 1:
            raise ValueError("node_cores must be greater than 1")
        if n_jobs <= 0:
            raise ValueError("n_jobs must be greater than 0")
        if Client is None or SLURMCluster is None:
            raise RuntimeError("Dask Slurm dependencies are not available")

        submit_hostname = os.environ.get("HOSTNAME")
        effective_scheduler_host = scheduler_host or submit_hostname

        scheduler_options = dict(cluster_kwargs.pop("scheduler_options", {}))
        if effective_scheduler_host and "host" not in scheduler_options:
            scheduler_options["host"] = effective_scheduler_host

        job_script_prologue = list(cluster_kwargs.pop("job_script_prologue", []))
        if submit_hostname and not any(
            line.startswith("export HOSTNAME=") for line in job_script_prologue
        ):
            # Propagate submit-node hostname to workers when requested by HPC setup.
            job_script_prologue.append(f"export HOSTNAME={shlex.quote(submit_hostname)}")

        worker_extra_args = list(cluster_kwargs.pop("worker_extra_args", []))
        if "--resources" not in worker_extra_args:
            worker_extra_args.extend(["--resources", f"CPU={node_cores}"])

        self.queue = queue
        self.node_cores = node_cores
        self.memory = memory
        self.walltime = walltime
        self.n_jobs = n_jobs
        self.project = project

        default_log_dir = os.path.join(os.path.expanduser("~"), "dask-worker-logs")
        raw_log_directory = cluster_kwargs.pop("log_directory", default_log_dir)
        log_directory = self._resolve_safe_log_directory(raw_log_directory)
        os.makedirs(log_directory, exist_ok=True)

        slurm_kwargs = {
            "queue": queue,
            "cores": node_cores,
            "processes": 1,
            "memory": memory,
            "walltime": walltime,
            "worker_extra_args": worker_extra_args,
            "job_script_prologue": job_script_prologue,
            "scheduler_options": scheduler_options,
            "log_directory": log_directory,
            **cluster_kwargs,
        }
        if scheduler_interface:
            slurm_kwargs.setdefault("interface", scheduler_interface)
        if project:
            slurm_kwargs["project"] = project

        self.cluster = SLURMCluster(**slurm_kwargs)
        self.cluster.scale(n_jobs)
        self.client = Client(self.cluster)
        self._config_signature = (
            queue,
            node_cores,
            memory,
            walltime,
            n_jobs,
            project,
            scheduler_host,
            scheduler_interface,
        )

    @classmethod
    def get_instance(
        cls,
        queue: str = "normal",
        node_cores: int = 32,
        memory: str = "64GB",
        walltime: str = "02:00:00",
        n_jobs: int = 1,
        project: str | None = None,
        scheduler_host: str | None = None,
        scheduler_interface: str | None = None,
        **cluster_kwargs: Any,
    ) -> SlurmDaskClusterSingleton:
        """Create (once) or return the singleton cluster manager instance."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    queue=queue,
                    node_cores=node_cores,
                    memory=memory,
                    walltime=walltime,
                    n_jobs=n_jobs,
                    project=project,
                    scheduler_host=scheduler_host,
                    scheduler_interface=scheduler_interface,
                    **cluster_kwargs,
                )
                return cls._instance

            requested_signature = (
                queue,
                node_cores,
                memory,
                walltime,
                n_jobs,
                project,
                scheduler_host,
                scheduler_interface,
            )
            if cls._instance._config_signature != requested_signature:
                raise RuntimeError(
                    "SlurmDaskClusterSingleton already initialized with different settings. "
                    "Call close_instance() first if you need a different configuration."
                )
            return cls._instance

    @classmethod
    def close_instance(cls) -> None:
        """Close and clear the singleton cluster instance."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.client.close()
                cls._instance.cluster.close()
                cls._instance = None

    def submit_subcommand(
        self,
        command: str | Sequence[str],
        cores: int,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
    ) -> Any:
        """Submit ``run_subcommand`` with worker CPU-resource constraints."""
        if cores <= 0:
            raise ValueError("cores must be greater than 0")
        if cores >= self.node_cores:
            raise ValueError(f"cores ({cores}) must be less than node_cores ({self.node_cores})")

        return self.client.submit(
            run_subcommand,
            command=command,
            cores=cores,
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=shell,
            resources={"CPU": cores},
        )
