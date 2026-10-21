"""Slurm computation backend using dask-jobqueue."""

from time import monotonic
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from almasim.scheduling.cluster import SLURM_DASK_AVAILABLE, SlurmDaskClusterSingleton

try:
    SLURM_AVAILABLE = SLURM_DASK_AVAILABLE
except ImportError:
    SLURM_AVAILABLE = False

from .base import DaskClientBackend

# How often wait_for_workers looks at the worker logs for connection failures.
_UNREACHABLE_CHECK_S = 15.0


class SlurmBackend(DaskClientBackend):
    """Slurm computation backend using dask-jobqueue."""

    def __init__(
        self,
        queue: str = "normal",
        project: Optional[str] = None,
        walltime: str = "02:00:00",
        cores: int = 1,
        node_cores: Optional[int] = None,
        memory: str = "4GB",
        n_workers: int = 4,
        scheduler_host: Optional[str] = None,
        scheduler_interface: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize Slurm backend.

        Parameters
        ----------
        queue : str
            Slurm queue name (default: "normal")
        project : str, optional
            Slurm project/account name
        walltime : str
            Job walltime in HH:MM:SS format (default: "02:00:00")
        cores : int
            Number of task cores per worker job (default: 1)
        node_cores : int, optional
            Total cores on each worker node. Defaults to ``cores + 1``.
            Set explicitly when using submit_subcommand() so the resource
            accounting matches the actual node size.
        memory : str
            Memory per worker (default: "4GB")
        n_workers : int
            Number of workers to start (default: 4)
        scheduler_host : str, optional
            IP or hostname the Dask scheduler advertises to workers. Use the
            internal/HPC network address when the public hostname is firewalled
            from compute nodes.
        scheduler_interface : str, optional
            Network interface for the scheduler/worker (e.g. "ib0", "eth0").
        **kwargs
            Additional arguments passed to SLURMCluster.
        """
        if not SLURM_AVAILABLE:
            raise ImportError(
                "dask-jobqueue is not installed. Install it with: pip install dask-jobqueue"
            )

        self.queue = queue
        self.project = project
        self.walltime = walltime
        self.cores = cores
        self.node_cores = node_cores if node_cores is not None else cores + 1
        self.memory = memory
        self.n_workers = n_workers
        self.scheduler_host = scheduler_host
        self.scheduler_interface = scheduler_interface
        job_extra_directives = list(kwargs.pop("job_extra_directives", None) or [])
        if "--requeue" not in job_extra_directives and "--no-requeue" not in job_extra_directives:
            job_extra_directives.append("--requeue")
        kwargs["job_extra_directives"] = job_extra_directives
        self.kwargs = kwargs
        self._cluster_manager: Optional[SlurmDaskClusterSingleton] = None

        self.cluster: Optional[Any] = None
        self.client: Optional[Any] = None
        self._start_cluster()

    def _start_cluster(self) -> None:
        """Start Slurm cluster and client via singleton manager."""
        manager = SlurmDaskClusterSingleton.get_instance(
            queue=self.queue,
            node_cores=self.node_cores,
            memory=self.memory,
            walltime=self.walltime,
            n_jobs=self.n_workers,
            project=self.project,
            scheduler_host=self.scheduler_host,
            scheduler_interface=self.scheduler_interface,
            **self.kwargs,
        )
        self._cluster_manager = manager
        self.cluster = manager.cluster
        self.client = manager.client

    def submit_subcommand(
        self,
        command: Union[str, Sequence[str]],
        cores: int,
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        timeout: Optional[float] = None,
        shell: bool = False,
    ) -> Any:
        """Submit a shell command to run on a Slurm worker.

        Delegates to ``SlurmDaskClusterSingleton.submit_subcommand`` so that
        both ``cli.image`` and ``cli.products`` go through the same code path.
        """
        if self._cluster_manager is None:
            raise RuntimeError("Dask cluster not initialized")
        return self._cluster_manager.submit_subcommand(
            command=command,
            cores=cores,
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=shell,
        )

    def submit_callable(self, func: Callable, *, cores: int, **kwargs: Any) -> Any:
        """Submit ``func(**kwargs)`` to a Slurm worker with ``cores`` CPU resources."""
        if self._cluster_manager is None:
            raise RuntimeError("Dask cluster not initialized")
        return self._cluster_manager.submit_callable(func, cores=cores, **kwargs)

    def wait_for_workers(self, n_workers: int, timeout: Optional[float] = None) -> int:
        """Wait until ``n_workers`` workers have registered or ``timeout`` s pass.

        Returns the number registered; a timeout is not an error. Raises
        ``RuntimeError`` as soon as a worker job logs that it could not reach
        the scheduler, since none of them will.
        """
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        deadline = None if timeout is None else monotonic() + timeout
        while True:
            step = _UNREACHABLE_CHECK_S
            if deadline is not None:
                step = min(step, deadline - monotonic())
            if step <= 0:
                break
            try:
                self.client.wait_for_workers(n_workers, timeout=step)
                break
            except (TimeoutError, OSError):
                pass
            manager = getattr(self, "_cluster_manager", None)
            report = manager.unreachable_scheduler_report() if manager is not None else None
            if report:
                raise RuntimeError(report)
        return len(self.client.scheduler_info().get("workers", {}))

    def close(self) -> None:
        """Close Slurm cluster and client."""
        if getattr(self, "_cluster_manager", None) is not None:
            SlurmDaskClusterSingleton.close_instance()
            self._cluster_manager = None
        self.client = None
        self.cluster = None
