"""Slurm computation backend using dask-jobqueue."""

from typing import Any, Callable, Dict, List, Optional

from almasim.scheduling.cluster import SLURM_DASK_AVAILABLE, SlurmDaskClusterSingleton

try:
    from dask import delayed as dask_delayed

    SLURM_AVAILABLE = SLURM_DASK_AVAILABLE
except ImportError:
    SLURM_AVAILABLE = False
    dask_delayed = None

from .base import ComputationBackend


class SlurmBackend(ComputationBackend):
    """Slurm computation backend using dask-jobqueue."""

    def __init__(
        self,
        queue: str = "normal",
        project: Optional[str] = None,
        walltime: str = "02:00:00",
        cores: int = 1,
        memory: str = "4GB",
        n_workers: int = 4,
        **kwargs: Dict[str, Any],
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
            Number of cores per worker (default: 1)
        memory : str
            Memory per worker (default: "4GB")
        n_workers : int
            Number of workers to start (default: 4)
        **kwargs
            Additional arguments passed to SLURMCluster
        """
        if not SLURM_AVAILABLE:
            raise ImportError(
                "dask-jobqueue is not installed. Install it with: pip install dask-jobqueue"
            )

        self.queue = queue
        self.project = project
        self.walltime = walltime
        self.cores = cores
        self.memory = memory
        self.n_workers = n_workers
        self.kwargs = kwargs
        self._cluster_manager: Optional[SlurmDaskClusterSingleton] = None

        self.cluster: Optional[Any] = None
        self.client: Optional[Any] = None
        self._start_cluster()

    def _start_cluster(self) -> None:
        """Start Slurm cluster and client via singleton manager."""
        manager = SlurmDaskClusterSingleton.get_instance(
            queue=self.queue,
            node_cores=self.cores + 1,
            memory=self.memory,
            walltime=self.walltime,
            n_jobs=self.n_workers,
            project=self.project,
            **self.kwargs,
        )
        self._cluster_manager = manager
        self.cluster = manager.cluster
        self.client = manager.client

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to Slurm workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.scatter(data, broadcast=broadcast)

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks using Slurm workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.compute(tasks, sync=sync)

    def gather(self, futures: Any) -> List[Any]:
        """Gather results from Slurm workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        if isinstance(futures, list):
            return self.client.gather(futures)
        else:
            return [self.client.gather([futures])[0]]

    def delayed(self, func: Callable) -> Callable:
        """Create a Dask delayed version of a function."""
        if dask_delayed is None:
            raise ImportError("Dask delayed is not available")
        return dask_delayed(func)

    def close(self) -> None:
        """Close Slurm cluster and client."""
        if getattr(self, "_cluster_manager", None) is not None:
            SlurmDaskClusterSingleton.close_instance()
            self._cluster_manager = None
        self.client = None
        self.cluster = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
