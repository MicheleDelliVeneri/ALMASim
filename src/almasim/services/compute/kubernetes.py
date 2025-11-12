"""Kubernetes computation backend using dask-kubernetes."""
from typing import Any, Callable, List, Optional, Dict

try:
    from dask_kubernetes import KubeCluster
    from dask.distributed import Client
    from dask import delayed as dask_delayed
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    KubeCluster = None
    Client = None
    dask_delayed = None

from .base import ComputationBackend


class KubernetesBackend(ComputationBackend):
    """Kubernetes computation backend using dask-kubernetes."""

    def __init__(
        self,
        namespace: Optional[str] = None,
        n_workers: int = 4,
        image: Optional[str] = None,
        resources: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ):
        """Initialize Kubernetes backend.
        
        Parameters
        ----------
        namespace : str, optional
            Kubernetes namespace (default: current namespace)
        n_workers : int
            Number of workers to start (default: 4)
        image : str, optional
            Docker image for workers
        resources : dict, optional
            Resource requests/limits for workers
        **kwargs
            Additional arguments passed to KubeCluster
        """
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "dask-kubernetes is not installed. Install it with: pip install dask-kubernetes"
            )

        self.namespace = namespace
        self.n_workers = n_workers
        self.image = image
        self.resources = resources or {}
        self.kwargs = kwargs

        self.cluster: Optional[KubeCluster] = None
        self.client: Optional[Client] = None
        self._start_cluster()

    def _start_cluster(self) -> None:
        """Start Kubernetes cluster and client."""
        cluster_kwargs = {
            "n_workers": self.n_workers,
            **self.kwargs,
        }
        if self.namespace:
            cluster_kwargs["namespace"] = self.namespace
        if self.image:
            cluster_kwargs["image"] = self.image
        if self.resources:
            cluster_kwargs["resources"] = self.resources

        self.cluster = KubeCluster(**cluster_kwargs)
        self.cluster.scale(self.n_workers)
        self.client = Client(self.cluster)

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to Kubernetes workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.scatter(data, broadcast=broadcast)

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks using Kubernetes workers."""
        if self.client is None:
            raise RuntimeError("Dask client not initialized")
        return self.client.compute(tasks, sync=sync)

    def gather(self, futures: Any) -> List[Any]:
        """Gather results from Kubernetes workers."""
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
        """Close Kubernetes cluster and client."""
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

