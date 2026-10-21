"""Kubernetes computation backend using dask-kubernetes."""

from typing import Any

try:
    from dask.distributed import Client
    from dask_kubernetes import KubeCluster

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    KubeCluster = None
    Client = None

from .base import DaskClientBackend


class KubernetesBackend(DaskClientBackend):
    """Kubernetes computation backend using dask-kubernetes."""

    def __init__(
        self,
        namespace: str | None = None,
        n_workers: int = 4,
        image: str | None = None,
        resources: dict[str, Any] | None = None,
        **kwargs: dict[str, Any],
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

        self.cluster: KubeCluster | None = None
        self.client: Client | None = None
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

    def close(self) -> None:
        """Close Kubernetes cluster and client."""
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None
