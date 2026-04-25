"""Dask computation backend."""

import io
import logging
import os
import tempfile
import zipfile
from typing import Any, Callable, List, Optional

try:
    from dask.distributed import Client, LocalCluster
    from dask import delayed as dask_delayed

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    Client = None
    LocalCluster = None
    dask_delayed = None

from .base import ComputationBackend

logger = logging.getLogger(__name__)
_CLIENT_NOT_INITIALIZED = "Dask client not initialized"


class DaskBackend(ComputationBackend):
    """Dask computation backend for distributed computing."""

    def __init__(
        self,
        scheduler: Optional[str] = None,
        n_workers: Optional[int] = None,
    ):
        """Initialize Dask backend.

        Parameters
        ----------
        scheduler : str, optional
            Scheduler address (e.g., "tcp://localhost:8786") or None for local
        n_workers : int, optional
            Number of workers (only used for local cluster)
        """
        if not DASK_AVAILABLE:
            raise ImportError(
                "Dask is not installed. Install it with: pip install dask distributed"
            )

        self.scheduler = scheduler
        self.n_workers = n_workers
        self.client: Optional[Client] = None
        self.cluster: Optional[LocalCluster] = None
        self._start_client()

    def _upload_package_to_workers(self) -> None:
        """Zip the almasim package and upload it to all Dask workers.

        This is required when workers are running in a separate environment
        (e.g. on the host machine) that does not have almasim installed.
        After uploading the source, also pip-installs any missing runtime
        dependencies on each worker.
        """
        import importlib.util

        # Locate the package WITHOUT importing it (importing triggers heavy deps)
        spec = importlib.util.find_spec("almasim")
        if spec is None or spec.origin is None:
            logger.warning("Could not locate almasim package for worker upload")
            return

        pkg_dir = os.path.dirname(spec.origin)
        src_dir = os.path.dirname(pkg_dir)  # parent of almasim/ → added to sys.path

        # --- 1. Zip and upload the almasim source ---
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(pkg_dir):
                    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
                    for fname in files:
                        if fname.endswith(".py"):
                            full_path = os.path.join(root, fname)
                            arcname = os.path.relpath(full_path, src_dir)
                            zf.write(full_path, arcname)
            buf.seek(0)

            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name
            try:
                self.client.upload_file(tmp_path)
                logger.info("Uploaded almasim source to Dask workers")
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            logger.warning("Could not upload almasim source to workers: %s", exc)
            return

        # --- 2. Install missing runtime dependencies on each worker ---
        # Map pip package name → importable module name
        _DEPS: "dict[str, str]" = {
            "matplotlib": "matplotlib",
            "astropy": "astropy",
            "numpy": "numpy",
            "pandas": "pandas",
            "scipy": "scipy",
            "h5py": "h5py",
            "scikit-image": "skimage",
            "tqdm": "tqdm",
            "pyvo": "pyvo",
            "tenacity": "tenacity",
        }

        def _install_missing(pkg_map: "dict[str, str]") -> str:
            import importlib.util as _ilu
            import subprocess
            import sys

            missing = [
                pip_name
                for pip_name, mod_name in pkg_map.items()
                if _ilu.find_spec(mod_name) is None
            ]
            if not missing:
                return "all deps present"
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", *missing],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"installed: {missing}"

        try:
            results = self.client.run(_install_missing, _DEPS)
            for worker, msg in results.items():
                logger.info("Worker %s: %s", worker, msg)
        except Exception as exc:
            logger.warning(
                "Could not install deps on Dask workers: %s. "
                "Ensure the worker environment has matplotlib, astropy, numpy, "
                "pandas, scipy, h5py, scikit-image, tqdm, pyvo, tenacity installed.",
                exc,
            )

    def _start_client(self) -> None:
        """Start Dask client."""
        if self.scheduler and self.scheduler != "threads":
            # Connect to external scheduler
            self.client = Client(self.scheduler)
            # Workers on the remote scheduler may not have almasim installed;
            # upload the package so tasks can be unpickled correctly.
            self._upload_package_to_workers()
        else:
            # Create local cluster with processes for true parallelism
            # Explicitly use LocalCluster to avoid threads parameter issues
            cluster_kwargs = {"processes": True}
            if self.n_workers is not None:
                cluster_kwargs["n_workers"] = self.n_workers
            self.cluster = LocalCluster(**cluster_kwargs)
            self.client = Client(self.cluster)

    def scatter(self, data: Any, broadcast: bool = False) -> Any:
        """Scatter data to Dask workers."""
        if self.client is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)
        return self.client.scatter(data, broadcast=broadcast)

    def compute(self, tasks: Any, sync: bool = True) -> Any:
        """Compute tasks using Dask."""
        if self.client is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)
        return self.client.compute(tasks, sync=sync)

    def gather(self, futures: Any) -> List[Any]:
        """Gather results from Dask futures."""
        if self.client is None:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)
        if isinstance(futures, list):
            return self.client.gather(futures)
        else:
            return [self.client.gather([futures])[0]]

    def delayed(self, func: Callable) -> Callable:
        """Create a Dask delayed version of a function.

        Returns a decorator that can be used to wrap function calls.
        """
        if dask_delayed is None:
            raise ImportError("Dask delayed is not available")
        # dask.delayed can be used as a decorator or function
        # When used as decorator: @delayed, when used as function: delayed(func)(*args)
        # We return it directly as it supports both patterns
        return dask_delayed(func)

    def close(self) -> None:
        """Close Dask client and cluster."""
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
