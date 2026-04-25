"""Unit tests for compute backends: local, slurm, kubernetes, factory."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from almasim.services.compute.local import DelayedTask, LocalBackend, _execute_task
from almasim.services.compute.sync import SyncBackend


# ===========================================================================
# _execute_task (module-level helper in local.py)
# ===========================================================================


@pytest.mark.unit
def test_execute_task_callable():
    """_execute_task calls and returns result of a callable."""
    result = _execute_task(lambda: 42)
    assert result == 42


@pytest.mark.unit
def test_execute_task_with_compute():
    """_execute_task calls .compute() on non-callable objects that have .compute()."""
    # Must be a non-callable object so the callable() check doesn't fire first.
    class HasCompute:
        def compute(self):
            return "computed"
    result = _execute_task(HasCompute())
    assert result == "computed"


@pytest.mark.unit
def test_execute_task_passthrough():
    """_execute_task passes through plain values."""
    result = _execute_task(99)
    assert result == 99


# ===========================================================================
# DelayedTask
# ===========================================================================


@pytest.mark.unit
def test_delayed_task_compute():
    """DelayedTask.compute() calls the wrapped function with args/kwargs."""
    dt = DelayedTask(lambda x, y: x + y, 3, 4)
    assert dt.compute() == 7


@pytest.mark.unit
def test_delayed_task_with_kwargs():
    """DelayedTask.compute() passes kwargs correctly."""
    dt = DelayedTask(lambda a, b=10: a * b, 5, b=3)
    assert dt.compute() == 15


# ===========================================================================
# LocalBackend
# ===========================================================================


@pytest.mark.unit
def test_local_backend_scatter_returns_data():
    """scatter() returns the data unchanged."""
    backend = LocalBackend(n_workers=1)
    try:
        data = [1, 2, 3]
        assert backend.scatter(data) is data
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_compute_list_sync():
    """compute() with list of callables returns results synchronously."""
    backend = LocalBackend(n_workers=1)
    try:
        tasks = [lambda: 1, lambda: 2, lambda: 3]
        results = backend.compute(tasks, sync=True)
        assert results == [1, 2, 3]
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_compute_single_sync():
    """compute() with single callable returns single result."""
    backend = LocalBackend(n_workers=1)
    try:
        result = backend.compute(lambda: "hello", sync=True)
        assert result == "hello"
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_gather_futures():
    """gather() collects results from futures with .result()."""
    backend = LocalBackend(n_workers=1)
    try:
        mock_future = MagicMock()
        mock_future.result.return_value = 42
        results = backend.gather([mock_future])
        assert results == [42]
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_gather_plain_values():
    """gather() passes through plain values without .result()."""
    backend = LocalBackend(n_workers=1)
    try:
        # Values without .result() attribute are returned as-is
        results = backend.gather([10, 20])
        assert results == [10, 20]
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_gather_single():
    """gather() with single non-list value returns single-element list."""
    backend = LocalBackend(n_workers=1)
    try:
        result = backend.gather(99)
        assert result == [99]
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_delayed():
    """delayed() returns a callable that creates DelayedTask."""
    backend = LocalBackend(n_workers=1)
    try:
        delayed_add = backend.delayed(lambda x, y: x + y)
        dt = delayed_add(3, 4)
        assert isinstance(dt, DelayedTask)
        assert dt.compute() == 7
    finally:
        backend.close()


@pytest.mark.unit
def test_local_backend_context_manager():
    """LocalBackend works as a context manager."""
    with LocalBackend(n_workers=1) as backend:
        result = backend.compute(lambda: "ctx", sync=True)
    assert result == "ctx"
    # After exit, executor should be closed
    assert backend.executor is None


@pytest.mark.unit
def test_local_backend_close_idempotent():
    """close() can be called multiple times without error."""
    backend = LocalBackend(n_workers=1)
    backend.close()
    backend.close()  # Should not raise


@pytest.mark.unit
def test_local_backend_compute_async_list():
    """compute() with sync=False returns futures."""
    backend = LocalBackend(n_workers=1)
    try:
        # Lambdas can't be pickled by ProcessPoolExecutor; use DelayedTask
        # (module-level picklable) instead.
        tasks = [DelayedTask(int, 1), DelayedTask(int, 2)]
        futures = backend.compute(tasks, sync=False)
        assert len(futures) == 2
        results = [f.result() for f in futures]
        assert results == [1, 2]
    finally:
        backend.close()


# ===========================================================================
# SyncBackend (already partially tested, add more coverage)
# ===========================================================================


@pytest.mark.unit
def test_sync_backend_context_manager():
    """SyncBackend works as a context manager."""
    with SyncBackend() as backend:
        assert backend is not None


@pytest.mark.unit
def test_sync_backend_scatter_returns_data():
    """SyncBackend.scatter() returns data unchanged."""
    backend = SyncBackend()
    data = {"key": "value"}
    assert backend.scatter(data) is data


@pytest.mark.unit
def test_sync_backend_gather():
    """SyncBackend.gather() returns the futures as-is."""
    backend = SyncBackend()
    futures = [1, 2, 3]
    result = backend.gather(futures)
    assert result == [1, 2, 3]


# ===========================================================================
# SlurmBackend — unavailable path
# ===========================================================================


@pytest.mark.unit
def test_slurm_backend_raises_when_unavailable():
    """SlurmBackend raises ImportError when dask-jobqueue is not installed."""
    import almasim.services.compute.slurm as slurm_mod

    original = slurm_mod.SLURM_AVAILABLE
    try:
        slurm_mod.SLURM_AVAILABLE = False
        from almasim.services.compute.slurm import SlurmBackend

        with pytest.raises(ImportError, match="dask-jobqueue"):
            SlurmBackend()
    finally:
        slurm_mod.SLURM_AVAILABLE = original


@pytest.mark.unit
def test_slurm_backend_methods_raise_when_client_none():
    """SlurmBackend methods raise RuntimeError when client is None."""
    import almasim.services.compute.slurm as slurm_mod

    original = slurm_mod.SLURM_AVAILABLE
    try:
        slurm_mod.SLURM_AVAILABLE = True
        from almasim.services.compute.slurm import SlurmBackend

        with patch.object(SlurmBackend, "_start_cluster", return_value=None):
            backend = SlurmBackend.__new__(SlurmBackend)
            backend.client = None
            backend.cluster = None

        with pytest.raises(RuntimeError, match="Dask client not initialized"):
            backend.scatter("data")

        with pytest.raises(RuntimeError, match="Dask client not initialized"):
            backend.compute([])

        with pytest.raises(RuntimeError, match="Dask client not initialized"):
            backend.gather([])
    finally:
        slurm_mod.SLURM_AVAILABLE = original


@pytest.mark.unit
def test_slurm_backend_context_manager_calls_close():
    """SlurmBackend.__exit__ calls close()."""
    import almasim.services.compute.slurm as slurm_mod

    original = slurm_mod.SLURM_AVAILABLE
    try:
        slurm_mod.SLURM_AVAILABLE = True
        from almasim.services.compute.slurm import SlurmBackend

        with patch.object(SlurmBackend, "_start_cluster", return_value=None):
            backend = SlurmBackend.__new__(SlurmBackend)
            backend.client = None
            backend.cluster = None

        backend.__enter__()
        assert backend.__exit__(None, None, None) is None
    finally:
        slurm_mod.SLURM_AVAILABLE = original


# ===========================================================================
# KubernetesBackend — unavailable path
# ===========================================================================


@pytest.mark.unit
def test_kubernetes_backend_raises_when_unavailable():
    """KubernetesBackend raises ImportError when dask-kubernetes is not installed."""
    import almasim.services.compute.kubernetes as k8s_mod

    original = k8s_mod.KUBERNETES_AVAILABLE
    try:
        k8s_mod.KUBERNETES_AVAILABLE = False
        from almasim.services.compute.kubernetes import KubernetesBackend

        with pytest.raises(ImportError, match="dask-kubernetes"):
            KubernetesBackend()
    finally:
        k8s_mod.KUBERNETES_AVAILABLE = original


@pytest.mark.unit
def test_kubernetes_backend_methods_raise_when_client_none():
    """KubernetesBackend methods raise RuntimeError when client is None."""
    import almasim.services.compute.kubernetes as k8s_mod

    original = k8s_mod.KUBERNETES_AVAILABLE
    try:
        k8s_mod.KUBERNETES_AVAILABLE = True
        from almasim.services.compute.kubernetes import KubernetesBackend

        with patch.object(KubernetesBackend, "_start_cluster", return_value=None):
            backend = KubernetesBackend.__new__(KubernetesBackend)
            backend.client = None
            backend.cluster = None

        with pytest.raises(RuntimeError, match="Dask client not initialized"):
            backend.scatter("data")

        with pytest.raises(RuntimeError, match="Dask client not initialized"):
            backend.compute([])

        with pytest.raises(RuntimeError, match="Dask client not initialized"):
            backend.gather([])
    finally:
        k8s_mod.KUBERNETES_AVAILABLE = original


@pytest.mark.unit
def test_kubernetes_backend_context_manager():
    """KubernetesBackend.__exit__ calls close()."""
    import almasim.services.compute.kubernetes as k8s_mod

    original = k8s_mod.KUBERNETES_AVAILABLE
    try:
        k8s_mod.KUBERNETES_AVAILABLE = True
        from almasim.services.compute.kubernetes import KubernetesBackend

        with patch.object(KubernetesBackend, "_start_cluster", return_value=None):
            backend = KubernetesBackend.__new__(KubernetesBackend)
            backend.client = None
            backend.cluster = None

        backend.__enter__()
        backend.__exit__(None, None, None)
    finally:
        k8s_mod.KUBERNETES_AVAILABLE = original


# ===========================================================================
# create_backend factory
# ===========================================================================


@pytest.mark.unit
def test_create_backend_sync():
    """create_backend('sync') returns a SyncBackend."""
    from almasim.services.compute.factory import create_backend

    backend = create_backend("sync")
    assert isinstance(backend, SyncBackend)


@pytest.mark.unit
def test_create_backend_local():
    """create_backend('local') returns a LocalBackend."""
    from almasim.services.compute.factory import create_backend

    backend = create_backend("local", n_workers=1)
    assert isinstance(backend, LocalBackend)
    backend.close()


@pytest.mark.unit
def test_create_backend_dask_with_mock():
    """create_backend('dask') creates a DaskBackend (mocked)."""
    from almasim.services.compute.factory import create_backend
    from almasim.services.compute.dask_backend import DaskBackend

    with patch.object(DaskBackend, "__init__", return_value=None):
        backend = create_backend("dask")
    assert isinstance(backend, DaskBackend)


@pytest.mark.unit
def test_create_backend_unknown_raises():
    """create_backend with unknown type raises ValueError."""
    from almasim.services.compute.factory import create_backend

    with pytest.raises(ValueError, match="Unknown backend type"):
        create_backend("warp_drive")


@pytest.mark.unit
def test_create_backend_slurm_raises_when_unavailable():
    """create_backend('slurm') raises when dask-jobqueue is not available."""
    import almasim.services.compute.slurm as slurm_mod
    from almasim.services.compute.factory import create_backend

    original = slurm_mod.SLURM_AVAILABLE
    try:
        slurm_mod.SLURM_AVAILABLE = False
        with pytest.raises(ImportError):
            create_backend("slurm")
    finally:
        slurm_mod.SLURM_AVAILABLE = original


@pytest.mark.unit
def test_create_backend_kubernetes_raises_when_unavailable():
    """create_backend('kubernetes') raises when dask-kubernetes is not available."""
    import almasim.services.compute.kubernetes as k8s_mod
    from almasim.services.compute.factory import create_backend

    original = k8s_mod.KUBERNETES_AVAILABLE
    try:
        k8s_mod.KUBERNETES_AVAILABLE = False
        with pytest.raises(ImportError):
            create_backend("kubernetes")
    finally:
        k8s_mod.KUBERNETES_AVAILABLE = original
