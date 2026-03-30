"""Unit tests for synchronous computation backend."""

from dask import delayed

from almasim.services.compute import SyncBackend, create_backend


def test_sync_backend_factory():
    """Factory should create the synchronous backend."""
    backend = create_backend("sync")
    assert isinstance(backend, SyncBackend)


def test_sync_backend_compute_and_gather():
    """Sync backend should compute plain callables and gather results."""
    backend = SyncBackend()

    result = backend.compute(lambda: 42)
    gathered = backend.gather(result)

    assert result == 42
    assert gathered == [42]


def test_sync_backend_supports_dask_delayed():
    """Sync backend should evaluate delayed objects in-process."""
    backend = SyncBackend()
    task = delayed(sum)([1, 2, 3])

    result = backend.compute(task)

    assert result == 6
