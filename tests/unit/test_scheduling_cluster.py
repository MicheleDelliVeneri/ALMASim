"""Unit tests for scheduling cluster utilities."""

from __future__ import annotations

import sys

import pytest

from almasim.scheduling import cluster as cluster_mod

try:
    from dask.distributed import Client, LocalCluster

    DASK_DISTRIBUTED_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Client = None
    LocalCluster = None
    DASK_DISTRIBUTED_AVAILABLE = False


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton state is clean for each test."""
    cluster_mod.SlurmDaskClusterSingleton.close_instance()
    yield
    cluster_mod.SlurmDaskClusterSingleton.close_instance()


@pytest.mark.unit
def test_run_subcommand_returns_stdout_stderr_and_returncode():
    """run_subcommand should capture stdout/stderr and exit code."""
    result = cluster_mod.run_subcommand(
        [
            sys.executable,
            "-c",
            ("import sys; print('hello-out'); print('hello-err', file=sys.stderr); sys.exit(3)"),
        ],
        cores=1,
    )

    assert "hello-out" in result.stdout
    assert "hello-err" in result.stderr
    assert result.returncode == 3


@pytest.mark.unit
def test_run_subcommand_rejects_cores_not_less_than_node(monkeypatch):
    """run_subcommand should enforce cores < node cores."""
    monkeypatch.setattr(cluster_mod.os, "cpu_count", lambda: 4)

    with pytest.raises(ValueError, match="must be less than node cores"):
        cluster_mod.run_subcommand([sys.executable, "-c", "print('x')"], cores=4)


@pytest.mark.unit
def test_run_subcommand_timeout_returns_124():
    """run_subcommand should return code 124 when subprocess times out."""
    result = cluster_mod.run_subcommand(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        cores=1,
        timeout=0.1,
    )

    assert result.returncode == 124
    assert "timed out" in result.stderr.lower()


@pytest.mark.unit
def test_singleton_get_instance_reuses_existing(monkeypatch):
    """get_instance should return the same object for same configuration."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.scaled_to = None
            self.closed = False

        def scale(self, n_jobs):
            self.scaled_to = n_jobs

        def close(self):
            self.closed = True

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    instance_a = cluster_mod.SlurmDaskClusterSingleton.get_instance(
        queue="normal",
        node_cores=8,
        memory="16GB",
        walltime="00:30:00",
        n_jobs=2,
    )
    instance_b = cluster_mod.SlurmDaskClusterSingleton.get_instance(
        queue="normal",
        node_cores=8,
        memory="16GB",
        walltime="00:30:00",
        n_jobs=2,
    )

    assert instance_a is instance_b
    assert instance_a.cluster.scaled_to == 2
    assert instance_a.cluster.kwargs["worker_extra_args"][-1] == "CPU=8"


@pytest.mark.unit
def test_singleton_rejects_different_reconfiguration(monkeypatch):
    """A second call with different config should fail."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def scale(self, n_jobs):
            self.n_jobs = n_jobs

        def close(self):
            return None

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster

        def close(self):
            return None

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=8, n_jobs=1)

    with pytest.raises(RuntimeError, match="already initialized"):
        cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=16, n_jobs=1)


@pytest.mark.unit
def test_submit_subcommand_passes_resource_constraints(monkeypatch):
    """submit_subcommand should submit with CPU resources equal to cores."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def scale(self, n_jobs):
            self.n_jobs = n_jobs

        def close(self):
            return None

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster
            self.submit_calls = []

        def submit(self, func, **kwargs):
            self.submit_calls.append((func, kwargs))
            return "dummy-future"

        def close(self):
            return None

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=10, n_jobs=1)
    future = instance.submit_subcommand(["echo", "hi"], cores=4)

    assert future == "dummy-future"
    assert len(instance.client.submit_calls) == 1
    submitted_func, submitted_kwargs = instance.client.submit_calls[0]
    assert submitted_func is cluster_mod.run_subcommand
    assert submitted_kwargs["resources"] == {"CPU": 4}
    assert submitted_kwargs["cores"] == 4


@pytest.mark.unit
def test_submit_subcommand_rejects_invalid_cores(monkeypatch):
    """submit_subcommand should reject cores >= node_cores."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def scale(self, n_jobs):
            self.n_jobs = n_jobs

        def close(self):
            return None

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster

        def close(self):
            return None

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=6, n_jobs=1)

    with pytest.raises(ValueError, match="must be less than node_cores"):
        instance.submit_subcommand(["echo", "x"], cores=6)


@pytest.mark.unit
def test_singleton_passes_hostname_to_job_script_prologue(monkeypatch):
    """Cluster init should export submit HOSTNAME in worker job prologue."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def scale(self, n_jobs):
            self.n_jobs = n_jobs

        def close(self):
            return None

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster

        def close(self):
            return None

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)
    monkeypatch.setenv("HOSTNAME", "login-node01")

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=8, n_jobs=1)

    prologue = instance.cluster.kwargs["job_script_prologue"]
    assert any(line == "export HOSTNAME=login-node01" for line in prologue)


@pytest.mark.unit
def test_singleton_sets_scheduler_host_from_hostname(monkeypatch):
    """Scheduler host should default to submit-side HOSTNAME."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def scale(self, n_jobs):
            self.n_jobs = n_jobs

        def close(self):
            return None

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster

        def close(self):
            return None

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)
    monkeypatch.setenv("HOSTNAME", "login-node01")

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=8, n_jobs=1)

    assert instance.cluster.kwargs["scheduler_options"]["host"] == "login-node01"


@pytest.mark.unit
def test_singleton_allows_explicit_scheduler_host_override(monkeypatch):
    """Explicit scheduler_host should override HOSTNAME-derived default."""

    class DummyCluster:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def scale(self, n_jobs):
            self.n_jobs = n_jobs

        def close(self):
            return None

    class DummyClient:
        def __init__(self, cluster):
            self.cluster = cluster

        def close(self):
            return None

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)
    monkeypatch.setenv("HOSTNAME", "login-node01")

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(
        node_cores=8,
        n_jobs=1,
        scheduler_host="internal-headnode",
    )

    assert instance.cluster.kwargs["scheduler_options"]["host"] == "internal-headnode"


@pytest.mark.unit
@pytest.mark.skipif(not DASK_DISTRIBUTED_AVAILABLE, reason="dask.distributed not installed")
def test_run_subcommand_end_to_end_with_local_dask_client():
    """run_subcommand should serialize through Dask and return SubcommandResult."""
    assert LocalCluster is not None
    assert Client is not None

    cluster = LocalCluster(n_workers=1, threads_per_worker=2, processes=False)
    client = Client(cluster)
    try:
        future = client.submit(
            cluster_mod.run_subcommand,
            command=[sys.executable, "-c", "print('demo-ok')"],
            cores=1,
        )
        result = future.result(timeout=30)
    finally:
        client.close()
        cluster.close()

    assert isinstance(result, cluster_mod.SubcommandResult)
    assert "demo-ok" in result.stdout
    assert result.returncode == 0
