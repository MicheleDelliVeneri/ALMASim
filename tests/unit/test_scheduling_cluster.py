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


# Fakes for dask-jobqueue's SLURMCluster and distributed's Client, shared by the
# singleton tests (each test used to define its own copy).


class DummyCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False

    def scale(self, n_jobs):
        self.n_jobs = n_jobs

    def close(self):
        self.closed = True


class DummyClient:
    def __init__(self, cluster):
        self.cluster = cluster
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure singleton state is clean for each test."""
    cluster_mod.SlurmDaskClusterSingleton.close_instance()
    yield
    cluster_mod.SlurmDaskClusterSingleton.close_instance()


@pytest.fixture(autouse=True)
def _no_slurm_partition(monkeypatch):
    """Keep the scheduler host independent of the Slurm cluster the tests run on."""
    monkeypatch.setattr(cluster_mod, "_partition_node_address", lambda queue: None)


class _RecordingCluster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.workers = {}

    def scale(self, n_jobs):
        self.n_jobs = n_jobs

    def close(self):
        return None


class _RecordingClient:
    def __init__(self, cluster):
        self.cluster = cluster

    def close(self):
        return None


@pytest.fixture
def recording_cluster(monkeypatch):
    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", _RecordingCluster)
    monkeypatch.setattr(cluster_mod, "Client", _RecordingClient)


@pytest.mark.unit
def test_default_scheduler_host_uses_the_address_routed_to_the_partition(monkeypatch):
    monkeypatch.setattr(cluster_mod, "_partition_node_address", lambda queue: "10.20.25.47")
    monkeypatch.setattr(
        cluster_mod,
        "_source_address_towards",
        lambda address: "10.20.25.11" if address == "10.20.25.47" else None,
    )
    monkeypatch.setenv("HOSTNAME", "compute-node11")

    assert cluster_mod.default_scheduler_host("zwickyA-CPU") == "10.20.25.11"


@pytest.mark.unit
def test_default_scheduler_host_never_picks_loopback(monkeypatch):
    monkeypatch.setattr(cluster_mod, "_partition_node_address", lambda queue: "10.20.25.47")
    monkeypatch.setattr(cluster_mod, "_source_address_towards", lambda address: "127.0.1.1")
    monkeypatch.setenv("HOSTNAME", "localhost")

    assert cluster_mod.default_scheduler_host("zwickyA-CPU") is None


@pytest.mark.unit
def test_source_address_towards_a_local_address_is_local():
    assert cluster_mod._source_address_towards("127.0.0.1") == "127.0.0.1"


@pytest.mark.unit
def test_partition_node_address_without_slurm_is_none(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setenv("PATH", "")
    assert cluster_mod._partition_node_address("any") is None


@pytest.mark.unit
def test_singleton_with_an_interface_leaves_the_host_to_dask(monkeypatch, recording_cluster):
    monkeypatch.setenv("HOSTNAME", "login-node01")

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(
        node_cores=8, n_jobs=1, scheduler_interface="ib0"
    )

    assert "host" not in instance.cluster.kwargs["scheduler_options"]
    assert instance.cluster.kwargs["interface"] == "ib0"


@pytest.mark.unit
def test_unreachable_scheduler_report_reads_the_worker_log(
    monkeypatch, tmp_path, recording_cluster
):
    monkeypatch.setenv("HOME", str(tmp_path))
    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(
        node_cores=8, n_jobs=1, log_directory=str(tmp_path / "logs")
    )
    job = type("Job", (), {"job_id": "5266"})()
    instance.cluster.workers = {"w0": job, "w1": type("Job", (), {"job_id": None})()}
    assert instance.unreachable_scheduler_report() is None, "no log yet"

    (tmp_path / "logs" / "dask-worker-5266.err").write_text(
        "OSError: Timed out trying to connect to tcp://127.0.1.1:45061 after 30 s\n"
    )

    report = instance.unreachable_scheduler_report()
    assert report is not None
    assert "job 5266" in report and "tcp://127.0.1.1:45061" in report


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
def test_the_last_holder_closes_the_shared_cluster(monkeypatch):
    """Two users of one cluster: the first release leaves it running (MADIV-28 W0.3)."""
    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)
    singleton = cluster_mod.SlurmDaskClusterSingleton

    first = singleton.get_instance(node_cores=8, n_jobs=1)
    second = singleton.get_instance(node_cores=8, n_jobs=1)
    assert first is second

    singleton.release_instance()
    assert singleton._instance is first
    assert not first.cluster.closed

    singleton.release_instance()
    assert singleton._instance is None
    assert first.cluster.closed and first.client.closed

    singleton.release_instance()  # nothing left to release: no error


@pytest.mark.unit
def test_close_instance_closes_whoever_still_holds_it(monkeypatch):
    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)
    singleton = cluster_mod.SlurmDaskClusterSingleton

    instance = singleton.get_instance(node_cores=8, n_jobs=1)
    singleton.get_instance(node_cores=8, n_jobs=1)
    singleton.close_instance()

    assert singleton._instance is None
    assert instance.cluster.closed
    # A new cluster after a forced close starts with one holder again.
    singleton.get_instance(node_cores=8, n_jobs=1)
    singleton.release_instance()
    assert singleton._instance is None


@pytest.mark.unit
def test_singleton_rejects_different_reconfiguration(monkeypatch):
    """A second call with different config should fail."""

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=8, n_jobs=1)

    with pytest.raises(RuntimeError, match="already initialized"):
        cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=16, n_jobs=1)


@pytest.mark.unit
def test_submit_subcommand_passes_resource_constraints(monkeypatch):
    """submit_subcommand should submit with CPU resources equal to cores."""

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

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=6, n_jobs=1)

    with pytest.raises(ValueError, match="must be less than node_cores"):
        instance.submit_subcommand(["echo", "x"], cores=6)


@pytest.mark.unit
def test_singleton_passes_hostname_to_job_script_prologue(monkeypatch):
    """Cluster init should export submit HOSTNAME in worker job prologue."""

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

    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)
    monkeypatch.setenv("HOSTNAME", "login-node01")

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(node_cores=8, n_jobs=1)

    assert instance.cluster.kwargs["scheduler_options"]["host"] == "login-node01"


@pytest.mark.unit
def test_singleton_allows_explicit_scheduler_host_override(monkeypatch):
    """Explicit scheduler_host should override HOSTNAME-derived default."""

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
def test_singleton_resolves_log_directory_within_home(monkeypatch, tmp_path):
    """Custom log_directory under HOME should be accepted and normalized."""

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    instance = cluster_mod.SlurmDaskClusterSingleton.get_instance(
        node_cores=8,
        n_jobs=1,
        log_directory="~/logs/../dask-worker-logs",
    )

    assert instance.cluster.kwargs["log_directory"].startswith(str(home_dir.resolve()))


@pytest.mark.unit
def test_singleton_rejects_log_directory_outside_home(monkeypatch, tmp_path):
    """Custom log_directory outside HOME should be rejected."""

    home_dir = tmp_path / "home"
    outside_dir = tmp_path / "outside"
    home_dir.mkdir()
    outside_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setattr(cluster_mod, "SLURM_DASK_AVAILABLE", True)
    monkeypatch.setattr(cluster_mod, "SLURMCluster", DummyCluster)
    monkeypatch.setattr(cluster_mod, "Client", DummyClient)

    with pytest.raises(ValueError, match="log_directory must be within the user's home directory"):
        cluster_mod.SlurmDaskClusterSingleton.get_instance(
            node_cores=8,
            n_jobs=1,
            log_directory=str(outside_dir),
        )


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    ["~", "~/../{sibling}", "{sibling}/logs", "~/logs/../../{sibling_name}/logs", "/etc"],
)
def test_resolve_safe_log_directory_rejects_paths_outside_home(monkeypatch, tmp_path, raw):
    home = tmp_path / "user"
    sibling = tmp_path / "user2"  # shares the home's name as a prefix
    home.mkdir()
    sibling.mkdir()
    monkeypatch.setenv("HOME", str(home))
    raw = raw.format(sibling=sibling, sibling_name=sibling.name)
    with pytest.raises(ValueError, match="within the user's home directory"):
        cluster_mod.SlurmDaskClusterSingleton._resolve_safe_log_directory(raw)


@pytest.mark.unit
def test_resolve_safe_log_directory_follows_symlinks_out_of_home(monkeypatch, tmp_path):
    home = tmp_path / "user"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / "logs").symlink_to(outside)
    monkeypatch.setenv("HOME", str(home))
    with pytest.raises(ValueError, match="within the user's home directory"):
        cluster_mod.SlurmDaskClusterSingleton._resolve_safe_log_directory("~/logs")
    resolved = cluster_mod.SlurmDaskClusterSingleton._resolve_safe_log_directory("~/a/../b")
    assert resolved == str((home / "b").resolve())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "message"), [(None, "must be a string"), ("  ", "must not be empty")]
)
def test_resolve_safe_log_directory_rejects_bad_values(raw, message):
    with pytest.raises(ValueError, match=message):
        cluster_mod.SlurmDaskClusterSingleton._resolve_safe_log_directory(raw)
