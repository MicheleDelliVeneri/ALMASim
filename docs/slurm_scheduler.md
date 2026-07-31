# SLURM Scheduler

This page describes how to use the SLURM-backed Dask cluster helper implemented in [src/almasim/scheduling/cluster.py](../src/almasim/scheduling/cluster.py).

## Overview

The module provides two main pieces:

- `SlurmDaskClusterSingleton`: singleton wrapper around `dask_jobqueue.slurm.SLURMCluster` and `dask.distributed.Client`
- `run_subcommand(...)`: function designed to be submitted through Dask and return:
  - `stdout`
  - `stderr`
  - `returncode`

## Main Behavior

### Cluster singleton

`SlurmDaskClusterSingleton` creates one shared SLURM-backed Dask cluster per process and reuses it.
If called again with a different configuration, it raises an error and asks for `close_instance()` first.

### CPU resource model

- Worker resources are declared as `CPU=<node_cores>`.
- Each submitted task requests `resources={"CPU": cores}`.
- Task cores are validated with `cores < node_cores`.

This keeps each task below total cores per node.

### Scheduler host/interface

For multi-network HPC systems, scheduler advertisement can be controlled:

- `scheduler_host`: host exposed by scheduler to workers.
- `scheduler_interface`: network interface used by Dask communication.

Default behavior uses submit-side `HOSTNAME` when `scheduler_host` is not provided.

## API sketch

```python
from almasim.scheduling.cluster import SlurmDaskClusterSingleton

manager = SlurmDaskClusterSingleton.get_instance(
    queue="normal",
    node_cores=8,
    memory="16GB",
    walltime="00:30:00",
    n_jobs=1,
    project=None,
    scheduler_host="headnode-internal",  # optional
    scheduler_interface="ib0",  # optional
)

future = manager.submit_subcommand(
    command=["bash", "-lc", "echo hello"],
    cores=2,
    timeout=30,
)
result = future.result(timeout=120)

print(result.returncode)
print(result.stdout)
print(result.stderr)

SlurmDaskClusterSingleton.close_instance()
```

## Demo Script

The complete runnable example is in [examples/slurm_cluster_submit_demo.py](../examples/slurm_cluster_submit_demo.py).

### What the demo does

1. Creates the singleton cluster
2. Waits for at least one worker
3. Submits a shell command with CPU constraints
4. Prints command result (`stdout`, `stderr`, `returncode`)
5. Closes the singleton

### Useful options

- `--queue`: SLURM partition
- `--node-cores`: total cores per node
- `--n-jobs`: number of SLURM jobs/workers
- `--scheduler-host`: explicit scheduler host advertised to workers
- `--scheduler-interface`: network interface for communication
- `--worker-start-timeout`: wait timeout for first worker
- `--result-timeout`: wait timeout for future result
- `--task-timeout`: timeout for worker-side subprocess

### Example command

```bash
python examples/slurm_cluster_submit_demo.py \
  --command "echo hello" \
  --cores 2 \
  --node-cores 8 \
  --n-jobs 1 \
  --scheduler-host "$(hostname)" \
  --scheduler-interface ib0 \
  --worker-start-timeout 60 \
  --result-timeout 120 \
  --task-timeout 30
```

## Troubleshooting

If workers do not appear, check:

1. Scheduler address printed by the demo is reachable from compute nodes.
2. `sacct` for failed `dask-worker` jobs.
3. `scontrol show job <jobid>` to find `StdOut`/`StdErr` logs.
4. Partition/account limits and firewall rules.

A common failure is worker timeout while connecting to scheduler on an unreachable host/interface.
