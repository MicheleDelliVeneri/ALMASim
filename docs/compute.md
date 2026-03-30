# Compute Backends

This page describes ALMASim’s compute backends.

## Location

Compute backends live in [`src/almasim/services/compute`](../src/almasim/services/compute).

## Supported Backends

ALMASim currently includes:

- synchronous in-process backend
- local process-based backend
- Dask backend
- Slurm-oriented backend
- Kubernetes-oriented backend

## Main Modules

- `base.py`
- `sync.py`
- `local.py`
- `dask_backend.py`
- `slurm.py`
- `kubernetes.py`
- `factory.py`

## Use Cases

- `sync`
  - notebooks
  - examples
  - debugging
- `local`
  - local CPU parallelism
- `dask`
  - distributed execution
  - cluster or workstation scheduling
- `slurm`
  - HPC job submission workflows
- `kubernetes`
  - cluster-native environments

## Design Goal

The simulation service layer should not need to know which backend is active. The backend selection is injected into the workflow and the simulation code uses the common backend interface.
