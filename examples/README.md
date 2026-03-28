# Staged API Examples

This directory contains runnable examples for the staged simulation API:

- `staged_pipeline_cli.py`
  Runs clean-cube generation, in-memory observation simulation, and ML shard export from the command line.
- `staged_pipeline_notebook.ipynb`
  Notebook walkthrough of the same pipeline for interactive exploration and DDRM data preparation.

The examples use the in-process `sync` compute backend so they do not require Dask, process pools, or a running scheduler.

CLI usage:

```bash
python examples/staged_pipeline_cli.py \
  --row-idx 0 \
  --project-name ddrm_demo \
  --ml-shard-path examples/output/ddrm_training_sample.h5
```
