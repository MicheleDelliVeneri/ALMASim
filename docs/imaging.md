# Imaging and Combination

This page describes ALMASim’s image-domain products and deconvolution workflow.

## Location

Imaging code lives in [`src/almasim/services/imaging`](../src/almasim/services/imaging).

## Current Imaging Products

ALMASim can produce:

- interferometric image cubes
- total-power image cubes
- TP+INT combined image cubes

These are built by `build_image_products()` in [`reconstruction.py`](../src/almasim/services/imaging/reconstruction.py).

## Deterministic Reconstruction

The current deterministic image-domain tools include:

- Wiener-style deconvolution
- TP/INT feather-style merging
- cube regridding
- cube-to-image previews

## CLEAN-Style Deconvolution

ALMASim also provides iterative CLEAN-style deconvolution.

Current behavior:

- operates per spectral slice
- supports resumable state
- distinguishes:
  - component model
  - restored cube
  - residual cube
  - clean-beam-convolved reference

## Edge Handling

The current CLEAN implementation pads slices before deconvolution and crops afterward, so edge and corner sources are handled more robustly than before.

## Imaging archive MeasurementSets with WSClean

`almasim image compute-parameters` and `almasim image image-from-ms` turn a
folder of calibrated MeasurementSets into one WSClean image per science
spectral window. They follow the same contract as the unpack and calibrate
stages: every task leaves a marker, a failure never ends the run, and a rerun
resumes from the markers.

```bash
# 1. one CSV row per (MS, science spectral window)
almasim image compute-parameters /data/ms-calibrated imaging_parameters.csv

# 2. image them on Slurm, 10 cores per WSClean, 8 workers
almasim image image-from-ms imaging_parameters.csv /data/images \
  --wsclean-bin /opt/spack/.../bin/wsclean \
  --num-cores 10 --max-cores-per-node 95 \
  --slurm-queue normal --slurm-n-jobs 8 --slurm-walltime 1-00:00:00 \
  --task-timeout 7200
```

**compute-parameters** reads each MS with python-casacore and writes the
WSClean geometry per spectral window: field of view, synthesised beam, pixel
count. By default it keeps only the spectral windows that actually hold
visibilities (an ALMA MS lists every correlator window, WVR and pointing ones
included, but after `split` only the science windows have rows) and records
the fields observed with the `OBSERVE_TARGET` intent in `target_field_ids`,
so imaging leaves the calibrator scans out. `--all-spws` disables that. An MS
whose `<ms>.done` calibration marker is missing is skipped when the folder
uses markers, and an MS that cannot be read is listed in
`<csv>.failed.tsv` while the scan continues (`--fail-fast` to abort instead).

**image-from-ms** runs one WSClean per CSV row, each in its own subprocess on
a worker, writing to `<output>/<ms stem>/SPW-<n>/wsclean-*.fits`. Next to
that directory it leaves:

| file | meaning |
|---|---|
| `SPW-<n>.done` | finished; `wsclean-image.fits` exists. JSON with the command and the run time |
| `SPW-<n>.failed` | failed; JSON with the error (exit code, the `what():` line of a crash, a timeout, a missing binary, or "exited 0 but wrote no image") and the log path |
| `SPW-<n>.log` | the full WSClean output, written live on the shared filesystem: `tail -f` it from the submit node |

Tasks with a `.done` marker are skipped on a rerun and everything else is
retried; `--overwrite-outputs` redoes all of them and `--skip-existing`
additionally trusts a `wsclean-image.fits` produced before markers existed
(and writes its marker). Failed tasks are listed at the end and the command
exits 1 once every task has been attempted; `--fail-fast` stops at the first
failure instead. If a worker dies with its task (walltime, node failure) the
driver writes the `.failed` marker itself, so the accounting stays complete:
work remaining is always *CSV rows − `.done` markers*.

Each WSClean gets `-j <num-cores>` and runs with `OPENBLAS_NUM_THREADS=1`
(WSClean refuses to start otherwise when linked against a threaded OpenBLAS),
`-mem` scaled to its share of the node, and `-field <target_field_ids>` when
the CSV has that column. `--postprocess-backend sync` runs the tasks one at a
time in the current process, which is handy for a smoke test on one MS.

## Frontend Pages

The frontend exposes:

- `Combination` page for TP/INT comparison products
- `Imaging` page for deconvolution workflows
- `Visualizer` page for general product inspection

## Example

The example CLI in [`examples/imaging_cli.py`](../examples/imaging_cli.py) generates synthetic imaging products and validates that the reconstruction improves on the dirty cube.
