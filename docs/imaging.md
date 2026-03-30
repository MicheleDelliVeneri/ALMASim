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

## Frontend Pages

The frontend exposes:

- `Combination` page for TP/INT comparison products
- `Imaging` page for deconvolution workflows
- `Visualizer` page for general product inspection

## Example

The example CLI in [`examples/imaging_cli.py`](../examples/imaging_cli.py) generates synthetic imaging products and validates that the reconstruction improves on the dirty cube.
