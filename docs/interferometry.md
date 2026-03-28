# Interferometry

This page describes ALMASim’s interferometric simulation layer.

## Location

The main code is in [`src/almasim/services/interferometry`](../src/almasim/services/interferometry).

## Main Modules

- `core.py`
- `antenna.py`
- `baselines.py`
- `frequency.py`
- `imaging.py`
- `multiconfig.py`
- `noise.py`
- `total_power.py`
- `utils.py`

## Current Scope

ALMASim currently focuses on single-pointing observation fidelity.

It supports:

- interferometric simulation for ALMA-like antenna arrays
- multi-configuration single-pointing observation plans
- total-power simulation
- combined interferometric result aggregation
- UV mask generation
- dirty beam and dirty image products

It does not currently target mosaics as a first-class workflow.

## Observation Planning

Single-pointing observation plans are built in [`src/almasim/services/observation_plan.py`](../src/almasim/services/observation_plan.py).

Plans can be:

- constructed directly
- inferred from metadata rows
- split automatically into `12m`, `7m`, and `TP` components when metadata contains mixed antenna prefixes

## Interferometric Products

The interferometry layer can produce:

- dirty image cube
- dirty visibility cube
- beam cube
- total sampling cube
- UV mask cube
- U cube
- V cube

These products are useful both for astronomy workflows and for inverse-problem or ML workflows.

## Multi-Configuration Support

The multi-configuration aggregation code is in [`multiconfig.py`](../src/almasim/services/interferometry/multiconfig.py).

It combines:

- per-config interferometric runs
- per-config metadata
- per-config dirty and beam products
- combined interferometric products

## Total Power

Total-power simulation is implemented in [`total_power.py`](../src/almasim/services/interferometry/total_power.py).

It provides:

- diffraction-limited TP beam estimation
- TP smoothing of model cubes
- TP dirty products
- multi-config TP combination

## Fidelity Direction

The current design goal is to move ALMASim closer to `simalma` for single-pointing fidelity while preserving library-first and ML-oriented access to intermediate products.
