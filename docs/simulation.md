# Simulation

This page describes the main ALMASim simulation workflow.

## Simulation Model

ALMASim simulates a single-pointing ALMA observation from a clean sky cube through measurement-space products and image-domain products.

The central service lives in [`src/almasim/services/simulation.py`](../src/almasim/services/simulation.py).

## Main Entry Points

- `SimulationParams`
- `generate_clean_cube()`
- `simulate_observation()`
- `image_products()`
- `export_results()`
- `run_simulation()`
- `write_ml_dataset_shard()`
- `estimate_simulation_footprint()`

These are re-exported at package level from [`src/almasim/__init__.py`](../src/almasim/__init__.py).

## Staged Workflow

### 1. `generate_clean_cube()`

This stage:

- resolves metadata and overrides
- computes beam-based cell size and final cube dimensions
- builds a clean sky cube from the chosen skymodel
- keeps the target at phase center by default
- applies optional explicit source offsets in arcseconds
- adds optional background sky components
- prepares observation-plan and metadata payloads

### 2. `simulate_observation()`

This stage:

- runs interferometric simulations for the configured single-pointing observation plan
- supports multiple interferometric configurations
- supports total-power simulation when TP configs are present
- returns dirty cubes, dirty visibilities, beam cubes, UV masks, and per-config results

### 3. `image_products()`

This stage:

- builds deterministic image-domain products from measurement-space outputs
- produces interferometric image cubes
- produces total-power image cubes
- produces TP+INT combined cubes

### 4. `export_results()`

This stage:

- persists optional result files to disk
- writes simulation parameter summaries
- writes cube products in the configured save mode
- writes optional ML shards

## Full Run

`run_simulation()` orchestrates the whole sequence:

1. generate clean cube
2. simulate observation
3. reconstruct image products
4. export results

## Simulation Parameters

`SimulationParams` includes:

- source and project metadata
- pointing coordinates
- ALMA band and frequency support
- continuum and line sensitivity metadata
- PWV and timing
- source type
- optional manual cube-dimension overrides
- optional manual SNR override
- background mode and level
- explicit source offsets
- output and compute settings

## Auto-Derived Parameters

If you do not override them manually, ALMASim can derive:

- `n_pix` from field of view and estimated cell size
- `n_channels` from the frequency-support metadata
- effective SNR from metadata sensitivity fields

## Preflight Estimate

`estimate_simulation_footprint()` computes:

- resolved pixel count
- resolved channel count
- cell size
- beam size
- raw cube size in GiB
- rough raw output footprint

The frontend uses this for the pre-run estimate panel.

## Outputs

Depending on the run configuration, ALMASim can produce:

- clean/model cube
- dirty cube
- dirty visibilities
- beam cube
- total sampling cube
- UV mask cube
- U/V coordinate cubes
- per-config interferometric results
- TP products
- TP+INT image products
- background cube
- ML HDF5 shard

## Persistence Modes

ALMASim supports:

- standard persisted file output
- pure Python mode with `persist=False`
- `save_mode="memory"` to avoid normal disk exports
- separate ML shard export even when standard outputs are skipped
