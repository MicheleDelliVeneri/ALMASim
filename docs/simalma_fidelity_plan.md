# ALMASim To `simalma` Fidelity Plan

## Purpose

This document is the implementation roadmap for making ALMASim closer to CASA
`simalma` in simulation fidelity while preserving ALMASim's current strengths:

- library-first Python API
- in-memory execution
- explicit access to clean cubes, dirty cubes, visibilities, UV masks, and
  measurement-operator products
- ML-friendly dataset export

The comparison target is the behavior implemented by CASA `simalma`,
`simobserve`, and `simanalyze`, with emphasis on physical fidelity rather than
CASA UI parity.

## Current High-Level Gap

Relative to `simalma`, ALMASim is currently strongest at synthetic sky
generation and ML-oriented outputs, but weaker in the parts that dominate
observational fidelity:

- atmospheric and thermal noise modeling
- multi-configuration ALMA orchestration
- single-pointing observing geometry and array semantics
- total power simulation and TP+INT combination
- CASA-like imaging, deconvolution, and fidelity diagnostics

## Guiding Principles

1. Match physical behavior before matching file formats.
2. Preserve the staged API:
   `generate_clean_cube() -> simulate_observation() -> export_results()`.
3. Add new capabilities as composable stages rather than one monolithic task.
4. Keep explicit intermediate products available for validation and ML use.
5. Validate each phase against CASA on shared sky models and observing setups.

## Target Architecture

The end-state pipeline should look like this:

1. `prepare_sky_model()`
2. `plan_observation()`
3. `simulate_interferometric_configs()`
4. `simulate_total_power()`
5. `combine_measurements()`
6. `image_products()`
7. `analyze_fidelity()`
8. `export_results()`

Suggested new service modules:

- `src/almasim/services/observation_plan.py`
- `src/almasim/services/interferometry/noise.py`
- `src/almasim/services/interferometry/multiconfig.py`
- `src/almasim/services/interferometry/total_power.py`
- `src/almasim/services/imaging/reconstruction.py`
- `src/almasim/services/imaging/analysis.py`

## Explicit Non-Goal

This roadmap is focused on improving fidelity for single-pointing ALMA
simulations. Full mosaic planning and multi-pointing observation support are
out of scope for now and should not block the core fidelity work.

## P0: Core Fidelity

### Status

Complete.

### Goal

Close the largest physics and observing-geometry gaps relative to `simalma`.

### Scope

#### 1. PWV-aware thermal noise and atmosphere model

Replace the current scalar complex Gaussian noise with a noise model that
depends on:

- PWV
- frequency
- bandwidth / channel width
- integration time
- antenna diameter
- elevation / airmass
- receiver and sky temperature terms

Minimum implementation target:

- deterministic function that returns per-channel noise standard deviations
- optional per-baseline scaling
- explicit config object stored in metadata

Suggested API:

```python
NoiseModelConfig(
    pwv_mm: float,
    ground_temperature_k: float,
    receiver_model: str,
    site: str = "ALMA",
)
```

```python
compute_channel_noise(config, freqs_hz, bandwidth_hz, integration_s, elevation_deg)
```

#### 2. Multi-configuration interferometric simulation

Support a list of ALMA interferometric configurations in one run, rather than a
single `antenna_array`.

Minimum implementation target:

- multiple 12 m configs
- optional ACA 7 m config
- per-config integration time
- per-config correlator and metadata
- per-config outputs plus a combined INT product

Suggested API:

```python
ObservationConfig(
    name: str,
    array_type: Literal["12m", "7m"],
    antenna_config: str,
    total_time_s: float,
    correlator: str | None = None,
)
```

#### 3. Single-pointing observation-plan layer

Add an explicit single-pointing observation-plan layer rather than relying on
implicit geometry spread across the simulator.

Minimum implementation target:

- single pointing
- explicit phase center
- explicit primary-beam definition
- explicit per-configuration observing metadata
- explicit observation-plan object shared across INT and TP paths

Suggested outputs:

- phase center
- primary beam model metadata
- observation-plan metadata
- per-configuration observing setup

### Deliverables

- noise model service and tests
- observation-plan dataclasses and planner
- multi-config simulation stage returning per-config dirty products
- regression coverage that verifies single-pointing config changes propagate into
  UV support and dirty-beam behavior

### Acceptance Criteria

- ALMASim can simulate at least:
  - one 12 m config
  - one 12 m + one 7 m configuration pair
- the single-pointing setup is explicit in the simulation inputs and metadata
- PWV materially affects output noise levels
- uv coverage and dirty beam change appropriately across configs

This phase is considered complete in-repo once the above behavior is covered by
tests. Cross-tool parity reporting against CASA remains part of the broader
parity and benchmark work described in P2.

## P1: Imaging And Combination

### Status

Complete.

### Goal

Add the CASA-like downstream steps that turn simulated measurements into
combined science products.

### Scope

#### 1. Total power simulation

Implement TP simulation as its own stage rather than treating all arrays as
interferometric.

Minimum implementation target:

- TP observing mode
- configurable number of TP antennas
- TP map extent rules
- TP gridding to image domain

#### 2. INT concatenation / joint imaging

Support combined imaging across multiple interferometric configs.

Minimum implementation target:

- concatenate or jointly image 12 m + 7 m products
- configurable relative weighting between arrays
- explicit combined dirty beam and combined sampling outputs

#### 3. TP + INT combination

Add a combination path equivalent in spirit to CASA feathering.

Minimum implementation target:

- regrid TP image to INT grid
- primary-beam-aware scaling
- feather-like frequency-domain combination
- export of intermediate combination products

#### 4. Deconvolution / reconstruction stage

ALMASim currently stops at dirty products. Add an imaging stage that can
produce deconvolved images.

Options:

- internal CLEAN-like implementation
- external imaging backend abstraction
- optional CASA bridge if available in the environment

The first target should be a simple, reproducible, non-interactive imaging
path, not full CASA feature parity.

### Deliverables

- total power simulation service
- TP gridding service
- INT combination service
- TP+INT feather/merge service
- first deconvolution stage with reproducible defaults
- staged API extension for image reconstruction outputs

### Acceptance Criteria

- ALMASim can produce:
  - per-config INT dirty products
  - combined INT image products
  - TP image products
  - TP+INT merged image products

This phase is considered complete in-repo once the above products are produced
through the staged API and covered by tests. Cross-tool comparison against CASA
for beam size, recovered flux fraction, and large-scale structure recovery
remains part of the broader parity work described in P2.

## P2: Parity, Diagnostics, And Ingestion

### Goal

Make ALMASim easy to validate directly against CASA and more usable for
realistic benchmark studies.

### Scope

#### 1. External skymodel ingestion

Support simulation from external FITS sky models and cubes, not only internally
generated models.

Minimum implementation target:

- FITS image input
- FITS cube input
- component-list-like source table input
- header override / alignment options

#### 2. CASA-style fidelity diagnostics

Add post-imaging comparison products similar in purpose to `simanalyze`.

Minimum implementation target:

- convolved model image
- residual image
- difference image
- fidelity map
- integrated flux comparison
- uv residual statistics

#### 3. Parity test suite

Build a small locked benchmark set comparing ALMASim against CASA on common
reference scenarios.

Suggested benchmark scenarios:

- compact single field, 12 m only
- 12 m + 7 m compact source
- TP + INT large-scale emission recovery

#### 4. Optional interoperability outputs

If needed after physical parity improves:

- MeasurementSet export or bridge
- CASA-compatible manifest / parameter dumps

This is deliberately P2 because it improves interoperability more than raw
fidelity.

### Deliverables

- FITS ingestion path
- fidelity-analysis service
- benchmark cases checked into `tests/` or `examples/`
- reproducible parity report template
- optional interoperability design note

### Acceptance Criteria

- ALMASim can ingest the same sky model used by CASA
- parity reports exist for the benchmark scenarios
- fidelity metrics are generated automatically
- regressions in parity can be caught in CI or a scheduled benchmark job

## Validation Strategy

For each phase, validate against a fixed CASA reference setup using the same:

- sky model
- array configuration
- integration time
- PWV
- phase center / primary beam setup
- imaging cell and image size

Track at minimum:

- dirty beam FWHM
- image noise RMS
- uv sampling coverage
- total recovered flux
- residual RMS
- fidelity image summary statistics

## Recommended Implementation Order

1. P0.1 noise model
2. P0.2 observation-plan and multi-config support
3. P0.3 single-pointing observation-plan layer
4. P1.1 total power simulation
5. P1.2 combined INT imaging
6. P1.3 TP+INT merge
7. P1.4 first deconvolution stage
8. P2.1 FITS ingestion
9. P2.2 fidelity diagnostics
10. P2.3 benchmark and parity suite
11. P2.4 optional interoperability outputs

## What Not To Regress

These features should remain first-class throughout the work:

- `persist=False` / `save_mode="memory"`
- HDF5 ML shard export
- direct access to `dirty_vis`, `uv_mask_cube`, `u_cube`, `v_cube`
- notebook and SLURM-friendly Python entry points

## Suggested First Milestone

The first milestone should be a narrow but meaningful parity case:

- external or synthetic FITS cube
- one 12 m configuration
- one PWV-aware noise model
- one single-pointing observation
- one dirty-image parity report against CASA

If that milestone is not quantitatively close, later work on TP combination and
downstream imaging will not matter much.
