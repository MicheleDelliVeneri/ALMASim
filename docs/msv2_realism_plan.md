# MSv2 Realism Plan

This document defines the implementation path from ALMASim's current MeasurementSet support to a more realistic ALMA-style MSv2 export.

It separates two targets:

- `Target A`: minimal valid MSv2
- `Target B`: realistic ALMA MSv2

The distinction matters because a table can be structurally valid and open in standard MeasurementSet readers, while still being far from a realistic ALMA observation product.

## Current State

ALMASim already has:

- row-wise internal visibility tables
- a pure-Python native MeasurementSet `.ms` writer
- core main-table columns:
  - `UVW`
  - `TIME`
  - `TIME_CENTROID`
  - `INTERVAL`
  - `EXPOSURE`
  - `ANTENNA1`
  - `ANTENNA2`
  - `DATA`
  - `MODEL_DATA`
  - `FLAG`
  - `WEIGHT`
  - `SIGMA`
- core subtables:
  - `ANTENNA`
  - `SPECTRAL_WINDOW`
  - `POLARIZATION`
  - `DATA_DESCRIPTION`
  - `FIELD`
  - `OBSERVATION`
  - `SOURCE`
  - `STATE`
  - `HISTORY`

This is enough to call the export a native measurement-set logical model, but not enough to call it a standard `.ms` writer or realistic ALMA output.

## Target A: Minimal Valid MSv2

This target means:

- the exported data can be serialized to a standard `.ms`
- the `.ms` opens in standard MeasurementSet readers without immediate schema failure
- the main table and required subtables are internally consistent
- visibility data are represented as rows, not gridded UV products

### Target A Gaps

- incomplete or simplified subtable semantics
- limited polarization model
- minimal scan and state structure
- simplified feed and processor description
- no guarantee yet that all downstream tools consuming MS input behave correctly
- no regression test against a standard external `.ms` reader/writer environment

### Target A Deliverables

1. Implement a native standard `.ms` serializer

- serialize the existing logical model into the standard on-disk MeasurementSet table layout
- keep linked subtables
- add a dedicated smoke script that writes and reopens a tiny MS

2. Add the remaining basic structural subtables

- `FEED`
- `PROCESSOR`
- `POINTING` as minimal single-pointing metadata

3. Tighten consistency rules

- `FIELD_ID`, `DATA_DESC_ID`, `OBSERVATION_ID`, `STATE_ID` must match subtable rows exactly
- `SPECTRAL_WINDOW` and `POLARIZATION` must agree with `DATA` shape
- `ANTENNA` row count must match antenna indices in the main table

4. Define a documented minimal compatibility contract

- which columns are guaranteed
- which subtables are guaranteed
- which downstream MS operations are expected to work
- which are explicitly out of scope

### Target A Acceptance Criteria

- ALMASim exports a small `.ms`
- standard MS readers can inspect the MS metadata without failure
- one documented smoke test exists outside the Python unit suite

## Target B: Realistic ALMA MSv2

This target means:

- the MS is not only valid, but observationally plausible
- metadata and table layout resemble a real ALMA single-pointing product closely enough to support downstream MS workflows with fewer caveats
- weights, sigmas, channelization, state structure, pointing metadata, and scan structure are physically meaningful

## Gap Analysis: Minimal Valid vs Realistic ALMA MSv2

### 1. Visibility sampling

Current:

- row-wise visibilities exist
- baseline/time/channel sampling is synthetic and simplified

Needed for realism:

- clearer scan segmentation
- more realistic integration cadence
- realistic mapping from observation plan to row grouping

### 2. Polarization and correlation products

Current:

- minimal single-correlation handling

Needed for realism:

- support realistic ALMA polarization products
- correct `CORR_TYPE` and `CORR_PRODUCT`
- consistent `FEED` table semantics

### 3. Spectral setup

Current:

- channel frequencies and widths are written
- one simplified SPW model

Needed for realism:

- richer spectral-window description
- sideband and reference metadata with clearer ALMA semantics
- support for multiple SPWs if simulation mode requires it later

### 4. Pointing and field metadata

Current:

- one field center
- no meaningful `POINTING` table

Needed for realism:

- single-pointing `POINTING` rows consistent with observation time
- explicit tie between phase center, field center, and pointing center

### 5. Scan / state / intent model

Current:

- one minimal state
- one simple scan numbering path

Needed for realism:

- better observation intents
- scan segmentation aligned with time ranges
- more realistic `STATE` and `OBS_MODE`

### 6. Weights, sigma, and noise provenance

Current:

- weights and sigmas are simulated and internally consistent
- not yet pipeline-like

Needed for realism:

- noise-derived weights from the ALMASim thermal model
- explicit relation between simulated sensitivity and `SIGMA`
- more realistic per-row weighting behavior

### 7. Antenna / feed / processor description

Current:

- `ANTENNA` is present
- `FEED` and `PROCESSOR` are missing or minimal

Needed for realism:

- basic ALMA-like feed metadata
- processor metadata consistent with a simulated correlator path

### 8. Ancillary subtables

Current:

- no meaningful `WEATHER`, `SYSCAL`, or similar calibration-oriented content

Needed for realism:

- at least a clear design decision on whether these are:
  - absent by design
  - minimal placeholders
  - physically simulated

## Implementation Phases

## Phase 1: Harden Minimal MSv2

Goal:

- make Target A reliable and testable

Work:

- add `FEED`, `PROCESSOR`, and minimal `POINTING`
- add a dedicated MS smoke-test script
- document the minimal schema contract
- ensure validation against standard external `.ms` readers on a supported environment

Files most likely involved:

- `src/almasim/services/products/ms_io.py`
- `src/almasim/services/interferometry/visibility.py`
- `docs/`
- `examples/`

Acceptance:

- minimal valid MSv2 is reproducible on a supported machine

## Phase 2: Observation Metadata Realism

Goal:

- make the MS observational metadata closer to ALMA single-pointing practice

Work:

- add meaningful `POINTING`
- refine `FIELD`, `STATE`, `OBSERVATION`, and scan structure
- add correlator/processor metadata
- improve antenna/feed semantics

Acceptance:

- metadata inspection looks plausible for a single-pointing ALMA-like simulation

## Phase 3: Weight / Sigma / Noise Realism

Goal:

- ensure statistical columns correspond to ALMASim's simulated noise model

Work:

- derive `SIGMA` and `WEIGHT` from the thermal noise path
- document the mapping from simulation sensitivity to row-level weights
- test consistency across channels and configs

Acceptance:

- `WEIGHT` and `SIGMA` are traceable to the simulation physics, not just placeholders

## Phase 4: Polarization and Spectral Fidelity

Goal:

- reduce the largest remaining scientific simplifications

Work:

- richer correlation handling
- proper `POLARIZATION` semantics
- improved SPW metadata
- optional multi-SPW support if needed

Acceptance:

- polarization/spectral metadata no longer look synthetic in obvious ways

## Phase 5: ALMA-Like Ancillary Tables

Goal:

- decide and implement what extra realism is needed beyond the core MS

Work:

- evaluate `WEATHER`, `SYSCAL`, `SOURCE` richness, `POINTING` density
- choose minimal placeholders versus physical simulation

Acceptance:

- ancillary table policy is explicit and documented

## Priority Order

Recommended order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

This order is deliberate:

- first make MS export robust
- then make metadata believable
- then improve scientific/statistical realism

## Non-Goals

This plan does not currently target:

- full ALMA archive parity
- pipeline-calibrated MS reproduction
- mosaic-specific measurement sets
- total-power archive product parity

The target is realistic single-pointing simulated interferometric MSv2 output.

## Practical Recommendation

The next concrete implementation step should be:

1. add minimal `POINTING`, `FEED`, and `PROCESSOR`
2. add a standalone `msv2_smoke.py` example
3. validate against standard external `.ms` readers instead of relying on ad hoc compatibility checks only

That closes the gap between "we can write a real MSv2" and "we can rely on it."
