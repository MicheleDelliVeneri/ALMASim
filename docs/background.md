# Background Sky

This page describes ALMASim’s additive background-sky simulation.

## Purpose

ALMASim can now add an astrophysical background sky to the clean cube before interferometric simulation.

This is separate from thermal or atmospheric noise.

## Default Behavior

The default background mode is `none`.

That means:

- the science target is centered at the phase center
- no additive background sky is injected

## Background Modes

### `blank_field_dsfg`

Adds a sparse field of faint dusty star-forming galaxies.

This is intended to approximate blank-field extragalactic confusion in the ALMA band regime.

### `dusty_diffuse`

Adds a correlated low-spatial-frequency dusty background field.

This is useful as a first approximation to diffuse mm/sub-mm background structure.

### `combined`

Adds both of the above.

## Controls

ALMASim exposes:

- `background_mode`
- `background_level`
- `background_seed`

`background_seed` makes the background reproducible.

## Current Physical Intent

The current background implementation is a pragmatic first step for ALMA-band simulations:

- faint unresolved dusty galaxies
- diffuse dusty structure with smooth spectral dependence

It is not yet a full cosmological sky model.

## Outputs

When a background is generated:

- it is added to the clean sky cube
- its total injected flux is recorded in metadata
- a `background-cube_*` product is exported when outputs are persisted

## Scope Notes

This is an astrophysical background, not:

- atmospheric emission
- calibration error
- correlator noise

Those belong elsewhere in the simulation chain.

