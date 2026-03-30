# Skymodels

This page describes ALMASim’s source-model layer.

## Location

Skymodel implementations live in [`src/almasim/skymodels`](../src/almasim/skymodels).

## Supported Source Types

ALMASim currently supports:

- `point`
- `gaussian`
- `extended`
- `diffuse`
- `galaxy-zoo`
- `molecular`
- `hubble-100`

## Common Structure

All skymodels are built on the base class in [`base.py`](../src/almasim/skymodels/base.py).

They receive:

- a datacube object
- continuum flux array
- line fluxes
- spectral line positions and widths
- spatial parameters when needed

## Point and Gaussian Models

For compact-source tests:

- `PointlikeSkyModel` inserts a point source into the cube
- `GaussianSkyModel` inserts an extended 2D Gaussian source

These are useful for:

- debugging
- PSF and CLEAN validation
- controlled synthetic datasets

## Physically Richer Models

### `extended`

Uses TNG-backed source content for more realistic extended systems.

### `galaxy-zoo`

Uses Galaxy Zoo image assets as spatial morphology priors.

### `hubble-100`

Uses Hubble-derived image assets.

### `molecular`

Generates molecular-cloud-like structured emission.

### `diffuse`

Builds diffuse emission fields.

## Source Placement

The science target is now:

- centered at the phase center by default
- optionally shifted by explicit `source_offset_x_arcsec` and `source_offset_y_arcsec`

Random source placement is no longer the default.

## Serendipitous Sources

ALMASim can optionally inject additional serendipitous sources with the existing serendipitous-source utilities.

That is separate from the new background-sky feature:

- serendipitous sources are discrete extra sources
- background sky is an additive field-level background component
