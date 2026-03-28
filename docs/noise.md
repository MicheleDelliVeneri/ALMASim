# Noise

This page describes ALMASim’s current noise model.

## Location

Noise handling lives primarily in [`src/almasim/services/interferometry/noise.py`](../src/almasim/services/interferometry/noise.py).

## Current Model

ALMASim uses a PWV-aware channel noise model that depends on:

- precipitable water vapor
- channel frequencies
- channel width
- integration time
- elevation
- antenna diameter
- number of antennas

The noise profile is computed per channel and then calibrated against a reference noise level used by the simulation pipeline.

## Inputs

The key inputs are:

- `pwv_mm`
- `ground_temperature_k`
- per-channel observing frequencies
- observing time
- antenna count
- array type through antenna diameter and array configuration

## SNR Handling

ALMASim supports:

- manual SNR override
- automatic SNR derivation from metadata sensitivity fields

The current auto mode uses:

- `Cont_sens_mJybeam`
- `Line_sens_10kms_mJybeam`

This keeps the default behavior tied to available ALMA metadata rather than arbitrary hardcoded values.

## What Noise Is Not

The background sky features described elsewhere are not part of the noise model.

ALMASim treats them separately:

- noise is an observing/corruption term
- background sky is an additive astrophysical sky component

## Future Directions

Likely future improvements include:

- more explicit atmospheric transmission coupling
- richer correlator-dependent noise settings
- better calibration of line and continuum sensitivity relationships across bands
