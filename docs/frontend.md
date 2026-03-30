# Frontend Workflows

This page summarizes the main Svelte frontend workflows.

## Routes

Current top-level routes include:

- `/metadata`
- `/data`
- `/simulations`
- `/visualizer`
- `/imaging`
- `/combination`

## Metadata

The metadata page is used to:

- query and filter metadata
- inspect result tables
- move into simulation or download workflows

## Simulations

The simulations page is used to:

- select metadata rows
- configure overrides
- preview resolved cube size and output footprint
- choose SNR and PWV modes
- center or offset the target
- choose a background-sky mode
- launch simulations
- inspect live status and logs
- cancel queued or running simulations

## Visualizer

The visualizer page is used to:

- browse output directories
- load ALMASim cube products
- inspect integrated views and basic statistics

## Imaging

The imaging page is used to:

- load dirty and beam products
- run iterative deconvolution
- continue deconvolution without restarting from zero
- compare dirty, component, restored, residual, and reference products

## Combination

The combination page is used to:

- inspect interferometric
- total-power
- TP+INT combined image products

## Simulation Logs

The simulations list includes a log viewer so you can inspect persisted or live backend logs per simulation.

