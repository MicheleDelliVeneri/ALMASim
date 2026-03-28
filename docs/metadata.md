# Metadata

This page describes ALMASim’s ALMA metadata workflows.

## Location

Metadata code lives in:

- [`src/almasim/services/metadata`](../src/almasim/services/metadata)
- [`src/almasim/services/metadata/tap`](../src/almasim/services/metadata/tap)

## Capabilities

ALMASim can:

- query ALMA metadata through TAP
- normalize TAP columns into stable application fields
- infer array type from antenna metadata
- export metadata rows to CSV
- feed metadata rows directly into simulation workflows

## Metadata-Driven Simulation

Metadata rows are used to derive:

- target coordinates
- ALMA band and frequency support
- PWV
- integration time
- field of view
- angular and velocity resolution
- continuum sensitivity
- line sensitivity
- antenna arrays
- observation configuration splits

The simulation page uses metadata rows as the main source of truth and only uses manual form fields as overrides.

## Query Interfaces

ALMASim supports metadata usage through:

- Python services
- backend API
- frontend metadata page
- CLI example scripts
- notebooks

## Relevant Example Scripts

- [`examples/query_metadata_cli.py`](../examples/query_metadata_cli.py)
- [`examples/query_metadata_notebook.ipynb`](../examples/query_metadata_notebook.ipynb)
