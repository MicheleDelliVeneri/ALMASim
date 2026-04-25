# ALMASim

[![PyPI version](https://badge.fury.io/py/almasim.svg)](https://pypi.org/project/almasim/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Documentation](https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/docs.yml/badge.svg)](https://michedelliveneri.github.io/ALMASim/)
[![CI](https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/lint_and_test.yml/badge.svg)](https://github.com/MicheleDelliVeneri/ALMASim/actions/workflows/lint_and_test.yml)
[![codecov](https://codecov.io/gh/MicheleDelliVeneri/ALMASim/branch/main/graph/badge.svg)](https://codecov.io/gh/MicheleDelliVeneri/ALMASim)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

ALMASim is a **library-first** Python environment for simulating ALMA observations, exploring ALMA metadata, downloading science products, and building ML-ready radio/mm-wave datasets.

It provides reusable services in [`src/almasim`](src/almasim) that can be driven by CLI scripts, Jupyter notebooks, a FastAPI backend, or direct Python code — all through the same staged API.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Staged Simulation API](#staged-simulation-api)
- [Skymodels](#skymodels)
- [Compute Backends](#compute-backends)
- [Metadata and Downloads](#metadata-and-downloads)
- [Backend Service](#backend-service)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Key Capabilities

**Simulation**
- Build clean sky cubes from point, Gaussian, extended, molecular-cloud, diffuse, Galaxy Zoo, and Hubble-100 source models
- Simulate single-pointing ALMA interferometric observations with multi-configuration support (12m, 7m, TP)
- PWV-aware per-channel noise model
- Additive astrophysical background sky — faint dusty galaxies, diffuse emission, or combined
- Optional serendipitous source injection
- Iterative CLEAN-style deconvolution with resumable state
- TP+INT feather-style image combination

**Data Products**
- Dirty cube, dirty visibilities, beam cube, UV mask cube, U/V coordinate cubes
- Interferometric, total-power, and combined TP+INT image cubes
- ML-ready HDF5 shards (clean cube + dirty cube + dirty visibilities + UV mask + metadata)
- Native MeasurementSet (`.ms`) export via CASA tools or python-casacore

**Metadata and Archive**
- Query ALMA observations via TAP with rich inclusion/exclusion filters
- Normalise TAP columns into stable application fields
- Resolve DataLink products, download ALMA data products with parallel support
- Unpack raw ASDMs into MeasurementSets
- Apply delivered calibration to produce calibrated science MSs

**Compute**
- Synchronous, local multiprocess, Dask, Slurm, and Kubernetes backends
- Backend-agnostic simulation service layer

---

## Architecture

```
src/almasim/          ← installable library  (pip install almasim)
  services/
    simulation.py     ← staged pipeline entry points
    interferometry/   ← UV sampling, baselines, noise, TP
    imaging/          ← deconvolution, TP+INT combination
    metadata/         ← TAP queries, normalisation
    products/         ← MS export, HDF5 shards, cube export
    compute/          ← backend abstraction
    archive/          ← ASDM unpack, calibration apply
    astro/            ← spectral lines, redshift, parameters
  skymodels/          ← source model implementations

backend/              ← FastAPI service  (Docker: ghcr.io/…/almasim-backend)
frontend/             ← Svelte UI  (requires Docker Compose)
examples/             ← CLI scripts and Jupyter notebooks
```

The library layer owns all domain logic. The backend is a thin adapter over library services. CLI scripts and notebooks call the same staged services directly.

---

## Installation

### Library only (cross-platform)

```bash
pip install almasim
```

### With CASA tools (Linux x86-64 only)

`casatools` and `casatasks` wheels are Linux-only. Install the optional `[casa]` extra on a supported Linux system:

```bash
pip install "almasim[casa]"
```

The `[casa]` extra enables:
- Native MeasurementSet export via `casatools`
- ASDM-to-MS conversion via `casatasks.importasdm`
- Calibration application via `casatasks.applycal`

Without `[casa]`, all simulation, imaging, metadata, and download features still work. The MS export path falls back to `python-casacore` if available:

```bash
pip install "almasim[ms-casacore]"
```

### From source (development)

```bash
git clone https://github.com/MicheleDelliVeneri/ALMASim.git
cd ALMASim
pip install uv
uv sync --group dev
```

### Backend service (Docker Compose)

The FastAPI backend and Svelte frontend require Docker Compose:

```bash
git clone https://github.com/MicheleDelliVeneri/ALMASim.git
cd ALMASim
docker compose up
```

The backend image is available pre-built from GHCR:

```bash
docker pull ghcr.io/michelledelliveneri/almasim-backend:latest
```

---

## Quick Start

### Query ALMA metadata

```python
from almasim.services.metadata.tap.service import query_by_science_type, InclusionFilters

df = query_by_science_type(
    include=InclusionFilters(science_keyword=["Galaxies"], band=[6])
)
print(df[["ALMA_source_name", "Band", "spatial_resolution"]].head())
```

### Run a simulation from a metadata row

```python
from almasim import SimulationParams, run_simulation
from pathlib import Path

params = SimulationParams.from_metadata_row(
    row,                          # pandas Series from a metadata query
    idx=0,
    main_dir=Path("src/almasim"),
    output_dir=Path("output"),
    project_name="my_project",
)

result = run_simulation(params)
```

### Use the staged API

```python
from almasim import (
    SimulationParams,
    generate_clean_cube,
    simulate_observation,
    image_products,
    export_results,
)

params = SimulationParams.from_metadata_row(row, idx=0, ...)

cube_result  = generate_clean_cube(params)
obs_result   = simulate_observation(params, cube_result)
img_result   = image_products(params, obs_result)
export_results(params, cube_result, obs_result, img_result)
```

---

## Staged Simulation API

The pipeline is split into four composable stages:

| Stage | Function | What it does |
|---|---|---|
| 1 | `generate_clean_cube()` | Build sky cube from skymodel, apply background |
| 2 | `simulate_observation()` | Run interferometric + TP simulation, return dirty products |
| 3 | `image_products()` | Deconvolve, combine INT+TP, build image cubes |
| 4 | `export_results()` | Write cubes, ML shards, parameter summaries to disk |

`run_simulation()` orchestrates all four in sequence.

`write_ml_dataset_shard()` exports an HDF5 shard (clean cube + dirty cube + dirty visibilities + UV mask + metadata) independently of the main export path.

`estimate_simulation_footprint()` returns resolved pixel count, channel count, cell size, beam size, and raw output size in GiB — useful for pre-run capacity checks.

Full reference: [Simulation docs](https://michedelliveneri.github.io/ALMASim/simulation.html)

---

## Skymodels

| Source type | Description |
|---|---|
| `point` | Point source — PSF and CLEAN validation |
| `gaussian` | 2-D Gaussian — compact extended source |
| `extended` | TNG-backed realistic extended emission |
| `galaxy-zoo` | Galaxy Zoo image morphology prior |
| `hubble-100` | Hubble Top-100 image morphology prior |
| `molecular` | Molecular cloud structured emission |
| `diffuse` | Correlated diffuse emission field |

All skymodels accept explicit `source_offset_x_arcsec` / `source_offset_y_arcsec` to shift the science target from phase center.

**Additive background sky** (independent of the main source):

| Mode | Effect |
|---|---|
| `blank_field_dsfg` | Faint dusty star-forming galaxies |
| `dusty_diffuse` | Correlated low-spatial-frequency dusty background |
| `combined` | Both of the above |

Full reference: [Skymodels docs](https://michedelliveneri.github.io/ALMASim/skymodels.html)

---

## Compute Backends

Select via `SimulationParams.compute_backend`:

| Backend | Use case |
|---|---|
| `sync` | Notebooks, examples, debugging |
| `local` | Local CPU parallelism |
| `dask` | Distributed execution, cluster scheduling |
| `slurm` | HPC job submission |
| `kubernetes` | Cluster-native environments |

Full reference: [Compute docs](https://michedelliveneri.github.io/ALMASim/compute.html)

---

## Metadata and Downloads

### Query metadata via TAP

```python
from almasim.services.metadata.tap.service import (
    query_by_science_type,
    InclusionFilters,
    ExclusionFilters,
)

df = query_by_science_type(
    include=InclusionFilters(
        science_keyword=["Galaxies"],
        band=[6, 7],
        public_only=True,
        science_only=True,
    ),
    exclude=ExclusionFilters(solar=True),
)
```

### Download products

```python
from almasim.services.download import resolve_products, run_download_job

products = resolve_products(df["member_ous_uid"].tolist())
run_download_job(products, destination=Path("downloads"), extract_tar=True)
```

Full reference: [Metadata docs](https://michedelliveneri.github.io/ALMASim/metadata.html) · [Downloads docs](https://michedelliveneri.github.io/ALMASim/downloads.html)

---

## Backend Service

The FastAPI backend exposes library services over HTTP and drives the Svelte frontend.

| Endpoint group | Purpose |
|---|---|
| `/api/v1/metadata` | TAP queries and metadata management |
| `/api/v1/simulation` | Simulation job submission and status |
| `/api/v1/download` | Product resolution and download jobs |
| `/api/v1/imaging` | Deconvolution and combination products |
| `/api/v1/visualizer` | Output browsing and product inspection |
| `/health` | Health check |
| `/docs` | Interactive OpenAPI docs (Swagger UI) |

Start locally for development:

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

Full reference: [Frontend docs](https://michedelliveneri.github.io/ALMASim/frontend.html)

---

## Examples

All examples use the `sync` compute backend and require no running scheduler.

| Script | Description |
|---|---|
| [`examples/query_metadata_cli.py`](examples/query_metadata_cli.py) | Query TAP, export metadata and product CSVs |
| [`examples/download_products_cli.py`](examples/download_products_cli.py) | Resolve and download ALMA products |
| [`examples/archive_ms_cli.py`](examples/archive_ms_cli.py) | Unpack ASDMs and apply calibration |
| [`examples/staged_pipeline_cli.py`](examples/staged_pipeline_cli.py) | Full pipeline: query → simulate → ML shard |
| [`examples/imaging_cli.py`](examples/imaging_cli.py) | Synthetic imaging + iterative deconvolution |

```bash
# Query metadata for Band 6 galaxy observations
python examples/query_metadata_cli.py \
  --science-keyword Galaxies --band 6 \
  --save-csv examples/output/metadata.csv

# Run a staged simulation from the first metadata row
python examples/staged_pipeline_cli.py \
  --metadata-csv examples/output/metadata.csv \
  --row-idx 0 --project-name demo \
  --ml-shard-path examples/output/demo.h5

# Iterative deconvolution demo
python examples/imaging_cli.py \
  --output-dir examples/output/imaging --cycles 180 --gain 0.12
```

Notebook equivalents: [`staged_pipeline_notebook.ipynb`](examples/staged_pipeline_notebook.ipynb) · [`query_metadata_notebook.ipynb`](examples/query_metadata_notebook.ipynb) · [`download_products_notebook.ipynb`](examples/download_products_notebook.ipynb)

### End-to-end archive pipeline (Marimo)

[`examples/e2e_archive_pipeline.py`](examples/e2e_archive_pipeline.py) is a reactive [Marimo](https://marimo.io) notebook that covers the full archive workflow interactively: query ALMA metadata → resolve DataLink products → download → unpack ASDMs → apply calibration.

```bash
# Install dev dependencies (includes marimo)
uv sync --group dev

# Interactive editing mode — cells re-run automatically as you edit
marimo edit examples/e2e_archive_pipeline.py

# Read-only app mode — run the pipeline step-by-step via the UI
marimo run examples/e2e_archive_pipeline.py
```

Steps 4 (unpack) and 5 (calibrate) require CASA tools (Linux x86-64 only):

```bash
pip install "almasim[casa]"
```

The notebook saves query filter presets as `.query.json` files so they can be reloaded across sessions.

---

## Documentation

Full documentation: **[michedelliveneri.github.io/ALMASim](https://michedelliveneri.github.io/ALMASim/)**

| Section | Topics |
|---|---|
| [Quick Start](https://michedelliveneri.github.io/ALMASim/quickstart.html) | Installation, first simulation |
| [Simulation](https://michedelliveneri.github.io/ALMASim/simulation.html) | Staged API, SimulationParams, outputs |
| [Interferometry](https://michedelliveneri.github.io/ALMASim/interferometry.html) | UV sampling, baselines, multi-config |
| [Noise](https://michedelliveneri.github.io/ALMASim/noise.html) | PWV-aware noise model |
| [Background Sky](https://michedelliveneri.github.io/ALMASim/background.html) | Additive astrophysical background |
| [Skymodels](https://michedelliveneri.github.io/ALMASim/skymodels.html) | Source models reference |
| [Imaging](https://michedelliveneri.github.io/ALMASim/imaging.html) | Deconvolution, TP+INT combination |
| [Metadata](https://michedelliveneri.github.io/ALMASim/metadata.html) | TAP queries, filters |
| [Downloads](https://michedelliveneri.github.io/ALMASim/downloads.html) | Product download workflow |
| [Compute Backends](https://michedelliveneri.github.io/ALMASim/compute.html) | Sync, Dask, Slurm, Kubernetes |
| [Frontend](https://michedelliveneri.github.io/ALMASim/frontend.html) | Svelte UI workflows |

Build docs locally:

```bash
uv sync --group dev
uv run sphinx-build -b html docs/source docs/build/html
```

---

## Contributing

```bash
git clone https://github.com/MicheleDelliVeneri/ALMASim.git
cd ALMASim
uv sync --group dev
uv run pytest --ignore=illustris_python
uv run ruff check .
uv run ruff format .
```

A release is published automatically when a version tag is pushed:

```bash
# 1. Bump version in pyproject.toml and src/almasim/__version__.py
# 2. Commit and tag
git tag v2.1.11
git push origin v2.1.11
```

The release pipeline then:
1. Validates that the tag matches `pyproject.toml`
2. Runs the full lint + test suite
3. Publishes wheel and sdist to PyPI via OIDC trusted publisher
4. Creates a GitHub Release with auto-generated changelog and attached artifacts
5. Builds and pushes the backend Docker image to GHCR

> **One-time PyPI setup:** register a [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) on PyPI with owner `MicheleDelliVeneri`, repo `ALMASim`, workflow `release.yml`, environment `pypi`.

---

## License

ALMASim is released under the [GNU General Public License v3](LICENSE.md).
