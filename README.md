# ALMASim

ALMASim is a library-first environment for simulating ALMA observations, exploring ALMA metadata, downloading products, and building ML-ready radio/mm datasets.

It is built around reusable Python services in [`src/almasim`](src/almasim), with a FastAPI backend in [`backend`](backend), a Svelte frontend in [`frontend`](frontend), and runnable examples in [`examples`](examples).

## Core Capabilities

- Query ALMA metadata from TAP and cache or export the results.
- Download ALMA products through a Python-first download service.
- Run staged simulations through:
  - `generate_clean_cube()`
  - `simulate_observation()`
  - `image_products()`
  - `export_results()`
- Run full end-to-end simulations with `run_simulation()`.
- Simulate interferometric and total-power products for single-pointing ALMA observations.
- Build multi-configuration single-pointing plans from metadata rows.
- Use PWV-aware channel noise models.
- Generate clean sky cubes from several skymodel families.
- Keep the science target at phase center by default, with optional explicit source offsets.
- Add optional ALMA-band background skies:
  - faint dusty galaxies
  - diffuse dusty background
  - combined background
- Export image-domain products including interferometric, TP, and TP+INT combination cubes.
- Run iterative CLEAN-style deconvolution with resumable state.
- Export ML-ready HDF5 shards containing clean cube, dirty cube, dirty visibilities, UV mask cube, and metadata.
- Use local, synchronous, Dask, Slurm, or Kubernetes-oriented compute backends.
- Run simulations from the frontend, CLI scripts, notebooks, or direct Python code.

## Main Interfaces

- Python API: [`src/almasim/__init__.py`](src/almasim/__init__.py)
- Simulation services: [`src/almasim/services/simulation.py`](src/almasim/services/simulation.py)
- Interferometry services: [`src/almasim/services/interferometry`](src/almasim/services/interferometry)
- Imaging services: [`src/almasim/services/imaging`](src/almasim/services/imaging)
- Metadata services: [`src/almasim/services/metadata`](src/almasim/services/metadata)
- Download service: [`src/almasim/services/download.py`](src/almasim/services/download.py)
- Frontend routes: [`frontend/src/routes`](frontend/src/routes)

## Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Run the staged Python API:

```python
from almasim import generate_clean_cube, simulate_observation, image_products, export_results
from almasim.services.simulation import SimulationParams
```

Run the examples:

```bash
python examples/staged_pipeline_cli.py --help
python examples/query_metadata_cli.py --help
python examples/download_products_cli.py --help
python examples/imaging_cli.py --help
```

Start the frontend/backend stack with Docker Compose or the repo’s local development workflow.

## Documentation

- [Documentation Index](docs/README.md)
- [Simulation](docs/simulation.md)
- [Interferometry](docs/interferometry.md)
- [Noise](docs/noise.md)
- [Background Sky](docs/background.md)
- [Skymodels](docs/skymodels.md)
- [Imaging and Combination](docs/imaging.md)
- [Metadata](docs/metadata.md)
- [Downloads](docs/downloads.md)
- [Compute Backends](docs/compute.md)
- [Frontend Workflows](docs/frontend.md)
- [Examples](examples/README.md)
- [SimALMA Fidelity Plan](docs/simalma_fidelity_plan.md)

## Repository Layout

- `src/almasim`: library code
- `backend`: FastAPI backend
- `frontend`: Svelte UI
- `examples`: CLI and notebook examples
- `docs`: markdown documentation and planning notes
- `tests`: unit, component, and integration tests
- `data`: local datasets and sample metadata

## Current Design Direction

ALMASim is intentionally moving toward a Python-first architecture:

- the library layer owns the domain logic
- the backend acts as an adapter over library services
- the frontend orchestrates workflows and visualization
- CLI scripts and notebooks use the same staged services directly

## Notes

- Single-pointing fidelity is the current focus.
- Multi-pointing mosaics are not the current target.
- Auto-estimation features such as cube size and output footprint are intended as preflight guidance, not exact disk forecasts.
- Background sky simulation is additive and optional.
