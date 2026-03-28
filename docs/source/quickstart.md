# Quick Start

Install the package in editable mode:

```bash
pip install -e .
```

Use the staged Python API:

```python
from almasim import generate_clean_cube, simulate_observation, image_products, export_results
from almasim.services.simulation import SimulationParams
```

Useful example entry points:

```bash
python examples/staged_pipeline_cli.py --help
python examples/query_metadata_cli.py --help
python examples/download_products_cli.py --help
python examples/imaging_cli.py --help
```

For the subsystem docs, continue with:

- [Simulation](simulation.md)
- [Interferometry](interferometry.md)
- [Imaging and Combination](imaging.md)
- [Metadata](metadata.md)
