# ALMASim Test Suite

This directory contains the test suite for ALMASim, organized into three categories:

## Test Structure

### Unit Tests (`tests/unit/`)
Tests for individual functions and classes in isolation. Each file is named
after the module it tests, so the tree follows `src/almasim`:
- `test_cli_<group>.py` - one per CLI command group (`cli/<group>.py`);
  `test_cli_main.py` covers the entry point, `clean` and the `cli_*` aliases
- `test_archive_*.py` - `services/archive` (`test_archive_pipeline.py`: the per-UID stages)
- `test_imaging_*.py`, `test_interferometry_*.py`, `test_simulation_*.py`,
  `test_skymodels_*.py`, `test_astro_*.py` - the matching services packages

There are no tests at the top of `tests/`: every test belongs to one of the
three categories.

### Component Tests (`tests/component/`)
Tests for integration between modules:
- `test_metadata_queries.py` - Metadata query integration
- `test_spectral_processing.py` - Spectral data processing
- `test_skymodel_integration.py` - Sky model integration
- `test_simulation_workflow.py` - Simulation workflow
- `test_skymodel_workflow.py`, `test_interferometer_workflow.py` - end-to-end sky model and interferometer runs

### Integration Tests (`tests/integration/`)
Backend API integration tests:
- `test_backend_api.py` - FastAPI endpoint tests
- `test_backend_simulation.py` - Backend simulation service tests
- `test_tap_live_query.py` - a live ALMA TAP query (marked `network`)

## Running Tests

Run all tests:
```bash
pytest
```

Run only unit tests:
```bash
pytest tests/unit/
```

Run only component tests:
```bash
pytest tests/component/
```

Run only integration tests:
```bash
pytest tests/integration/
```

Run tests with markers:
```bash
pytest -m unit
pytest -m component
pytest -m integration
pytest -m "not slow"  # Skip slow tests
```

## Test Markers

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.component` - Component tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.network` - Tests requiring network access

## Fixtures

Common fixtures are defined in `conftest.py`:
- `repo_root` - Repository root directory
- `main_dir` - Main almasim directory
- `test_data_dir` - Test data directory
- `sample_metadata_row` - Sample metadata row for testing


