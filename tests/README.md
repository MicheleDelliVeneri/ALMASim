# ALMASim Test Suite

This directory contains the test suite for ALMASim, organized into three categories:

## Test Structure

### Unit Tests (`tests/unit/`)
Tests for individual functions and classes in isolation:
- `test_astro_*.py` - Astronomical utility functions
- `test_interferometry_*.py` - Interferometry utilities
- `test_skymodels_*.py` - Sky model classes
- `test_services_utils.py` - Service utilities

### Component Tests (`tests/component/`)
Tests for integration between modules:
- `test_metadata_queries.py` - Metadata query integration
- `test_spectral_processing.py` - Spectral data processing
- `test_skymodel_integration.py` - Sky model integration
- `test_simulation_workflow.py` - Simulation workflow

### Integration Tests (`tests/integration/`)
Backend API integration tests:
- `test_backend_api.py` - FastAPI endpoint tests
- `test_backend_simulation.py` - Backend simulation service tests

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


