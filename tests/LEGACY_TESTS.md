# Legacy Test Files - Review Status

## Files to Keep (Still Relevant)

### `test_services_simulation.py`
**Status**: ✅ KEEP - Tests SimulationParams creation
- `test_from_metadata_row_populates_required_fields` - Tests parameter extraction
- `test_from_metadata_row_honours_overrides` - Tests parameter overrides
- **Reason**: These are unit tests for the SimulationParams class, which is still used

### `test_astro.py`
**Status**: ✅ KEEP - Tests astro utility functions
- `test_luminosity_functions` - Tests line info, redshift, snapshot functions
- **Reason**: Unit tests for astro utilities, still relevant

### `test_alma.py`
**Status**: ✅ KEEP - Tests metadata queries and antenna functions
- `test_query` - Tests TAP queries
- `test_alma_functions` - Tests antenna configuration functions
- **Reason**: Component tests for metadata and antenna functionality

## Files to Refactor/Remove

### `test_skymodels.py`
**Status**: ⚠️ REFACTOR - Mixed concerns, should be split
- `test_skymodel_generation` - Tests full workflow (sky model + serendipitous)
- **Action**: Split into:
  - Component test for sky model generation (already in `tests/component/test_skymodel_generation.py`)
  - Component test for serendipitous sources (already covered)

### `test_interferometer.py`
**Status**: ⚠️ REFACTOR - Tests full workflow
- `test_interferometer_runs` - Tests full simulation workflow
- **Action**: Split into:
  - Component test for interferometer initialization (already in `tests/component/test_interferometric_simulation.py`)
  - Component test for interferometer execution (already in `tests/component/test_interferometric_simulation.py`)

## New Component Test Structure

### ✅ Created: `tests/component/test_datasets.py`
- Tests dataset download functionality
- Tests RemoteMachine dataclass
- Tests directory structure creation

### ✅ Created: `tests/component/test_skymodel_generation.py`
- Tests pointlike, Gaussian, diffuse sky models
- Tests serendipitous source insertion
- Tests datacube creation and header generation

### ✅ Created: `tests/component/test_interferometric_simulation.py`
- Tests interferometer initialization
- Tests baseline preparation
- Tests full simulation run
- Tests progress signals
- Tests different save modes

### ✅ Created: `tests/component/test_metadata_gathering.py`
- Tests metadata query functions
- Tests metadata loading and normalization

## Recommendation

1. **Keep**: `test_services_simulation.py`, `test_astro.py`, `test_alma.py`
2. **Remove**: `test_skymodels.py`, `test_interferometer.py` (functionality moved to component tests)
3. **Update**: Any remaining references to use new component test structure


