"""Pytest configuration and shared fixtures."""

import faulthandler
import os

# Fix astropy logging conflict with pytest
# This must be done before importing astropy
import sys
import warnings
from pathlib import Path

import pytest


# Monkey-patch astropy logger to handle pytest conflicts
def _patch_astropy_logger():
    """Patch astropy logger to avoid conflicts with pytest."""
    try:
        import astropy.logger

        # Override the disable_warnings_logging method to be a no-op
        original_disable = astropy.logger.disable_warnings_logging

        def safe_disable(*args, **kwargs):
            try:
                return original_disable(*args, **kwargs)
            except Exception:
                pass  # Ignore errors when disabling warnings

        astropy.logger.disable_warnings_logging = safe_disable
    except (ImportError, AttributeError):
        pass  # astropy not available or already configured


# Apply patch if astropy hasn't been imported yet
if "astropy" not in sys.modules:
    # Set up environment before astropy import
    os.environ.setdefault("ASTROPY_LOGGING", "0")
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="astropy")
else:
    _patch_astropy_logger()

faulthandler.enable()
os.environ["LC_ALL"] = "C"


def pytest_collection_modifyitems(config, items):
    """Skip network-marked tests unless explicitly enabled."""
    if os.environ.get("ALMASIM_RUN_NETWORK_TESTS") == "1":
        return

    skip_network = pytest.mark.skip(
        reason="network tests disabled; set ALMASIM_RUN_NETWORK_TESTS=1 to enable",
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture
def repo_root():
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def main_dir(repo_root):
    """Return the main almasim directory."""
    return repo_root / "src" / "almasim"


@pytest.fixture
def test_data_dir(repo_root):
    """Return the test data directory."""
    return repo_root / "data"


@pytest.fixture
def sample_metadata_row(test_data_dir):
    """Load a sample metadata row for testing."""
    import pandas as pd

    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    return metadata.iloc[0]
