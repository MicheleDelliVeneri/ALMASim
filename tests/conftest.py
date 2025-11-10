"""Pytest configuration and shared fixtures."""
import os
import faulthandler
from pathlib import Path

import pytest

faulthandler.enable()
os.environ["LC_ALL"] = "C"


@pytest.fixture
def repo_root():
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def main_dir(repo_root):
    """Return the main almasim directory."""
    return repo_root / "almasim"


@pytest.fixture
def test_data_dir(repo_root):
    """Return the test data directory."""
    return repo_root / "almasim" / "metadata"


@pytest.fixture
def sample_metadata_row(test_data_dir):
    """Load a sample metadata row for testing."""
    import pandas as pd
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    return metadata.iloc[0]


