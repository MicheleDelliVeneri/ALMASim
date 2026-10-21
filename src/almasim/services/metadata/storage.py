"""Writing metadata query results."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_dataframe_csv(dataframe: Any, destination: Path) -> None:
    """Persist a dataframe-like object to CSV without an index column."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
