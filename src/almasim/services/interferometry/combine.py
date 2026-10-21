"""Shared steps of combining per-configuration runs (interferometric and total power)."""

from __future__ import annotations

from typing import Any

import numpy as np


def single_run_result(per_config_results: list[dict[str, Any]], what: str) -> dict[str, Any] | None:
    """The combined result when there is one run (or None when there are several).

    Raises ``ValueError`` for an empty list.
    """
    if not per_config_results:
        raise ValueError(f"Cannot combine an empty {what} result list")
    if len(per_config_results) > 1:
        return None
    combined = dict(per_config_results[0])
    combined["per_config_results"] = per_config_results
    combined["combined_config_count"] = 1
    return combined


def normalized_config_weights(n_runs: int, config_weights: list[float] | None) -> np.ndarray:
    """Per-run weights summing to 1: negatives clipped, all-zero treated as equal."""
    if config_weights is None:
        config_weights = [1.0] * n_runs
    weights = np.clip(np.asarray(config_weights, dtype=float), 0.0, None)
    if not np.any(weights):
        weights = np.ones_like(weights)
    return weights / np.sum(weights)
