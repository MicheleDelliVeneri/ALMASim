"""Total-power noise follows the simulation seed (MADIV-28, maintainer decision 2026-10-20).

It used to default to np.random.default_rng(0), so every TP run had the same noise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from almasim.services import simulation

pytestmark = pytest.mark.unit


def _stage(seed, n_tp=2):
    stage = MagicMock()
    stage.seed = seed
    stage.interferometer_runs = []
    stage.total_power_runs = [
        {"noise": 0.0, "antenna_diameter_m": 12.0, "config_name": f"tp{i}", "array_type": "TP"}
        for i in range(n_tp)
    ]
    return stage


def _generators(monkeypatch, seed):
    seen = []

    def fake_tp(*args, rng=None, **kwargs):
        seen.append(rng)
        raise StopIteration  # nothing past the TP call matters here

    monkeypatch.setattr(simulation.stages.uin, "simulate_total_power_observation", fake_tp)
    with pytest.raises(StopIteration):
        simulation.simulate_observation(_stage(seed))
    return seen


def test_the_tp_noise_generator_is_seeded_from_the_simulation_seed(monkeypatch):
    rng = _generators(monkeypatch, 42)[0]
    expected = np.random.default_rng(42)
    assert rng.bit_generator.state == expected.bit_generator.state


def test_without_a_seed_runs_get_different_noise(monkeypatch):
    first = _generators(monkeypatch, None)[0].normal(size=4)
    second = _generators(monkeypatch, None)[0].normal(size=4)
    assert not np.array_equal(first, second)


def test_simulation_params_carry_the_seed():
    fields = simulation.SimulationParams.__dataclass_fields__
    assert "seed" in fields and fields["seed"].default is None
