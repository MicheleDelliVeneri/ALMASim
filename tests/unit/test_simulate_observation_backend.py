"""simulate_observation without a compute backend images in-process (MADIV-28 T6.5).

The backend's simulation service passes ``compute_backend=None`` unless the
request names one, and the Interferometer then failed on ``None.scatter``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from almasim.services import simulation
from almasim.services.compute.sync import SyncBackend

pytestmark = pytest.mark.unit


def _backend_used(monkeypatch, compute_backend):
    seen = []

    def fake_interferometer(*, backend, **kwargs):
        seen.append(backend)
        raise StopIteration  # nothing past construction matters here

    monkeypatch.setattr(simulation.uin, "Interferometer", fake_interferometer)
    stage = MagicMock()
    stage.interferometer_runs = [{}]
    stage.total_power_runs = []
    with pytest.raises(StopIteration):
        simulation.simulate_observation(stage, compute_backend=compute_backend)
    return seen[0]


def test_no_backend_falls_back_to_the_in_process_one(monkeypatch):
    assert isinstance(_backend_used(monkeypatch, None), SyncBackend)


def test_a_given_backend_is_used_as_is(monkeypatch):
    backend = MagicMock()
    assert _backend_used(monkeypatch, backend) is backend
