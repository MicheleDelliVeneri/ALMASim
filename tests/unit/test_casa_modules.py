"""CASA is imported through services.casa.tools()/tasks() (MADIV-28 T2.5)."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from almasim.services import casa

pytestmark = pytest.mark.unit


def test_tools_and_tasks_return_the_installed_modules(monkeypatch):
    fake_tools, fake_tasks = ModuleType("casatools"), ModuleType("casatasks")
    monkeypatch.setitem(sys.modules, "casatools", fake_tools)
    monkeypatch.setitem(sys.modules, "casatasks", fake_tasks)
    assert casa.tools() is fake_tools
    assert casa.tasks() is fake_tasks


def test_a_missing_casa_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "casatasks", None)  # import raises ImportError
    with pytest.raises(ImportError, match=r"almasim\[casa\]"):
        casa.tasks()
