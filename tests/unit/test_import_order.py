"""Every services module imports on its own, first, in a fresh interpreter.

Package ``__init__`` re-exports can close import cycles that only show when a
module is the first one imported: ``products.cube_export`` failed that way
(cube_export -> imaging -> products.preview -> interferometry -> core ->
cube_export) while the unit tests, importing in another order, passed.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

import almasim.services

pytestmark = pytest.mark.unit

_MODULES = sorted(
    info.name
    for info in pkgutil.walk_packages(almasim.services.__path__, "almasim.services.")
    if not info.name.rsplit(".", 1)[-1].startswith("_")
)


def test_every_services_module_imports_first():
    script = (
        "import importlib, sys\n"
        "failed = []\n"
        "for name in sys.argv[1:]:\n"
        "    for loaded in [m for m in sys.modules if m.startswith('almasim')]:\n"
        "        del sys.modules[loaded]\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ImportError as exc:\n"
        "        if 'circular import' in str(exc) or 'partially initialized' in str(exc):\n"
        "            failed.append(f'{name}: {exc}')\n"
        "print('\\n'.join(failed))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *_MODULES], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout
