"""The frontend's package version follows the Python package's.

The release workflow checks the tag against ``pyproject.toml`` only, so the
frontend's ``package.json`` had drifted (2.1.10 against 2.1.16).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_package_json_has_the_pyproject_version():
    with (ROOT / "pyproject.toml").open("rb") as f:
        version = tomllib.load(f)["project"]["version"]
    package = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    assert package["version"] == version
    assert lock["version"] == version
    assert lock["packages"][""]["version"] == version
