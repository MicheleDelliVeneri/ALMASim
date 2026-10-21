"""The frontend's copies of the OpenAPI schemas are current (MADIV-28 T7.4, U3.1).

The frontend's API types (``schema.d.ts``) are generated from
``frontend/src/lib/api/openapi.json`` (API v1, the old backend) and
``frontend/src/lib/api/v2/openapi.json`` (API v2, the web console); after a
schema change run ``python scripts/generate_openapi.py`` and
``npm run gen:api``. The schemas are built in a fresh interpreter: other tests
stub backend modules the routes depend on.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "frontend" / "src" / "lib" / "api"
STORED = {"v1": API / "openapi.json", "v2": API / "v2" / "openapi.json"}


@pytest.mark.parametrize("version", sorted(STORED))
def test_openapi_json_matches_the_server(version):
    pytest.importorskip("fastapi")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_openapi.py"), "--stdout", version],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    diff = "\n".join(
        difflib.unified_diff(
            STORED[version].read_text(encoding="utf-8").splitlines(),
            result.stdout.splitlines(),
            lineterm="",
            n=1,
        )
    )
    hint = "run: python scripts/generate_openapi.py && (cd frontend && npm run gen:api)"
    assert not diff, f"{hint}\n{diff[:3000]}"
