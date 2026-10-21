"""The web app's copy of the console's OpenAPI schema is current (MADIV-28 T7.4, U3.1).

The web app's API types (``schema.d.ts``) are generated from
``frontend/src/lib/api/v2/openapi.json``; after a schema change run
``python scripts/generate_openapi.py`` and ``npm run gen:api``. The schema is
built in a fresh interpreter, as the script runs it.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
STORED = ROOT / "frontend" / "src" / "lib" / "api" / "v2" / "openapi.json"


def test_openapi_json_matches_the_console():
    pytest.importorskip("fastapi")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_openapi.py"), "--stdout"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    diff = "\n".join(
        difflib.unified_diff(
            STORED.read_text(encoding="utf-8").splitlines(),
            result.stdout.splitlines(),
            lineterm="",
            n=1,
        )
    )
    hint = "run: python scripts/generate_openapi.py && (cd frontend && npm run gen:api)"
    assert not diff, f"{hint}\n{diff[:3000]}"
