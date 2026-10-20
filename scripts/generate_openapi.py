"""Write the backend's OpenAPI schema for the frontend's generated types (MADIV-28 T7.4).

    python scripts/generate_openapi.py            # writes frontend/src/lib/api/openapi.json
    python scripts/generate_openapi.py --stdout   # prints it instead
    cd frontend && npm run gen:api                # regenerates schema.d.ts from it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "frontend" / "src" / "lib" / "api" / "openapi.json"


def openapi_schema() -> dict:
    """The backend application's OpenAPI document."""
    for path in (ROOT / "backend", ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from app.main import app

    return app.openapi()


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    if "--stdout" in sys.argv:
        sys.stdout.write(render(openapi_schema()))
    else:
        TARGET.write_text(render(openapi_schema()), encoding="utf-8")
        print(f"wrote {TARGET.relative_to(ROOT)}")
