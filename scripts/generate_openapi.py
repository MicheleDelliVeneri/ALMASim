"""Write the console's OpenAPI schema for the web app's generated types (MADIV-28 T7.4, U3.1).

python scripts/generate_openapi.py            # writes frontend/src/lib/api/v2/openapi.json
python scripts/generate_openapi.py --stdout   # prints it instead
cd frontend && npm run gen:api                # regenerates schema.d.ts from it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "frontend" / "src" / "lib" / "api" / "v2" / "openapi.json"


def openapi_schema() -> dict:
    """The OpenAPI document of the console (``almasim serve``, API v2)."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from almasim.jobs.profile import Profile
    from almasim.web.app import create_app

    return create_app("schema", static_dir=None, profile=Profile()).openapi()


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    if "--stdout" in sys.argv:
        sys.stdout.write(render(openapi_schema()))
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(render(openapi_schema()), encoding="utf-8")
        print(f"wrote {TARGET.relative_to(ROOT)}")
