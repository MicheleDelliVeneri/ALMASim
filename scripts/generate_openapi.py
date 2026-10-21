"""Write the OpenAPI schemas for the frontend's generated types (MADIV-28 T7.4, U3.1).

There are two: the old backend's API v1 (``backend/app``) and the web
console's API v2 (``almasim.web``, which ``almasim serve`` runs).

python scripts/generate_openapi.py                   # writes both openapi.json files
python scripts/generate_openapi.py --stdout v1|v2    # prints one instead
cd frontend && npm run gen:api                       # regenerates both schema.d.ts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "frontend" / "src" / "lib" / "api"
TARGETS = {"v1": API / "openapi.json", "v2": API / "v2" / "openapi.json"}


def _on_path(*paths: Path) -> None:
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def openapi_schema(version: str = "v1") -> dict:
    """The OpenAPI document of API ``version`` ("v1" or "v2")."""
    if version == "v2":
        _on_path(ROOT / "src")
        from almasim.jobs.profile import Profile
        from almasim.web.app import create_app

        return create_app("schema", static_dir=None, profile=Profile()).openapi()
    _on_path(ROOT / "backend", ROOT / "src")
    from app.main import app

    return app.openapi()


def render(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    if "--stdout" in sys.argv:
        args = sys.argv[sys.argv.index("--stdout") + 1 :]
        sys.stdout.write(render(openapi_schema(args[0] if args else "v1")))
    else:
        for version, target in TARGETS.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render(openapi_schema(version)), encoding="utf-8")
            print(f"wrote {target.relative_to(ROOT)}")
