"""Route handlers that do blocking work are plain functions (MADIV-28 T1.8).

FastAPI runs a ``def`` handler in its threadpool but an ``async def`` handler
on the event loop, so an ``async def`` that does blocking I/O or computation
freezes every other request, the simulation WebSocket included. Only the
handlers that really await something may be coroutines. Checked statically,
so the backend's dependencies are not needed.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROUTERS = Path(__file__).resolve().parents[2] / "backend" / "app" / "api" / "v1" / "routers"
ASYNC_HANDLERS = {"websocket_status", "integrate_datacube"}


def _coroutines(path: Path) -> dict[str, bool]:
    """Async top-level functions of a module and whether each awaits anything."""
    tree = ast.parse(path.read_text())
    return {
        node.name: any(
            isinstance(inner, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for inner in ast.walk(node)
        )
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
    }


def test_only_awaiting_handlers_are_coroutines():
    found = {}
    for path in sorted(ROUTERS.glob("*.py")):
        found.update(_coroutines(path))
    assert set(found) == ASYNC_HANDLERS
    assert all(found.values()), "every coroutine handler awaits something"
