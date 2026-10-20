"""Library code has no runtime asserts and chains the exceptions it raises (MADIV-28 T4.7).

``python -O`` strips ``assert``, so a check written as one silently stops
running; an exception raised while handling another without ``from`` hides
what caused it (or, with ``from None``, says so deliberately).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC = Path(__file__).resolve().parents[2] / "src" / "almasim"


def _modules():
    for path in sorted(SRC.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def test_no_assert_statements_in_library_code():
    found = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path, tree in _modules()
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]
    assert found == []


def test_exceptions_raised_while_handling_another_are_chained():
    found = []
    for path, tree in _modules():
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            for node in ast.walk(handler):
                if not isinstance(node, ast.Raise) or node.exc is None or node.cause is not None:
                    continue
                if isinstance(node.exc, ast.Name) and node.exc.id == handler.name:
                    continue  # re-raising the caught exception
                if ast.unparse(node.exc).startswith("typer.Exit"):
                    continue  # the CLI's exit code, not an error
                found.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert found == []
