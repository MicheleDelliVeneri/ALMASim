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


class _NoStdout:
    """A Popen whose output was not captured."""

    stdout = None

    def __init__(self, *args, **kwargs):
        pass

    def kill(self):
        pass


def test_a_child_without_captured_output_is_an_error_not_an_assert(tmp_path, monkeypatch):
    """The checks that replaced the asserts still run under ``python -O``."""
    import subprocess
    from collections import deque

    from almasim.services.imaging.wsclean import runner
    from almasim.services.runtime import child

    monkeypatch.setattr(subprocess, "Popen", _NoStdout)
    with pytest.raises(RuntimeError, match="output of wsclean is not captured"):
        with (tmp_path / "log").open("w") as log:
            runner._run_once(
                ["wsclean"], tmp_path, log, {}, None, deque(), lambda m, **k: RuntimeError(m)
            )
    with pytest.raises(RuntimeError, match="output of casa is not captured"):
        child.run_stage_subprocess(["casa"], tmp_path / "stage.log", cwd=tmp_path, env={})
