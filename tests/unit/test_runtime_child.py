"""call_in_child runs any library function in a new process (MADIV-28 T2.4)."""

from __future__ import annotations

import os

import pytest

from almasim.services.runtime.child import ChildFailed, call_in_child, error_line

pytestmark = pytest.mark.unit


def test_the_target_is_named_by_module_and_function():
    assert call_in_child("json:dumps", {"obj": [1, 2]}, what="dump", env=os.environ) == "[1, 2]"


def test_a_failing_child_raises_with_its_stderr(tmp_path):
    log = tmp_path / "task.log"
    with pytest.raises(
        ChildFailed, match="parse exited with return code 1: .*JSONDecodeError"
    ) as e:
        call_in_child("json:loads", {"s": "not json"}, what="parse", env=os.environ, log_path=log)
    assert e.value.returncode == 1 and "Traceback" in e.value.stderr
    assert "JSONDecodeError" in log.read_text()


def test_a_timeout_names_the_step():
    with pytest.raises(RuntimeError, match="nap killed after exceeding the 0s task timeout"):
        call_in_child(
            "subprocess:run",
            {"args": ["sleep", "5"]},
            what="nap",
            env=os.environ,
            timeout=0.2,
            timeout_noun="task timeout",
        )


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["ok", "RuntimeError: boom", "done"], "RuntimeError: boom"),
        (["terminate called", "  what():  bad alloc"], "what():  bad alloc"),
        (["nothing here"], ""),
    ],
)
def test_error_line_finds_the_last_reason(lines, expected):
    assert error_line(lines) == expected
