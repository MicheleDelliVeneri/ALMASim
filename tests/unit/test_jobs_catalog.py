"""The command catalogue and validation (web console plan, S2.2)."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from almasim.jobs.catalog import catalog, validate
from almasim.jobs.policy import KNOWN_COMMANDS
from almasim.jobs.profile import CommandPolicy, Profile

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def commands() -> dict:
    return {c.name: c for c in catalog()}


def _param(commands, command: str, flag: str) -> dict:
    return next(p for p in commands[command].params if flag in p["flags"])


def test_every_runnable_command_is_described_and_serialisable(commands):
    assert set(commands) == set(KNOWN_COMMANDS)
    json.dumps([asdict(c) for c in commands.values()])
    assert commands["metadata"].executor == "slurm" and commands["metadata"].needs_internet
    assert not commands["image image-from-ms"].needs_internet
    assert commands["image image-from-ms"].summary


def test_the_executor_is_where_the_profile_runs_each_command():
    profile = Profile(executor="server", commands={"predict": CommandPolicy(executor="slurm")})
    executors = {c.name: c.executor for c in catalog(profile)}
    assert executors["metadata"] == "server" and executors["predict"] == "slurm"


def test_types_ranges_and_choices_come_from_the_cli(commands):
    cores = _param(commands, "image image-from-ms", "--num-cores")
    assert cores["type"] == "integer" and cores["min"] == 1
    assert _param(commands, "image image-from-ms", "--cube")["type"] == "boolean"
    backend = _param(commands, "image image-from-ms", "--backend")
    assert backend["type"] == "choice" and backend["choices"] == ["sync", "slurm"]
    assert "--postprocess-backend" not in backend["flags"]  # deprecated aliases stay hidden
    first = commands["image image-from-ms"].params[0]
    assert first["kind"] == "argument" and first["type"] == "path"


def test_the_former_free_text_choices_are_listed(commands):
    """CLI-4: these were plain strings checked by hand."""
    source = _param(commands, "simulation", "--source-type")["choices"]
    assert {"point", "gaussian", "galaxy-zoo"} <= set(source)
    assert _param(commands, "simulation", "--save-mode")["choices"] == [
        "memory",
        "npz",
        "h5",
        "fits",
    ]
    assert _param(commands, "simulation", "--backend")["choices"] == ["sync", "local"]
    assert _param(commands, "simulation", "--imaging-algorithm")["choices"] == ["legacy"]
    products = _param(commands, "products download", "--product-filter")["choices"]
    assert products[0] == "all" or "all" in products
    assert {"raw", "calibration"} <= set(products)


def test_clean_takes_arguments_it_does_not_declare(commands):
    assert commands["clean"].extra_args
    assert not commands["image clean-quality"].extra_args


@pytest.mark.parametrize(
    "argv",
    [
        ["image", "clean-quality", "/out"],
        ["metadata", "--band", "3"],
        ["metadata", "query", "--band", "3"],
        ["simulation", "--metadata-path", "m.csv", "--save-mode", "NPZ"],
        ["image", "image-from-ms", "--help"],
    ],
)
def test_valid_command_lines_pass(argv):
    assert validate(argv) == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["simulation", "--metadata-path", "m.csv", "--save-mode", "pickle"], "pickle"),
        (["image", "image-from-ms", "p.csv", "out", "--num-cores", "0"], "num-cores"),
        (["image", "clean-quality", "/out", "--no-such-option"], "no-such-option"),
        (["simulation"], "metadata-path"),
        (["rm", "-rf"], "unknown command"),
    ],
)
def test_invalid_command_lines_say_why(argv, message):
    problems = validate(argv)
    assert problems and message in problems[0]
