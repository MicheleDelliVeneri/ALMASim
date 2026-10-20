"""WSCleanCommand: rewriting WSClean command lines (MADIV-28 T4.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from almasim.services.imaging.wsclean.command import (
    WSCleanCommand,
    _dirty_command,
    _selfcal_command,
    _with_data_column,
    continuum_restart_commands,
)

pytestmark = pytest.mark.unit

BASE = [
    "wsclean",
    "-name",
    "/out/wsclean",
    "-auto-mask",
    "3",
    "-niter",
    "1000",
    "-no-update-model-required",
    "-field",
    "0",
    "/data/x.ms",
]


def test_parse_keeps_program_options_and_ms_apart():
    command = WSCleanCommand.parse(BASE)
    assert command.program == "wsclean" and command.ms == "/data/x.ms"
    assert command.argv() == BASE
    assert command.value("-niter") == "1000" and command.value("-spws") is None


def test_parse_needs_a_program_and_an_ms():
    with pytest.raises(ValueError, match="program and an MS"):
        WSCleanCommand.parse(["wsclean"])


def test_without_drops_flags_with_their_values():
    command = WSCleanCommand.parse(BASE).without("-auto-mask", "-no-update-model-required")
    assert command.argv() == [
        "wsclean",
        "-name",
        "/out/wsclean",
        "-niter",
        "1000",
        "-field",
        "0",
        "/data/x.ms",
    ]


def test_without_refuses_a_flag_of_unknown_arity():
    with pytest.raises(ValueError, match="unknown number of values"):
        WSCleanCommand.parse(BASE).without("-field")


def test_with_option_replaces_in_place_or_appends_before_the_ms():
    command = WSCleanCommand.parse(BASE)
    assert command.with_option("-niter", 0).argv()[5:7] == ["-niter", "0"]
    appended = command.with_option("-data-column", "DATA").argv()
    assert appended[-3:] == ["-data-column", "DATA", "/data/x.ms"]
    with pytest.raises(ValueError, match="takes 1 value"):
        command.with_option("-niter")


def test_model_update_swaps_the_flag_in_place():
    command = WSCleanCommand.parse(BASE)
    assert command.with_model_update(True).argv()[7] == "-update-model-required"
    assert command.with_model_update(False) == command
    bare = WSCleanCommand("wsclean", (), "x.ms")
    assert bare.with_model_update(True).argv() == ["wsclean", "-update-model-required", "x.ms"]


def test_continued_and_substituted():
    command = WSCleanCommand("wsclean", ("-temp-dir", "__SCRATCH__"), "__MS__")
    done = command.continued().substituted("__SCRATCH__", Path("/s")).substituted("__MS__", "y.ms")
    assert done.argv() == ["wsclean", "-temp-dir", "/s", "-continue", "y.ms"]


def test_helpers_keep_their_command_lines():
    """The helpers built on WSCleanCommand produce the same lists as before T4.3."""
    out = Path("/o")
    assert _selfcal_command(BASE, out, "selfcal-0") == [
        "wsclean",
        "-name",
        "/o/selfcal-0",
        "-auto-mask",
        "3",
        "-niter",
        "1000",
        "-update-model-required",
        "-field",
        "0",
        "/data/x.ms",
    ]
    assert _dirty_command([*BASE[:-1], "-continue", BASE[-1]], out, "d", "CORRECTED_DATA") == [
        "wsclean",
        "-name",
        "/o/d",
        "-field",
        "0",
        "-niter",
        "0",
        "-no-update-model-required",
        "-data-column",
        "CORRECTED_DATA",
        "/data/x.ms",
    ]
    assert _with_data_column(BASE, "DATA") == [*BASE[:-1], "-data-column", "DATA", BASE[-1]]
    first, second = continuum_restart_commands(BASE)
    assert first[7] == "-update-model-required" and second == [*first[:-1], "-continue", BASE[-1]]
