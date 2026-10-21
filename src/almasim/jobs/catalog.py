"""Every runnable command, described from the CLI itself (web console plan, S2.2).

The web console builds its forms from :func:`catalog`, which walks the Typer
app the way ``scripts/generate_cli_reference.py`` does: each parameter's
kind, type, choices, range, default, help and help panel. A form can then
never offer an option the CLI lacks, and new commands appear by themselves.
:func:`validate` parses a command line with the CLI's own parser, so the
console never submits one the CLI would reject.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import Any

import typer

from almasim.cli._options import DEPRECATED_FLAGS

from .model import command_name
from .policy import KNOWN_COMMANDS, NETWORK_COMMANDS


@dataclass(frozen=True)
class CommandInfo:
    name: str
    summary: str
    help: str
    executor: str  # by default; a profile can move a command (policy.plan_job)
    needs_internet: bool
    params: list[dict[str, Any]]
    extra_args: bool  # takes arguments it does not declare (clean passes them to WSClean)


def _json_default(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    if callable(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _type_info(param: Any) -> dict[str, Any]:
    """The parameter's type, by duck typing: typer ships its own copy of click."""
    t = param.type
    name = getattr(t, "name", "")
    if getattr(param, "is_flag", False) or name == "boolean":
        return {"type": "boolean"}
    choices = getattr(t, "choices", None)
    if choices is not None:
        return {"type": "choice", "choices": [str(_json_default(c)) for c in choices]}
    if name in ("int range", "float range", "int", "integer", "float"):
        return _number_info(t, name)
    if hasattr(t, "dir_okay") and hasattr(t, "file_okay"):
        return _path_info(t)
    return {"type": "string"}


def _number_info(t: Any, name: str) -> dict[str, Any]:
    info: dict[str, Any] = {"type": "number" if "float" in name else "integer"}
    for bound in ("min", "max"):
        if getattr(t, bound, None) is not None:
            info[bound] = getattr(t, bound)
    return info


def _path_info(t: Any) -> dict[str, Any]:
    if not t.dir_okay:
        kind = "file"
    elif not t.file_okay:
        kind = "folder"
    else:
        kind = "path"
    return {"type": "path", "path_kind": kind, "must_exist": bool(getattr(t, "exists", False))}


def describe_param(param: Any) -> dict[str, Any] | None:
    """One parameter as the console's form needs it; None for hidden ones."""
    if getattr(param, "hidden", False) or param.name in (None, "help"):
        return None
    is_option = param.param_type_name == "option"
    info: dict[str, Any] = {
        "name": param.name,
        "kind": "option" if is_option else "argument",
        # Deprecated spellings still parse, but forms offer only the current one.
        "flags": [f for f in param.opts if f not in DEPRECATED_FLAGS],
        "negative_flags": list(param.secondary_opts),
        "required": bool(param.required),
        "multiple": bool(getattr(param, "multiple", False) or param.nargs == -1),
        "default": _json_default(param.default),
        "help": (getattr(param, "help", None) or "").strip(),
        "panel": getattr(param, "rich_help_panel", None),
    }
    info.update(_type_info(param))
    return info


def _leaf_commands() -> dict[str, Any]:
    from almasim import cli

    found: dict[str, Any] = {}
    for name, module_name, app_name, _help in cli._GROUPS:
        group = typer.main.get_command(getattr(import_module(module_name, "almasim.cli"), app_name))
        subcommands = getattr(group, "commands", None)  # typer's own click: duck-type groups
        if isinstance(subcommands, dict):
            for sub, command in subcommands.items():
                if not command.hidden:
                    found[f"{name} {sub}"] = command
        else:
            found[name] = group
    clean = typer.Typer(add_completion=False)
    clean.command(  # as almasim.cli registers it: unknown arguments go to WSClean
        "clean", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
    )(import_module("almasim.cli.clean").clean_command)
    found["clean"] = typer.main.get_command(clean)
    doctor = typer.Typer(add_completion=False)
    doctor.command("doctor")(cli.doctor_command)
    found["doctor"] = typer.main.get_command(doctor)
    return {name: cmd for name, cmd in found.items() if name in KNOWN_COMMANDS}


@cache
def _commands() -> dict[str, Any]:
    return _leaf_commands()


def catalog() -> list[CommandInfo]:
    """Every command the console can run, in the CLI's order."""
    out = []
    for name, command in _commands().items():
        params = [p for p in (describe_param(x) for x in command.params) if p is not None]
        out.append(
            CommandInfo(
                name=name,
                summary=(command.short_help or (command.help or "").split("\n\n")[0]).strip(),
                help=(command.help or "").strip(),
                executor="slurm",
                needs_internet=name in NETWORK_COMMANDS,
                params=params,
                extra_args=bool(command.context_settings.get("allow_extra_args")),
            )
        )
    return out


def _arguments_after_command(argv: list[str], name: str, command: Any) -> list[str]:
    rest = argv[len(name.split()) :]
    # Single-command groups accept their subcommand word too (`metadata query`).
    if " " not in name and rest and rest[0] == command.name and command.name != name:
        rest = rest[1:]
    return rest


def validate(argv: list[str]) -> list[str]:
    """The problems the CLI's parser finds in ``almasim <argv>``; empty when it would run."""
    name = command_name(argv)
    command = _commands().get(name)
    if command is None:
        return [f"unknown command {' '.join(argv[:2])!r}"]
    try:
        ctx = command.make_context(f"almasim {name}", _arguments_after_command(argv, name, command))
    except Exception as exc:
        if type(exc).__name__ == "Exit":
            return []  # --help and the like: valid, if useless as a job
        format_message = getattr(exc, "format_message", None)
        if format_message is None:
            raise
        return [format_message()]
    ctx.close()
    return []
