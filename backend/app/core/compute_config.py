"""Check the compute-backend settings a request may pass (MADIV-28 W0.4).

``compute_backend_config`` used to reach the backend constructors, and through
``SlurmBackend`` the dask-jobqueue ``SLURMCluster``, as raw keyword arguments.
A request could therefore add job directives or a job-script prologue, and a
partition name with a newline in it would have written its own ``#SBATCH``
lines. Each backend now takes only the settings listed here, each checked
against a pattern or a range, and anything else is refused with HTTP 400. Patterns are
matched whole (`fullmatch`): `$` alone would let a value end in a newline.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.core.path_utils import require_allowed_scheduler, resolve_safe_path

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")
_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}$")
_WALLTIME = re.compile(r"^(?:\d{1,3}-)?\d{1,3}:\d{2}:\d{2}$")
_MEMORY = re.compile(r"^\d{1,7}(?:\.\d{1,3})?\s?(?:[KMGT]i?B?|[KMGT]B)?$", re.IGNORECASE)
_K8S_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}$")
_K8S_QUANTITY = re.compile(r"^\d+(?:\.\d+)?(?:m|[KMGTPE]i?)?$")


def _bad(message: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def _text(pattern: re.Pattern[str], what: str) -> Callable[[str, Any], Any]:
    def check(key: str, value: Any) -> Any:
        if not isinstance(value, str) or not pattern.fullmatch(value):
            raise _bad(f"compute_backend_config.{key} must be {what}")
        return value

    return check


def _count(low: int, high: int) -> Callable[[str, Any], Any]:
    def check(key: str, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise _bad(f"compute_backend_config.{key} must be an integer from {low} to {high}")
        return value

    return check


def _scheduler(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        raise _bad(f"compute_backend_config.{key} must be a string")
    return require_allowed_scheduler(value)


def _log_directory(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        raise _bad(f"compute_backend_config.{key} must be a string")
    return str(
        resolve_safe_path(
            value,
            Path.home(),
            detail=f"compute_backend_config.{key} must be within the user's home directory",
        )
    )


def _k8s_resources(key: str, value: Any) -> Any:
    if not isinstance(value, dict) or not set(value) <= {"requests", "limits"}:
        raise _bad(f"compute_backend_config.{key} may hold only 'requests' and 'limits'")
    for part in value.values():
        if not isinstance(part, dict) or not set(part) <= {"cpu", "memory"}:
            raise _bad(f"compute_backend_config.{key} may set only 'cpu' and 'memory'")
        for quantity in part.values():
            if not isinstance(quantity, str) or not _K8S_QUANTITY.fullmatch(quantity):
                raise _bad(f"compute_backend_config.{key} quantities must look like 2, 500m or 4Gi")
    return value


_WORKERS = _count(0, 1024)

ALLOWED: dict[str, dict[str, Callable[[str, Any], Any]]] = {
    "sync": {},
    "local": {"n_workers": _WORKERS},
    "dask": {"scheduler": _scheduler, "n_workers": _WORKERS},
    "slurm": {
        "queue": _text(_NAME, "a partition name"),
        "project": _text(_NAME, "an account name"),
        "walltime": _text(_WALLTIME, "a time limit such as 02:00:00 or 1-12:00:00"),
        "cores": _count(1, 1024),
        "node_cores": _count(1, 1024),
        "memory": _text(_MEMORY, "a memory size such as 16GB"),
        "n_workers": _WORKERS,
        "scheduler_host": _text(_HOST, "a host name or address"),
        "scheduler_interface": _text(_NAME, "a network interface name"),
        "log_directory": _log_directory,
    },
    "kubernetes": {
        "namespace": _text(_K8S_NAME, "a Kubernetes namespace"),
        "n_workers": _WORKERS,
        "image": _text(_IMAGE, "a container image reference"),
        "resources": _k8s_resources,
    },
}


def validate_compute_backend_config(backend: str, config: dict[str, Any] | None) -> dict[str, Any]:
    """``config`` with every value checked, or HTTP 400 naming what is not allowed."""
    allowed = ALLOWED.get(backend.lower())
    if allowed is None:
        raise _bad(f"Unknown compute backend {backend!r}; choose one of {', '.join(ALLOWED)}")
    # The form sends cleared fields as empty strings: they mean "not set".
    config = {k: v for k, v in (config or {}).items() if v is not None and v != ""}
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        accepted = ", ".join(allowed) or "no settings"
        raise _bad(
            f"compute_backend_config for {backend!r} does not accept {', '.join(unknown)}; "
            f"it accepts {accepted}"
        )
    return {key: allowed[key](key, value) for key, value in config.items()}
