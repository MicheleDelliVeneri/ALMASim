"""Child-process environments for WSClean and CASA (policy to be unified in T2.3).

Moved out of archive_imaging.py (MADIV-28 module split).
"""

from __future__ import annotations

import os
from typing import Mapping

# Prefer the system C library over spack builds, but keep whatever the caller
# had after it (same policy as the unpack/calibrate subprocess wrappers).
_SYSTEM_LIBRARY_DIRS = "/lib64:/usr/lib64:/usr/local/lib64:/lib:/usr/lib:/usr/local/lib"


def wsclean_environment(cores: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a WSClean child using ``cores`` threads.

    WSClean refuses to start when it is linked against a multi-threaded
    OpenBLAS unless ``OPENBLAS_NUM_THREADS=1``; its own ``-j`` flag carries the
    parallelism. The other threading knobs follow the requested core count.
    """
    env = dict(os.environ if base is None else base)
    env["OPENBLAS_NUM_THREADS"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(max(int(cores), 1))
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{_SYSTEM_LIBRARY_DIRS}:{existing}" if existing else _SYSTEM_LIBRARY_DIRS
    )
    return env


def casa_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a CASA child: one thread each, and no system library
    directories in ``LD_LIBRARY_PATH``.

    ``wsclean_environment`` and the Slurm worker's job script put /lib64 and
    friends first for WSClean's spack build; casatools then loads the
    system PMIx under its bundled Open MPI and fails to import
    (``libopen-pal.so.80: undefined symbol: pmix_framework_names``, every
    task of job 5299, 2026-10-17).
    """
    env = wsclean_environment(1, base)
    system = set(_SYSTEM_LIBRARY_DIRS.split(":"))
    kept = [d for d in env.get("LD_LIBRARY_PATH", "").split(":") if d and d not in system]
    if kept:
        env["LD_LIBRARY_PATH"] = ":".join(kept)
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return env
