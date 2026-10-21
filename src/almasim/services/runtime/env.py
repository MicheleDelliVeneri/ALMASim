"""Child-process environments for WSClean and CASA (MADIV-28 T2.3).

One library-path policy: the system C library goes ahead of spack builds
(GLIBC conflicts), and every CASA child additionally searches casatools'
bundled libraries before the system ones, because ``LD_LIBRARY_PATH`` is
searched before RUNPATH and a system ``libpmix`` would otherwise shadow the
bundled one (``libopen-pal.so.80: undefined symbol: pmix_framework_names``,
job 5299, 2026-10-17).
"""

from __future__ import annotations

import os
from collections.abc import Mapping

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


def casa_bundled_lib_dir() -> str | None:
    """The directory of shared libraries ``casatools`` ships in its wheel, if any.

    Found with ``find_spec``, so the package is located without importing the
    extension modules that fail in exactly the situation this guards against.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("casatools")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    lib_dir = os.path.join(os.path.dirname(spec.origin), "__casac__", "lib")
    return lib_dir if os.path.isdir(lib_dir) else None


def casa_library_path(existing: str, bundled: str | None) -> str:
    """``LD_LIBRARY_PATH`` for a CASA child: bundled libs, system libs, then ``existing``."""
    dirs = ([bundled] if bundled else []) + _SYSTEM_LIBRARY_DIRS.split(":")
    path = ":".join(dirs)
    return f"{path}:{existing}" if existing else path


_FIND = object()


def casa_environment(
    base: Mapping[str, str] | None = None, *, bundled: str | None | object = _FIND
) -> dict[str, str]:
    """Environment for a CASA child: one thread each, casatools' libraries first.

    ``bundled`` defaults to :func:`casa_bundled_lib_dir`.
    """
    source = dict(os.environ if base is None else base)
    env = wsclean_environment(1, source)
    lib_dir = casa_bundled_lib_dir() if bundled is _FIND else bundled
    env["LD_LIBRARY_PATH"] = casa_library_path(source.get("LD_LIBRARY_PATH", ""), lib_dir)  # type: ignore[arg-type]
    return env
