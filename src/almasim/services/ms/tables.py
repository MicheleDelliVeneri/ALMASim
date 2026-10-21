"""Reading MeasurementSet tables with python-casacore (MADIV-28 T2.7).

:func:`open_table` closes every table it opens; ``shared=True`` reads without
taking the read lock, which several tasks reading one MS over NFS need
(``lockoptions='usernoread'``: a read lock makes each of them write the
shared ``table.lock``, which failed with ``LockFile.cc: Failed AlwaysAssert
tracePWRITE`` when three tasks of one MS ran together, job 3835).

The other helpers work on a table the caller has open. Arrays come back in
python-casacore's row-major layout (see :mod:`.schema`).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

TARGET_INTENT = "OBSERVE_TARGET"


@contextmanager
def open_table(
    path: str | os.PathLike[str],
    *,
    readonly: bool = True,
    shared: bool = False,
    opener: Callable[..., Any] | None = None,
) -> Iterator[Any]:
    """The casacore table at ``path``, closed when the block ends.

    ``opener`` replaces ``casacore.tables.table`` (the CLI passes the one it
    resolves, which its tests fake).
    """
    if opener is None:
        from casacore.tables import table as casacore_table

        opener = casacore_table
    kwargs: dict[str, Any] = {"ack": False}
    if not readonly:
        kwargs["readonly"] = False
    if shared:
        kwargs["lockoptions"] = "usernoread"
    tb = opener(str(path), **kwargs)
    try:
        yield tb
    finally:
        close = getattr(tb, "close", None)
        if callable(close):
            close()


def spw_of_data_description(data_description: Any) -> list[int]:
    """The spectral window of each DATA_DESCRIPTION row."""
    return [int(x) for x in data_description.getcol("SPECTRAL_WINDOW_ID")]


def polarization_of_data_description(data_description: Any) -> list[int]:
    """The POLARIZATION row of each DATA_DESCRIPTION row."""
    return [int(x) for x in data_description.getcol("POLARIZATION_ID")]


def antenna_names(antenna: Any) -> list[str]:
    """The ANTENNA table's names, in row order."""
    return [str(name) for name in antenna.getcol("NAME")]


def intent_tokens(obs_mode: str) -> list[str]:
    """``CALIBRATE_PHASE#ON_SOURCE,CALIBRATE_WVR#ON_SOURCE`` -> its comma-separated intents."""
    return [token.strip() for token in str(obs_mode).split(",") if token.strip()]


def intent_names(obs_mode: str) -> set[str]:
    """A STATE ``OBS_MODE``'s intents without sub-intent (``CALIBRATE_PHASE``)."""
    return {token.split("#", 1)[0] for token in intent_tokens(obs_mode)}


def intent_matches(obs_mode: str, intents: Sequence[str]) -> bool:
    """Whether a STATE ``OBS_MODE`` carries one of ``intents``.

    An intent with a ``#`` sub-intent must match a token exactly; one without
    (``OBSERVE_TARGET``) matches any of its sub-intents.
    """
    tokens = intent_tokens(obs_mode)
    for intent in intents:
        if "#" in intent:
            if intent in tokens:
                return True
        elif any(token.split("#", 1)[0] == intent for token in tokens):
            return True
    return False
