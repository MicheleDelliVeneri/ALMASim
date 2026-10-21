"""One clock for the backend: UTC (MADIV-28 T6.4).

In-memory job state and the API use timezone-aware UTC datetimes. The
database's ``DateTime`` columns have no zone and hold UTC; :func:`as_utc`
attaches it when such a value leaves the database. The in-memory stores used
local time while the database used UTC, so a list mixing both sorted wrongly
and the browser read the stored times as local.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import overload


def utc_now() -> datetime:
    """The current time, timezone-aware UTC."""
    return datetime.now(UTC)


def db_now() -> datetime:
    """The current UTC time for a database ``DateTime`` column (no zone)."""
    return utc_now().replace(tzinfo=None)


@overload
def as_utc(value: datetime) -> datetime: ...
@overload
def as_utc(value: None) -> None: ...
def as_utc(value: datetime | None) -> datetime | None:
    """``value`` as aware UTC; a naive value is a database time, already UTC."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)
