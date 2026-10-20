"""Thread-safe in-memory state of running jobs (MADIV-28 T6.4).

The simulation, download and metadata-query stores each kept a dict and a
lock and stamped ``updated_at`` by hand; :class:`LiveProgress` is that once.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generic, Iterator, Optional, TypeVar

from app.core.clock import utc_now

T = TypeVar("T")


class LiveProgress(Generic[T]):
    """Jobs by id; :meth:`editing` changes one under the lock and stamps ``updated_at``."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}
        self._lock = threading.Lock()

    def put(self, key: str, item: T) -> T:
        with self._lock:
            self._items[key] = item
            return item

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            return self._items.get(key)

    def pop(self, key: str) -> bool:
        """Drop ``key``; whether it was there."""
        with self._lock:
            return self._items.pop(key, None) is not None

    def values(self) -> list[T]:
        with self._lock:
            return list(self._items.values())

    @contextmanager
    def editing(self, key: str) -> Iterator[Optional[T]]:
        """The job (None when unknown), held under the lock; ``updated_at`` is stamped after."""
        with self._lock:
            item = self._items.get(key)
            yield item
            if item is not None and hasattr(item, "updated_at"):
                item.updated_at = utc_now()  # type: ignore[attr-defined]
