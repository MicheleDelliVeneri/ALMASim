"""The ALMA TAP connection: mirror selection and retried searches (MADIV-28 T6.10)."""

from __future__ import annotations

import logging

import pyvo
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_TAP_URLS = [
    "https://almascience.eso.org/tap",
    "https://almascience.nao.ac.jp/tap",
    "https://almascience.nrao.edu/tap",
]

_TAP_TEST_TIMEOUT = 30  # seconds for the connectivity test query

# The mirror that last answered; reused until a query on it fails.
_connected_service: pyvo.dal.TAPService | None = None


def reset_tap_service() -> None:
    """Forget the cached mirror, so the next query probes the mirrors again."""
    global _connected_service
    _connected_service = None


def get_tap_service(max_rounds: int = 2) -> pyvo.dal.TAPService:
    """Return a connected ALMA TAP service, trying each mirror in turn.

    The first mirror that answers a test query is cached and returned by
    later calls, so each query does not pay for a new connectivity probe.
    Tries each mirror URL up to *max_rounds* times total. Raises
    ``RuntimeError`` if every attempt fails so callers can handle the
    failure rather than blocking forever.
    """
    global _connected_service
    if _connected_service is not None:
        return _connected_service
    errors: list[str] = []
    for _ in range(max_rounds):
        for url in _TAP_URLS:
            try:
                service = pyvo.dal.TAPService(url)
                service.search(
                    "SELECT TOP 1 * FROM ivoa.obscore",
                    response_timeout=_TAP_TEST_TIMEOUT,
                )
                logger.info("Connected to the ALMA TAP service at %s", url)
                _connected_service = service
                return service
            except Exception as exc:  # noqa: BLE001 - the next mirror is tried
                msg = f"{url}: {exc}"
                errors.append(msg)
                logger.warning("Could not connect to the ALMA TAP service at %s", msg)
    raise RuntimeError(
        f"All ALMA TAP mirrors unreachable after {max_rounds} rounds. Errors:\n" + "\n".join(errors)
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_with_retry(service, query):
    """Search TAP service with retry logic."""
    try:
        return service.search(query, response_timeout=120).to_table().to_pandas()
    except Exception:
        # The next query probes the mirrors again instead of reusing this one.
        reset_tap_service()
        raise
