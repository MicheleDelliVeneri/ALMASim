"""The CLI shows the library's log records once (MADIV-28 T2.10)."""

from __future__ import annotations

import logging

import pytest

from almasim import cli_shared
from almasim.services import utils

pytestmark = pytest.mark.unit


@pytest.fixture
def almasim_logger():
    log = logging.getLogger("almasim")
    saved = (list(log.handlers), log.level)
    yield log
    log.handlers[:], log.level = saved[0], saved[1]


def test_configure_logging_adds_one_stderr_handler(almasim_logger, capsys):
    cli_shared.configure_logging()
    cli_shared.configure_logging()
    handlers = [
        h for h in almasim_logger.handlers if isinstance(h, cli_shared._CurrentStderrHandler)
    ]
    assert len(handlers) == 1
    logging.getLogger("almasim.scheduling.cluster").info("Dask dashboard: http://x:8787")
    logging.getLogger("almasim.services.x").warning("careful")
    assert capsys.readouterr().err.splitlines() == [
        "Dask dashboard: http://x:8787",
        "WARNING: careful",
    ]


def test_report_goes_to_the_sink_or_the_log_never_both(almasim_logger, caplog):
    log = logging.getLogger("almasim.test.report")
    echoed = []
    with caplog.at_level(logging.DEBUG, logger="almasim.test.report"):
        utils.report(log, echoed.append, "copied", logging.WARNING)
        utils.report(log, None, "logged", logging.WARNING)
    assert echoed == ["WARNING: copied"]
    levels = {r.getMessage(): r.levelno for r in caplog.records}
    assert levels == {"copied": logging.DEBUG, "logged": logging.WARNING}


def test_log_message_uses_the_callback_or_logging(caplog):
    seen = []
    with caplog.at_level(logging.INFO, logger="almasim.simulation"):
        utils.log_message(seen.append, "to the callback")
        utils.log_message(None, "to the log")
        utils.log_message(seen.append, "remote", remote=True)
    assert seen == ["to the callback"]
    assert [r.getMessage() for r in caplog.records] == ["to the log", "remote"]
