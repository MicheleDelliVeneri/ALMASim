"""Unit tests for almasim.services.archive.qa0."""

from __future__ import annotations

from pathlib import Path

import pytest

from almasim.services.archive.qa0 import (
    QA0Report,
    calibration_is_delivered,
    parse_qa0_report_text,
    qa0_report_path,
)

# Text as pypdf extracts it from a delivered report, including the line break
# between "QA0 Status" and its value.
SEMIPASS_TEXT = """QA0 Report

Execution Block Summary
Project Code 2023.1.01253.S SchedBlock A2744-20_a_04_TM1
ExecBlock uid://A002/X10d9399/Xae5e ExecBlock Status FAIL
QA0 Status
 SemiPass Exec. Fraction 0.00
Repr. frequency 128.639 GHz (Sky) Band ALMA_RB_04
QA0 comment Failed just after initial calibration scans have finished,
and thus almost no target data.
Page 1 of
 10
"""

PASS_TEXT = """QA0 Report

Execution Block Summary
Project Code 2023.1.00868.S SchedBlock X_a_03_TM1
ExecBlock uid://A002/X10e2702/X5c84 ExecBlock Status SUCCESS
QA0 Status
 Pass Exec. Fraction 1.34
QA0 comment No severe issue was found.
Page 1 of
 12
"""


@pytest.mark.unit
def test_parse_semipass_report():
    report = parse_qa0_report_text(SEMIPASS_TEXT, "uid___A002_X10d9399_Xae5e")
    assert report.qa0_status == "SemiPass"
    assert report.execblock_status == "FAIL"
    assert report.execution_fraction == 0.0
    # The comment is rejoined across its line break.
    assert report.comment.startswith("Failed just after initial calibration")
    assert "almost no target data." in report.comment
    assert report.calibration_delivered is False


@pytest.mark.unit
def test_parse_pass_report():
    report = parse_qa0_report_text(PASS_TEXT, "uid___A002_X10e2702_X5c84")
    assert report.qa0_status == "Pass"
    assert report.execution_fraction == 1.34
    assert report.calibration_delivered is True


@pytest.mark.unit
def test_execblock_status_does_not_decide_delivery():
    """A SemiPass EB often reports ExecBlock Status SUCCESS; only QA0 Status counts."""
    text = SEMIPASS_TEXT.replace("ExecBlock Status FAIL", "ExecBlock Status SUCCESS")
    report = parse_qa0_report_text(text)
    assert report.execblock_status == "SUCCESS"
    assert report.calibration_delivered is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "status,expected",
    [
        ("Pass", True),
        ("pass", True),
        ("SemiPass", False),
        ("SemiPass,", False),  # trailing comma occurs in real reports
        (" Pass ", True),
        ("", False),
    ],
)
def test_calibration_is_delivered(status, expected):
    assert calibration_is_delivered(status) is expected


@pytest.mark.unit
def test_missing_fields_do_not_raise():
    report = parse_qa0_report_text("QA0 Report\nnothing useful here\n")
    assert report == QA0Report(
        eb_uid="", qa0_status="", execblock_status="", execution_fraction=None, comment=""
    )
    assert report.calibration_delivered is False


@pytest.mark.unit
def test_qa0_report_path():
    path = qa0_report_path("/data/member.uid___A001_X1_X2", "uid___A002_X3_X4")
    assert path == Path("/data/member.uid___A001_X1_X2/qa/uid___A002_X3_X4.qa0_report.pdf")
