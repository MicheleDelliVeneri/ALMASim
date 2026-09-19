"""Read ALMA QA0 reports.

Every execution block ships a ``<eb_uid>.qa0_report.pdf`` in its member OUS
``qa/`` directory. Its ``QA0 Status`` field decides whether ALMA delivers
calibration for that EB at all: ``Pass`` blocks always carry calibration
products, ``SemiPass`` blocks never do. ALMA still ships the raw ASDM for a
SemiPass block, so resolving and downloading without consulting this field
pulls execution blocks that can never be calibrated.

Do not use ``ExecBlock Status`` for this decision -- it is a separate field and
does not discriminate: a SemiPass block frequently carries
``ExecBlock Status = SUCCESS``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "QA0Report",
    "calibration_is_delivered",
    "parse_qa0_report_text",
    "qa0_report_path",
    "read_qa0_report",
]

# QA0 statuses for which ALMA delivers calibration products.
_DELIVERED_STATUSES = frozenset({"pass"})


@dataclass(frozen=True)
class QA0Report:
    """The fields of a QA0 report that decide whether an EB is usable."""

    eb_uid: str
    qa0_status: str
    execblock_status: str
    execution_fraction: float | None
    comment: str

    @property
    def calibration_delivered(self) -> bool:
        return calibration_is_delivered(self.qa0_status)


def calibration_is_delivered(qa0_status: str) -> bool:
    """Return True when ``qa0_status`` is one ALMA delivers calibration for."""
    # Reports occasionally carry a trailing comma ("SemiPass,").
    return qa0_status.strip().rstrip(",").lower() in _DELIVERED_STATUSES


def qa0_report_path(member_dir: str | Path, eb_uid: str) -> Path:
    """Return the QA0 report path for ``eb_uid`` inside a member OUS directory."""
    return Path(member_dir) / "qa" / f"{eb_uid}.qa0_report.pdf"


def parse_qa0_report_text(text: str, eb_uid: str = "") -> QA0Report:
    """Parse the first page of a QA0 report.

    Kept separate from the PDF read so the field patterns can be tested without
    a binary fixture.
    """

    def field(pattern: str) -> str:
        # DOTALL: QA0 comments routinely wrap across several lines. It is safe
        # for the other patterns here, none of which use ".".
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    # "QA0 Status" and its value are split across lines in the delivered layout.
    fraction_text = field(r"Exec\. Fraction\s+([\d.]+)")
    comment = field(r"QA0 comment\s+(.*?)(?:\nPage \d|\Z)")
    return QA0Report(
        eb_uid=eb_uid,
        qa0_status=field(r"QA0 Status\s*\n?\s*(\S+)"),
        execblock_status=field(r"ExecBlock Status\s+(\S+)"),
        execution_fraction=float(fraction_text) if fraction_text else None,
        comment=" ".join(comment.split()),
    )


def read_qa0_report(pdf_path: str | Path, eb_uid: str = "") -> QA0Report:
    """Read and parse the QA0 report at ``pdf_path``."""
    from pypdf import PdfReader

    path = Path(pdf_path)
    if not eb_uid:
        eb_uid = path.name.removesuffix(".qa0_report.pdf")
    text = PdfReader(str(path)).pages[0].extract_text()
    return parse_qa0_report_text(text, eb_uid)
