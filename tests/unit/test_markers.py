"""Stage markers are written one way for unpack, calibrate and image (MADIV-28 T2.6)."""

from __future__ import annotations

import json

import pytest

from almasim.services.archive import calibration as archive_calibration
from almasim.services.archive import unpack as archive_unpack
from almasim.services.imaging import markers as imaging_markers
from almasim.services.markers import StageMarkers

pytestmark = pytest.mark.unit


def test_done_clears_an_earlier_failure(tmp_path):
    markers = StageMarkers(tmp_path / "x.done", tmp_path / "x.failed")
    markers.write_failed("boom", stage="unpack", uid="u")
    assert markers.has_failed() and not markers.is_done()

    markers.write_done(uid="u", rows=3)
    assert markers.is_done() and not markers.has_failed()
    payload = json.loads(markers.done.read_text())
    assert payload["rows"] == 3 and "completed_at" in payload


def test_failure_creates_its_directory_and_records_the_log(tmp_path):
    markers = StageMarkers(tmp_path / "a" / "x.done", tmp_path / "a" / "x.failed")
    markers.write_failed("boom", stage="calibrate", log_path=tmp_path / "x.log", uid="u")
    payload = json.loads(markers.failed.read_text())
    assert {k: payload[k] for k in ("uid", "stage", "error", "log")} == {
        "uid": "u",
        "stage": "calibrate",
        "error": "boom",
        "log": str(tmp_path / "x.log"),
    }


def test_stage_payloads_keep_their_fields(tmp_path):
    archive_unpack.write_raw_ms_failure_marker(tmp_path, "uid_A", "no rows")
    archive_unpack.write_raw_ms_marker(tmp_path, "uid_A", rows=10)
    assert set(json.loads(archive_unpack.raw_ms_marker_path(tmp_path, "uid_A").read_text())) == {
        "uid",
        "output",
        "rows",
        "completed_at",
    }
    assert not archive_unpack.raw_ms_failure_marker_path(tmp_path, "uid_A").exists()

    archive_calibration.write_calibration_failure_marker(
        tmp_path, "uid_B", "no tables", stage="apply"
    )
    failed = json.loads(
        archive_calibration.calibration_failure_marker_path(tmp_path, "uid_B").read_text()
    )
    assert set(failed) == {"uid", "stage", "error", "failed_at"} and failed["stage"] == "apply"


def test_a_failed_image_task_drops_its_old_done_marker(tmp_path):
    out = tmp_path / "ms" / "SPW-3"
    out.mkdir(parents=True)
    imaging_markers.write_imaging_marker(out, ms_path="a.ms", spw=3, seconds=1.25)
    done = json.loads(imaging_markers.imaging_marker_path(out).read_text())
    assert done["seconds"] == 1.2 and done["spw"] == 3

    imaging_markers.write_imaging_failure_marker(
        out, ms_path="a.ms", spw=3, error="x", returncode=2
    )
    assert not imaging_markers.imaging_marker_path(out).exists()
    failed = json.loads(imaging_markers.imaging_failure_marker_path(out).read_text())
    assert set(failed) == {"ms", "spw", "stage", "output", "error", "failed_at", "returncode"}
