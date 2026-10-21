"""Imaging task markers next to each task directory (MADIV-28 T8.2: a direct test)."""

import json

import pytest

from almasim.services.imaging import markers
from almasim.services.imaging.products import (
    CUBE_NAME,
    IMAGE_FILENAME,
    continuum_product,
    cube_product,
)

pytestmark = pytest.mark.unit


def test_marker_paths_sit_next_to_the_task_directory(tmp_path):
    out = tmp_path / "ms" / "SPW-3"
    assert markers.imaging_marker_path(out) == tmp_path / "ms" / "SPW-3.done"
    assert markers.imaging_failure_marker_path(out).name == "SPW-3.failed"
    assert markers.imaging_log_path(out).name == "SPW-3.log"


def test_done_marker_records_the_commands_and_completes_the_task(tmp_path):
    out = tmp_path / "SPW-1"
    out.mkdir()
    (out / IMAGE_FILENAME).write_bytes(b"")
    assert not markers.is_imaging_complete(out)
    path = markers.write_imaging_marker(
        out, ms_path="x.ms", spw=1, commands=[["wsclean", "-name", "a b"]], seconds=12.34
    )
    payload = json.loads(path.read_text())
    assert payload["commands"] == ["wsclean -name 'a b'"] and payload["seconds"] == 12.3
    assert markers.is_imaging_complete(out)
    assert not markers.is_imaging_complete(out, required=(IMAGE_FILENAME, "missing.fits"))


def test_a_failure_drops_the_done_marker(tmp_path):
    out = tmp_path / "SPW-2"
    markers.write_imaging_marker(out, ms_path="x.ms", spw=2, command=["wsclean"])
    failed = markers.write_imaging_failure_marker(
        out, ms_path="x.ms", spw=2, error="boom", returncode=139
    )
    assert not markers.imaging_marker_path(out).exists()
    payload = json.loads(failed.read_text())
    assert payload["returncode"] == 139 and "boom" in payload["error"]


def test_product_names():
    assert continuum_product("residual") == "wsclean-residual.fits"
    assert cube_product("image") == f"{CUBE_NAME}-image.fits"
