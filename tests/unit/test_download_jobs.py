"""Re-running and removing download jobs from their records (MADIV-28 T6.7)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from almasim.services.download.jobs import (
    delete_job_files,
    json_list,
    redownload_options,
    stored_products,
)

pytestmark = pytest.mark.unit


def _record(destination, **overrides):
    values = dict(
        destination=str(destination),
        files=[
            SimpleNamespace(access_url="https://x/a.tar", filename="a.tar", content_length="10")
        ],
        unpack_ms=False,
        generate_calibrated_visibilities=False,
        clean_intermediate_files=False,
        archive_output_root=None,
        casa_data_root=None,
        skip_casa_data_update=False,
        raw_measurement_sets=json.dumps([]),
        calibrated_measurement_sets=json.dumps([]),
        manifest_path=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_json_list():
    assert json_list('["a"]') == ["a"] and json_list("{}") == [] and json_list(None) == []


def test_stored_products_reuse_the_original_access_urls(tmp_path):
    (product,) = stored_products(_record(tmp_path))
    assert (product.access_url, product.filename, product.content_length) == (
        "https://x/a.tar",
        "a.tar",
        10,
    )


def test_request_values_win_and_calibration_implies_unpacking_and_extraction(tmp_path):
    record = _record(tmp_path, clean_intermediate_files=True, archive_output_root="/old")
    request = SimpleNamespace(
        max_parallel=None,
        extract_tar=False,
        unpack_ms=None,
        generate_calibrated_visibilities=True,
        clean_intermediate_files=None,
        archive_output_root="/new",
        casa_data_root=None,
        skip_casa_data_update=None,
    )
    options = redownload_options(record, request)
    assert options.max_parallel == 3
    assert options.generate_calibrated_visibilities and options.unpack_ms and options.extract_tar
    assert options.clean_intermediate_files is True, "stored value kept"
    assert options.archive_output_root == "/new", "request value wins"


def test_stored_options_apply_when_the_request_gives_none(tmp_path):
    options = redownload_options(_record(tmp_path, unpack_ms=True), SimpleNamespace())
    assert (
        options.unpack_ms and options.extract_tar and not options.generate_calibrated_visibilities
    )


def test_delete_removes_only_what_is_inside_the_destination(tmp_path):
    dest = tmp_path / "dest"
    (dest / "raw.ms").mkdir(parents=True)
    (dest / "a.tar").write_text("x")
    (dest / "a.tar.part").write_text("x")
    outside = tmp_path / "keep.txt"
    outside.write_text("x")
    record = _record(
        dest,
        raw_measurement_sets=json.dumps(["raw.ms", str(outside)]),
        manifest_path="../keep.txt",
    )
    assert delete_job_files(record) == (3, 2)
    assert not (dest / "raw.ms").exists() and not (dest / "a.tar").exists()
    assert outside.exists()
