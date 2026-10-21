"""API v2 files: browse, read and fetch under the profile's roots (web console plan, S2.4)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("fastapi")
from astropy.io import fits
from fastapi.testclient import TestClient

from almasim.jobs.profile import Profile
from almasim.web.app import COOKIE, create_app

pytestmark = pytest.mark.unit

TOKEN = "files-test-token"


@pytest.fixture
def area(tmp_path):
    data = tmp_path / "data"
    (data / "run 1").mkdir(parents=True)
    (data / ".hidden").write_text("x")
    (data / "params.csv").write_text("ms,spw\na.ms,25\nb.ms,27\nc.ms,29\n")
    (data / "quality.json").write_text(json.dumps({"snr": 45}))
    (data / "log.txt").write_text("line one\nline two\n")
    cube = np.arange(3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    cube[0, 1, 0, 0] = np.nan
    hdu = fits.PrimaryHDU(cube)
    hdu.header["BUNIT"] = "Jy/beam"
    hdu.writeto(data / "cube.fits")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key").write_text("no")
    return tmp_path


@pytest.fixture
def client(area):
    profile = Profile(runs_root=area / "runs", roots=(area / "data",))
    app = create_app(TOKEN, static_dir=area / "none", profile=profile, events_interval=60)
    c = TestClient(app)
    c.cookies.set(COOKIE, TOKEN)
    return c


def test_roots_and_listing(client, area):
    roots = client.get("/api/v2/files/roots").json()["roots"]
    assert [r["path"] for r in roots] == [str(area / "data"), str(area / "runs")]
    assert roots[1]["runs"] and not roots[1]["exists"]

    listing = client.get("/api/v2/files/list", params={"path": str(area / "data")}).json()
    names = [e["name"] for e in listing["entries"]]
    assert names[0] == "run 1"  # folders first
    assert ".hidden" not in names and "params.csv" in names
    with_hidden = client.get(
        "/api/v2/files/list", params={"path": str(area / "data"), "hidden": True}
    )
    assert ".hidden" in [e["name"] for e in with_hidden.json()["entries"]]


def test_paths_outside_the_roots_are_refused(client, area):
    for path in (str(area / "secret"), str(area / "data" / ".." / "secret" / "key"), "/etc/passwd"):
        assert client.get("/api/v2/files/text", params={"path": path}).status_code == 403
    link = area / "data" / "escape"
    os.symlink(area / "secret", link)
    assert client.get("/api/v2/files/list", params={"path": str(link)}).status_code == 403


def test_relative_paths_are_taken_from_the_first_root(client, area):
    text = client.get("/api/v2/files/text", params={"path": "log.txt"}).json()
    assert text["text"] == "line one\nline two\n" and text["next_offset"] == text["size"]


def test_mkdir_inside_a_root(client, area):
    made = client.post("/api/v2/files/mkdir", json={"path": str(area / "data" / "new" / "deep")})
    assert made.status_code == 201 and (area / "data" / "new" / "deep").is_dir()
    assert client.post("/api/v2/files/mkdir", json={"path": str(area / "x")}).status_code == 403


def test_csv_pages_and_json(client, area):
    page = client.get(
        "/api/v2/files/csv",
        params={"path": str(area / "data" / "params.csv"), "start": 1, "rows": 1},
    ).json()
    assert (
        page["columns"] == ["ms", "spw"] and page["rows"] == [["b.ms", "27"]] and page["total"] == 3
    )
    assert client.get(
        "/api/v2/files/json", params={"path": str(area / "data" / "quality.json")}
    ).json() == {"snr": 45}
    bad = client.get("/api/v2/files/json", params={"path": str(area / "data" / "log.txt")})
    assert bad.status_code == 422


def test_fits_header_and_one_plane(client, area):
    path = str(area / "data" / "cube.fits")
    header = client.get("/api/v2/files/fits", params={"path": path}).json()
    assert header["shape"] == [1, 3, 4, 5]
    assert {"key": "BUNIT", "value": "Jy/beam", "comment": ""} in header["cards"]

    plane = client.get("/api/v2/files/fits/plane", params={"path": path, "channel": 1}).json()
    assert plane["shape"] == [4, 5] and plane["step"] == 1
    assert plane["data"][0][0] is None  # the NaN
    assert plane["data"][0][1] == 21.0 and plane["max"] == 39.0
    too_far = client.get("/api/v2/files/fits/plane", params={"path": path, "channel": 3})
    assert too_far.status_code == 400
    not_fits = client.get("/api/v2/files/fits", params={"path": str(area / "data" / "log.txt")})
    assert not_fits.status_code == 422


def test_download_and_missing_files(client, area):
    got = client.get("/api/v2/files/download", params={"path": str(area / "data" / "log.txt")})
    assert got.status_code == 200 and got.text.startswith("line one")
    assert "log.txt" in got.headers["content-disposition"]
    assert (
        client.get("/api/v2/files/text", params={"path": str(area / "data" / "nope")}).status_code
        == 404
    )
    assert client.get("/api/v2/files/text", params={"path": str(area / "data")}).status_code == 400
