"""API v2: your profile from the Settings page (web console plan, U3.8)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from almasim.jobs.profile import load_profile
from almasim.web.app import COOKIE, create_app

pytestmark = pytest.mark.unit

TOKEN = "profile-test-token"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    site = tmp_path / "site.toml"
    site.write_text(f'runs_root = "{tmp_path}/runs"\n[slurm]\npartition = "zwickyA-CPU"\n')
    user = tmp_path / "home" / ".config" / "almasim" / "almasim.toml"
    monkeypatch.setenv("ALMASIM_SITE_PROFILE", str(site))
    monkeypatch.setenv("ALMASIM_PROFILE", str(user))
    user.parent.mkdir(parents=True)
    user.write_text('[slurm]\naccount = "ska"\n')
    profile = load_profile()
    app = create_app(TOKEN, static_dir=tmp_path / "none", profile=profile, events_interval=60)
    client = TestClient(app)
    client.cookies.set(COOKIE, TOKEN)
    return client, app, site, user


def test_both_profiles_and_what_they_add_up_to(setup):
    client, _app, site, user = setup
    got = client.get("/api/v2/profile").json()
    assert got["site"] == {"path": str(site), "exists": True, "text": site.read_text()}
    assert got["user"]["path"] == str(user) and 'account = "ska"' in got["user"]["text"]
    assert got["effective"]["partition"] == "zwickyA-CPU"
    assert got["effective"]["account"] == "ska"


def test_your_profile_is_saved_and_used_at_once(setup):
    client, app, _site, user = setup
    text = '[slurm]\npartition = "zwickyB-CPU"\naccount = "ska"\n'
    got = client.put("/api/v2/profile/user", json={"text": text})
    assert got.status_code == 200, got.text
    assert user.read_text() == text
    assert got.json()["effective"]["partition"] == "zwickyB-CPU"
    assert app.state.profile.partition == "zwickyB-CPU"
    assert app.state.events.profile is app.state.profile
    assert app.state.cluster.profile is app.state.profile


def test_a_mistake_is_refused_and_nothing_changes(setup):
    client, app, _site, user = setup
    before = user.read_text()
    bad = client.put("/api/v2/profile/user", json={"text": "[slurm]\npartiton = 'x'\n"})
    assert bad.status_code == 400
    assert str(user) in bad.json()["detail"] and "unknown key" in bad.json()["detail"]
    assert user.read_text() == before
    assert app.state.profile.partition == "zwickyA-CPU"
    assert [p.name for p in user.parent.iterdir()] == ["almasim.toml"]  # no temp file left
