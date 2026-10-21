"""almasim serve: the token, the socket and the server record (web console plan, S2.1)."""

from __future__ import annotations

import os
import stat
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from almasim.web import serve as state  # noqa: E402
from almasim.web.app import COOKIE, create_app  # noqa: E402

pytestmark = pytest.mark.unit

TOKEN = "secret-token-for-tests"


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(TOKEN, static_dir=tmp_path / "no-build"), follow_redirects=False)


def test_without_the_token_nothing_is_served(client):
    api = client.get("/api/v2/health")
    assert api.status_code == 401 and api.json() == {"detail": "Not authenticated"}
    page = client.get("/")
    assert page.status_code == 401 and "almasim serve" in page.text
    assert (
        client.get("/api/v2/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    )
    client.cookies.set(COOKIE, "wrong")
    assert client.get("/api/v2/health").status_code == 401


def test_the_link_trades_its_token_for_a_cookie(client):
    response = client.get("/?token=" + TOKEN + "&dir=/data")
    assert response.status_code == 303
    assert response.headers["location"] == "/?dir=%2Fdata"
    cookie = response.headers["set-cookie"]
    assert f"{COOKIE}={TOKEN}" in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert client.get("/api/v2/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200


def test_a_wrong_link_token_is_refused(client):
    assert client.get("/?token=nope").status_code == 401
    assert client.post("/api/v2/health?token=" + TOKEN).status_code == 401  # only GET trades


def test_api_clients_may_send_a_bearer_token(client):
    response = client.get("/api/v2/health", headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200


def test_the_built_web_app_is_served_when_present(tmp_path):
    (tmp_path / "index.html").write_text("<p>console</p>")
    app_client = TestClient(create_app(TOKEN, static_dir=tmp_path))
    app_client.cookies.set(COOKIE, TOKEN)
    assert "console" in app_client.get("/").text


def test_an_app_needs_a_token():
    with pytest.raises(ValueError):
        create_app("")


def test_the_server_record_is_private_and_forgets_dead_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    info = state.ServerInfo(pid=os.getpid(), token="t", host="zwicky", socket="/s/serve.sock")
    path = state.write_info(info)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert state.read_info() == info
    assert info.ssh_line().endswith(
        "-L 8765:/s/serve.sock " + os.environ.get("USER", "you") + "@zwicky"
    )
    assert info.url() == "http://localhost:8765/?token=t"

    state.write_info(state.ServerInfo(pid=999999999, token="t", host="h", port=9000))
    assert state.read_info() is None


def test_the_console_answers_on_a_socket_only_its_user_can_open(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = state.state_dir()
    sock = folder / state.SOCKET_NAME
    server = uvicorn.Server(uvicorn.Config(create_app(TOKEN), uds=str(sock), log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while not sock.exists():
            assert time.monotonic() < deadline, "the server never bound its socket"
            time.sleep(0.05)
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700
        transport = httpx.HTTPTransport(uds=str(sock))
        with httpx.Client(transport=transport, base_url="http://console") as http:
            assert http.get("/api/v2/health").status_code == 401
            ok = http.get("/api/v2/health", headers={"Authorization": f"Bearer {TOKEN}"})
            assert ok.json()["status"] == "ok"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
