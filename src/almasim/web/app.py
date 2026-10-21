"""The console's FastAPI application and its access token (web console plan, S2.1).

The server runs as one user on a shared login node. It listens on a socket
only that user can open (or on 127.0.0.1), and every request must also carry
the token ``almasim serve`` printed: as the ``almasim_token`` cookie, which
opening the printed link once sets, or as ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import Scope

from almasim.__version__ import __version__
from almasim.jobs.profile import Profile, load_profile

from .api import cluster, commands, files, jobs, simulation
from .api import profile as profile_api
from .events import EventHub
from .events import router as events_router

COOKIE = "almasim_token"
# The built web app: inside the package in a wheel (S2.6); in a source
# checkout, the frontend's own build folder (`npm run build`).
STATIC_DIR = Path(__file__).parent / "static"
CHECKOUT_BUILD = Path(__file__).resolve().parents[3] / "frontend" / "build"
WEB_DIR_ENV = "ALMASIM_WEB_DIR"

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>ALMASim</title>
<p>The ALMASim web console is running, but its pages are not built into this
install. The API is at <a href="/api/v2/health">/api/v2</a>.</p>"""

_LOCKED = """<!doctype html><meta charset="utf-8"><title>ALMASim</title>
<p>Open the link <code>almasim serve</code> printed (it carries the access
token), or run <code>almasim serve --status</code> to see it again.</p>"""


class AppFiles(StaticFiles):
    """The built web app. Its pages are routed in the browser, so a path that
    is not a file (``/jobs/01J…``) gets the app's fallback page; the browser
    then shows the right page. API paths never fall back."""

    FALLBACK = "200.html"

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or path.split("/", 1)[0] == "api":
                raise
            return await super().get_response(self.FALLBACK, scope)


class TokenMiddleware(BaseHTTPMiddleware):
    """Let a request through only with the server's token."""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self.token = token

    def _valid(self, value: str | None) -> bool:
        if not value:
            return False
        return hmac.compare_digest(value.encode(), self.token.encode())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        offered = request.query_params.get("token")
        if offered is not None and request.method == "GET":
            if not self._valid(offered):
                return HTMLResponse(_LOCKED, status_code=401)
            # Trade the link's token for a cookie, and drop it from the address bar.
            rest = {k: v for k, v in request.query_params.items() if k != "token"}
            target = request.url.path + (f"?{urlencode(rest)}" if rest else "")
            response = RedirectResponse(target, status_code=303)
            response.set_cookie(
                COOKIE, self.token, httponly=True, samesite="strict", path="/", max_age=30 * 86400
            )
            return response

        bearer = request.headers.get("authorization", "")
        presented = request.cookies.get(COOKIE) or (
            bearer[7:] if bearer.lower().startswith("bearer ") else None
        )
        if not self._valid(presented):
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)
            return HTMLResponse(_LOCKED, status_code=401)
        return await call_next(request)


def web_dir(environ: Mapping[str, str] | None = None) -> Path | None:
    """Where the built web app is: ``$ALMASIM_WEB_DIR``, else the packaged
    copy, else a source checkout's ``frontend/build``; None when there is none."""
    env = os.environ if environ is None else environ
    if env.get(WEB_DIR_ENV):
        return Path(env[WEB_DIR_ENV]).expanduser()
    for candidate in (STATIC_DIR, CHECKOUT_BUILD):
        if (candidate / AppFiles.FALLBACK).is_file():
            return candidate
    return None


def create_app(
    token: str,
    static_dir: Path | None = None,
    *,
    profile: Profile | None = None,
    events_interval: float = 15.0,
) -> FastAPI:
    """The console app, serving the built web app from ``static_dir`` (default: ``web_dir()``).

    ``profile`` is the job profile the server submits with (default: the
    one ``almasim`` reads from the environment); ``events_interval`` is how
    often, in seconds, it looks at the runs between actions.
    """
    if not token:
        raise ValueError("the console needs an access token")
    profile = profile if profile is not None else load_profile()
    hub = EventHub(profile, interval=events_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.start()
        try:
            yield
        finally:
            hub.stop()

    app = FastAPI(
        title="ALMASim web console", version=__version__, docs_url="/api/docs", lifespan=lifespan
    )
    app.state.profile = profile
    app.state.events = hub
    app.state.cluster = cluster.ClusterView(profile)
    app.add_middleware(TokenMiddleware, token=token)

    @app.get("/api/v2/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(commands.router)
    app.include_router(jobs.router)
    app.include_router(files.router)
    app.include_router(cluster.router)
    app.include_router(simulation.router)
    app.include_router(profile_api.router)
    app.include_router(events_router)

    static_dir = static_dir if static_dir is not None else web_dir()
    if static_dir is not None and (static_dir / AppFiles.FALLBACK).is_file():
        app.mount("/", AppFiles(directory=static_dir, html=True), name="web")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def not_built() -> str:
            return _NOT_BUILT

    return app
