"""Check that a wheel carries the web app, and that an install of it serves it (S2.6).

python scripts/check_web_wheel.py dist/almasim-*.whl     # what the wheel holds
python scripts/check_web_wheel.py --serve                  # the installed almasim serves it

The second form runs in an environment where the wheel is installed with
the web extra (and httpx, for the test client), outside the source tree, so
the packaged copy is the one found.
"""

from __future__ import annotations

import sys
import zipfile


def check_wheel(path: str) -> None:
    names = set(zipfile.ZipFile(path).namelist())
    page = "almasim/web/static/200.html"
    if page not in names:
        sys.exit(f"{path}: no {page}: the web app was not built into the wheel")
    assets = [n for n in names if n.startswith("almasim/web/static/_app/immutable/")]
    if not assets:
        sys.exit(f"{path}: the web app's scripts are missing")
    print(f"{path}: web app present ({len(assets)} asset files)")


def check_serve() -> None:
    from fastapi.testclient import TestClient

    from almasim.jobs.profile import Profile
    from almasim.web.app import COOKIE, STATIC_DIR, create_app, web_dir

    found = web_dir({})
    if found != STATIC_DIR:
        sys.exit(f"the installed almasim finds its web app at {found}, not {STATIC_DIR}")
    client = TestClient(create_app("check-token-0123456789", profile=Profile()))
    client.cookies.set(COOKIE, "check-token-0123456789")
    for path in ("/", "/jobs/01JCHECK"):
        page = client.get(path)
        if page.status_code != 200 or "_app/immutable" not in page.text:
            sys.exit(f"GET {path}: {page.status_code}, not the web app")
    print(f"the installed almasim serves its web app from {found}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        check_serve()
    elif len(sys.argv) == 2:
        check_wheel(sys.argv[1])
    else:
        sys.exit(__doc__)
