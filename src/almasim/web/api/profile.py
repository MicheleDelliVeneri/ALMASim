"""Your profile, from the Settings page (web console plan, U3.8).

The site profile is shown as it is; it belongs to whoever runs the shared
install. Your own profile (``ALMASIM_PROFILE``, else
``~/.config/almasim/almasim.toml``) can be replaced: the new text is read as
a profile together with the site's first, and saved only if it is valid.
The console then uses it at once, for the next runs and for what it shows.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from almasim.jobs.profile import Profile, ProfileError, load_profile, profile_paths

router = APIRouter(prefix="/api/v2/profile", tags=["profile"])

MAX_PROFILE = 64 * 1024


class ProfileFile(BaseModel):
    path: str | None
    exists: bool
    text: str


class Effective(BaseModel):
    """What the merged profile says, as the console uses it."""

    runs_root: str
    roots: list[str]
    executor: str | None
    partition: str | None
    account: str | None
    python: str
    wsclean: str | None
    casa_data: str | None
    partitions: list[str]
    commands: list[str]


class Profiles(BaseModel):
    site: ProfileFile
    user: ProfileFile
    effective: Effective


class UserProfile(BaseModel):
    text: str = Field(..., max_length=MAX_PROFILE)


def _file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "text": ""}
    exists = path.is_file()
    return {"path": str(path), "exists": exists, "text": path.read_text() if exists else ""}


def _paths(profile: Profile) -> tuple[Path | None, Path]:
    site, user = profile_paths()
    return profile.site_file or site, profile.user_file or user


def _view(profile: Profile) -> dict[str, Any]:
    site, user = _paths(profile)
    return {
        "site": _file(site),
        "user": _file(user),
        "effective": {
            "runs_root": str(profile.runs_root),
            "roots": [str(r) for r in profile.roots],
            "executor": profile.executor,
            "partition": profile.partition,
            "account": profile.account,
            "python": profile.python,
            "wsclean": profile.wsclean,
            "casa_data": str(profile.casa_data) if profile.casa_data else None,
            "partitions": sorted(profile.partitions),
            "commands": sorted(profile.commands),
        },
    }


@router.get("", response_model=Profiles)
def get_profiles(request: Request) -> dict[str, Any]:
    """The site and user profiles' text, and what they add up to."""
    return _view(request.app.state.profile)


@router.put("/user", response_model=Profiles)
def put_user_profile(body: UserProfile, request: Request) -> dict[str, Any]:
    """Replace your profile, if the new text makes a valid profile with the site's."""
    site, user = _paths(request.app.state.profile)
    user.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".almasim-", suffix=".toml", dir=user.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body.text)
        try:
            load_profile(site=site, user=tmp)
        except ProfileError as exc:
            raise HTTPException(
                status_code=400, detail=str(exc).replace(str(tmp), str(user))
            ) from exc
        tmp.chmod(0o644)
        tmp.replace(user)
    finally:
        tmp.unlink(missing_ok=True)
    profile = load_profile(site=site, user=user)
    request.app.state.profile = profile
    request.app.state.events.profile = profile
    request.app.state.cluster.profile = profile
    request.app.state.cluster.forget()
    return _view(profile)
