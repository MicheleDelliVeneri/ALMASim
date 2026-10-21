"""The command catalogue and validation (web console plan, S2.2/S2.3)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from almasim.jobs.catalog import catalog, validate
from almasim.jobs.model import ResourcesError
from almasim.jobs.policy import PolicyError, plan_job

router = APIRouter(prefix="/api/v2/commands", tags=["commands"])


class CommandLine(BaseModel):
    argv: list[str] = Field(..., min_length=1, description="The command line after `almasim`.")
    resources: dict[str, Any] = Field(default_factory=dict)


@router.get("")
def list_commands(request: Request) -> list[dict[str, Any]]:
    """Every command the console can run, with its parameters and where it runs."""
    return [asdict(c) for c in catalog(request.app.state.profile)]


@router.post("/validate")
def validate_command(body: CommandLine, request: Request) -> dict[str, Any]:
    """What the CLI's parser says about ``argv``, and what would be submitted."""
    problems = validate(body.argv)
    if problems:
        return {"ok": False, "problems": problems, "plan": None}
    try:
        plan = plan_job(body.argv, request.app.state.profile, body.resources)
    except (PolicyError, ResourcesError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "problems": [],
        "plan": {
            "command": plan.command,
            "executor": plan.executor,
            "argv": plan.argv,
            "resources": plan.resources.to_dict(),
        },
    }
