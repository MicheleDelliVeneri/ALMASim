"""Runs as the API returns them (web console plan, S2.3)."""

from __future__ import annotations

from typing import Any

from almasim.jobs.service import RunView, short_id


def run_json(run: RunView) -> dict[str, Any]:
    job, status = run.job, run.status
    return {
        "id": job.id,
        "short_id": short_id(job.id),
        "owner": job.owner,
        "command": job.command,
        "argv": job.argv,
        "title": job.title,
        "executor": job.executor,
        "resources": job.resources.to_dict(),
        "params": job.params,
        "parents": job.parents,
        "slurm_job_id": job.slurm_job_id,
        "state": run.state.value,
        "finished": run.state.finished,
        "reason": run.reason,
        "created_at": job.created_at,
        "started_at": status.started_at,
        "finished_at": status.finished_at,
        "host": status.host,
        "exit_code": status.exit_code,
        "signal": status.signal,
        "run_dir": str(run.run_dir),
    }
