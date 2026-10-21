"""Runs of ``almasim`` commands as jobs: one folder per run, shared by the CLI and the server.

A job is one ``almasim`` command with its arguments, the resources it asked
for and where it runs (a Slurm job, or a process beside the server on sites whose compute
nodes have no internet). Everything about it lives in its run
folder, so the folder is the record: a restarted server, a teammate's
server and the terminal all read the same files (web console plan, J1.1).
"""

from .model import (
    EXECUTORS,
    MANIFEST_SCHEMA,
    Job,
    JobState,
    Resources,
    ResourcesError,
    command_name,
    is_job_id,
    new_job_id,
)
from .rundir import (
    MANIFEST_NAME,
    STATUS_NAME,
    ManifestError,
    RunStatus,
    create_run_dir,
    current_state,
    iter_runs,
    read_manifest,
    read_status,
    write_manifest,
    write_status,
)

__all__ = [
    "EXECUTORS",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "STATUS_NAME",
    "Job",
    "JobState",
    "ManifestError",
    "Resources",
    "ResourcesError",
    "RunStatus",
    "command_name",
    "create_run_dir",
    "current_state",
    "is_job_id",
    "iter_runs",
    "new_job_id",
    "read_manifest",
    "read_status",
    "write_manifest",
    "write_status",
]
