"""Update the team's shared install from a git ref (``almasim update-shared``).

``scripts/install_shared.sh`` makes the install and records where it came
from in ``install.env``. An update is one Slurm job, since it builds and
downloads: it clones the ref, builds the web app, runs that ref's own
``install_shared.sh`` with the recorded template, then ``almasim doctor``
with the new environment. So the install logic lives in one place, the
script, at the version being installed.
"""

from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

from .profile import Profile

RECORD_NAME = "install.env"
# What the job runs; it inherits the PATH of whoever submits it.
TOOLS = ("git", "npm", "uv")


class SharedInstallError(ValueError):
    """The install or its record is not usable for an update."""


@dataclass(frozen=True)
class Source:
    repo: str
    # Relative to the repo (found in the clone), or absolute.
    template: str
    commit: str = ""


def find_prefix(python_prefix: str = sys.prefix) -> Path | None:
    """The shared install this Python belongs to (its ``env/``), if it is one."""
    env = Path(python_prefix).resolve()
    prefix = env.parent
    if env.name == "env" and (prefix / "almasim.sh").is_file():
        return prefix
    return None


def read_source(prefix: Path) -> Source | None:
    """What ``install_shared.sh`` recorded, or None for an install made before it did."""
    path = prefix / RECORD_NAME
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return Source(values.get("repo", ""), values.get("template", ""), values.get("commit", ""))


def resolve_source(recorded: Source | None, repo: str | None, template: str | None) -> Source:
    """The recorded source with the command line's overrides."""
    base = recorded or Source("", "")
    source = Source(repo or base.repo, template or base.template, base.commit)
    missing = [name for name in ("repo", "template") if not getattr(source, name)]
    if missing:
        flags = " and ".join(f"--{name}" for name in missing)
        raise SharedInstallError(
            f"no {' or '.join(missing)} recorded in {RECORD_NAME} (an install made before "
            f"the record): pass {flags} once; the update records them"
        )
    return source


def job_script(prefix: Path, work: Path, source: Source, ref: str) -> str:
    """The batch script: clone ``ref``, build, install, check."""
    q = shlex.quote
    template = source.template
    template_arg = q(template) if Path(template).is_absolute() else f'"$src"/{q(template)}'
    return "\n".join(
        [
            "#!/bin/bash",
            "set -euo pipefail",
            "umask 002  # the group shares the install",
            f"src={q(str(work / 'src'))}",
            "trap 'rm -rf \"$src\"' EXIT",
            f'git clone --quiet --no-checkout {q(source.repo)} "$src"',
            # A branch (as on the remote), else a tag or a commit.
            f"ref={q(ref)}",
            'commit=$(git -C "$src" rev-parse --verify --quiet "origin/$ref^{commit}" \\',
            '         || git -C "$src" rev-parse --verify "$ref^{commit}")',
            'git -C "$src" checkout --quiet --detach "$commit"',
            'echo "building $(git -C "$src" log --oneline -1)"',
            '(cd "$src/frontend" && npm ci --no-audit --no-fund && npm run build)',
            f'"$src/scripts/install_shared.sh" {q(str(prefix))} {template_arg}',
            f"ALMASIM_SITE_PROFILE={q(str(prefix / 'site.toml'))} "
            f"{q(str(prefix / 'env' / 'bin' / 'almasim'))} doctor",
            "",
        ]
    )


def sbatch_command(profile: Profile, script: Path, log: Path) -> list[str]:
    """``sbatch --wait`` for the update job: it exits with the job's exit code."""
    cmd = [
        "sbatch",
        "--wait",
        "--parsable",
        "--job-name=almasim-update-shared",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=01:00:00",
        f"--output={log}",
    ]
    if profile.partition:
        cmd.append(f"--partition={profile.partition}")
    if profile.account:
        cmd.append(f"--account={profile.account}")
    return [*cmd, str(script)]


__all__ = [
    "RECORD_NAME",
    "TOOLS",
    "SharedInstallError",
    "Source",
    "find_prefix",
    "job_script",
    "read_source",
    "resolve_source",
    "sbatch_command",
]
