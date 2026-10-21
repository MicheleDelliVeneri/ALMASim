# Security Policy

## Supported versions

Security fixes are made on the `main` branch and released in the next
version of the 2.1 series.

| Version | Supported |
|---|---|
| 2.1.x (latest release) | ✅ |
| older releases | ❌ |

## Reporting a vulnerability

**Please do not report security vulnerabilities in public issues,
discussions or pull requests.**

Report them privately through GitHub:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the issue, the affected component and version (or commit), and
   the steps to reproduce it. Include a proof of concept if you have one.

We aim to acknowledge reports within **5 working days** and to send a first
assessment within **10 working days**. We will keep you informed while we
work on a fix and agree with you on when it is disclosed. Reporters are
credited in the advisory unless they prefer otherwise.

## Scope

In scope:

- the `almasim` Python package and its command-line interface;
- the web console (`almasim serve`, `src/almasim/web/`) and its pages
  (`frontend/`);
- the Docker images and compose files in this repository;
- the GitHub Actions workflows in `.github/workflows/`.

Out of scope:

- vulnerabilities in third-party software ALMASim runs or depends on
  (WSClean, CASA, Dask, Slurm, Python or npm packages); please report those
  upstream. Tell us as well if ALMASim's use of them makes the issue
  exploitable;
- issues that need an attacker who already controls the user's account or
  machine;
- the scientific correctness of simulated or imaged data (please open a
  normal issue).

## Running ALMASim securely

- **Dask on Slurm.** Batch commands start a Dask scheduler and dashboard
  (port 8787) without authentication, reachable by every host that can reach
  the address it binds to. Run them only on trusted cluster networks, do not
  expose the scheduler or dashboard ports to the internet, and use
  `--scheduler-host` / `--scheduler-interface` to bind to an internal
  interface where possible.
- **Web console.** `almasim serve` listens on a socket only its user can
  open, or on a loopback port, and asks for a token. Reach it through an SSH
  tunnel; do not bind it to a public interface. Behind anything wider, put an
  authenticating reverse proxy with TLS in front of it.
- **Credentials.** Never commit API keys, tokens or `kaggle.json`. In CI they
  are provided only through repository secrets. Use fine-grained tokens with
  the minimum permissions needed.
- **Downloaded data.** Archive downloads, extracted MeasurementSets and
  pipeline products are untrusted input: extract them into their own
  directories and do not execute scripts shipped with them without reviewing
  them.
- **CASA runtime data** is downloaded from the official CASA data
  repositories only when missing; to avoid network downloads on compute
  nodes, pre-populate it and pass `--casa-data-root`.

## Dependencies

Python dependencies are pinned in `uv.lock`, and npm dependencies in the
frontend lock file. Dependabot checks the Python, documentation and npm
dependencies weekly, and its updates go through the same CI (lint, type
checks, tests and coverage) as any other change.
