#!/bin/bash
# A shared ALMASim install for a team on a cluster (web console plan, M4.2).
#
#   scripts/install_shared.sh <prefix> <site profile template> [wheel]
#
# makes, under <prefix> (e.g. /SKAO_data/cycle11/almasim):
#   env/          a Python 3.12 environment with almasim, its web console and
#                 casatools/casatasks: the Python every job runs
#   site.toml     the site profile: the template, with that Python filled in
#   almasim.sh    what users source: puts almasim on PATH, names the profile
#
# The wheel (default: built here from this checkout, the web app included)
# needs `npm run build` in frontend/ first. Run it again with a new wheel to
# update: the environment is reinstalled in place, the profile rewritten.
# Needs uv, and internet to fetch Python and the packages (on zwicky: run it
# on a compute node).
set -euo pipefail

prefix=${1:?usage: install_shared.sh <prefix> <site profile template> [wheel]}
template=${2:?usage: install_shared.sh <prefix> <site profile template> [wheel]}
wheel=${3:-}
repo=$(cd "$(dirname "$0")/.." && pwd)

umask 002  # the group shares the install
mkdir -p "$prefix"

if [ -z "$wheel" ]; then
  [ -f "$repo/frontend/build/200.html" ] || { echo "build the web app first: cd frontend && npm ci && npm run build" >&2; exit 1; }
  rm -rf "$prefix/dist"
  ALMASIM_REQUIRE_WEB=1 uv build --wheel --out-dir "$prefix/dist" "$repo"
  wheel=$(ls "$prefix"/dist/almasim-*.whl)
fi

uv venv --allow-existing --python 3.12 "$prefix/env"
# The versions tested (uv.lock), not the newest: a newer casatools, say,
# can change what calibration gives.
uv export --project "$repo" --frozen --no-hashes --no-emit-project --extra web --extra casa \
  --no-default-groups > "$prefix/constraints.txt"
uv pip install --python "$prefix/env/bin/python" --constraints "$prefix/constraints.txt" \
  --reinstall-package almasim "$wheel[web,casa]"

python_line="executable = \"$prefix/env/bin/python\""
sed "s|^executable = .*|$python_line|" "$template" > "$prefix/site.toml"
grep -q "^$python_line" "$prefix/site.toml" || { echo "$template has no [python] executable line" >&2; exit 1; }

cat > "$prefix/almasim.sh" <<SH
# ALMASim, shared install: source this file (e.g. from ~/.bashrc).
export PATH="$prefix/env/bin:\$PATH"
export ALMASIM_SITE_PROFILE="$prefix/site.toml"
SH

version=$("$prefix/env/bin/python" -c "from almasim.__version__ import __version__; print(__version__)")
echo "installed almasim $version in $prefix"
echo "users: source $prefix/almasim.sh, then: almasim doctor --slurm; almasim serve"
