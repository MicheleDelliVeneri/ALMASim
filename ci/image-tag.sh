#!/usr/bin/env bash
# Print the CI image tag for the current checkout: a hash of every file that
# goes into the image, so a commit always maps to the image built from its own
# dependency pins.
set -euo pipefail
cd "$(dirname "$0")/.."
cat ci/image/Dockerfile pyproject.toml uv.lock \
    | sha256sum | cut -c1-16 | sed 's/^/deps-/'
