#!/bin/bash
# Start an iteration over the research subset:
#   launch.sh NAME CONFIG.json [COUNT [ARRAY]]    (config: {"mode": "control"|"sim", "seed": N, "errors": {...}})
set -euo pipefail
SBATCH_EXTRA=${SBATCH_EXTRA:-}
ITER=/SKAO_data/cycle11/MADIV-28/noise/iter-$1
mkdir -p "$ITER/logs"
cp "$2" "$ITER/config.json"
N=$(cd "$(dirname "$0")" && pwd)
COUNT=${3:-20}  # the research subset (window.py subset)
ARRAY=${4:-0-$((COUNT - 1))}
# NOISE_SET=all (all 100 windows) passes through to the jobs; SBATCH_EXTRA adds
# sbatch options, e.g. more memory and time for the big windows.
# shellcheck disable=SC2086
sbatch --parsable --array="$ARRAY" $SBATCH_EXTRA -o "$ITER/logs/%a.log" "$N/window.sh" "$ITER"
