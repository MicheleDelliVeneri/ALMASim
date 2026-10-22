#!/bin/bash
# Start an iteration over the research subset:
#   launch.sh NAME CONFIG.json [COUNT [ARRAY]]    (config: {"mode": "control"|"sim", "seed": N, "errors": {...}})
set -euo pipefail
ITER=/SKAO_data/cycle11/MADIV-28/noise/iter-$1
mkdir -p "$ITER/logs"
cp "$2" "$ITER/config.json"
N=$(cd "$(dirname "$0")" && pwd)
COUNT=${3:-20}  # the research subset (window.py subset)
ARRAY=${4:-0-$((COUNT - 1))}
sbatch --parsable --array="$ARRAY" -o "$ITER/logs/%a.log" "$N/window.sh" "$ITER"
