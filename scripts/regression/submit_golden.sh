#!/usr/bin/env bash
# Submit one MADIV-28 golden run for a git ref of this repository.
#
#   scripts/regression/submit_golden.sh [GIT_REF] [NOTE]
#
# Extra sbatch options (for example --dependency=afterany:JOBID) can be passed
# in SBATCH_EXTRA.
#
# Exports GIT_REF into $MADIV28/regression/<date>-<sha>/src, copies this
# harness next to it (so later edits to the working tree cannot change a
# queued run) and submits golden_run.sbatch on it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
# shellcheck source=golden.env
source "$HERE/golden.env"

REF="${1:-HEAD}"
NOTE="${2:-}"
SHA="$(git -C "$REPO" rev-parse --short=8 "$REF^{commit}")"
RUN="$MADIV28/regression/$(date +%Y%m%d-%H%M%S)-$SHA"

umask 0002
mkdir -p "$RUN/src" "$RUN/harness" "$MADIV28/regression/logs"
git -C "$REPO" archive "$SHA" | tar -x -C "$RUN/src"
cp "$HERE"/{golden.env,golden_run.sbatch,compare_products.py} "$RUN/harness/"
printf 'ref=%s\nsha=%s\nnote=%s\nsubmitted=%s\n' "$REF" "$(git -C "$REPO" rev-parse "$SHA")" \
    "$NOTE" "$(date --iso-8601=seconds)" > "$RUN/run.txt"

# shellcheck disable=SC2086
sbatch --parsable ${SBATCH_EXTRA:-} --job-name "golden-$SHA" \
    --output "$MADIV28/regression/logs/golden-$SHA-%j.log" \
    "$RUN/harness/golden_run.sbatch" "$RUN"
echo "run directory: $RUN"
