#!/usr/bin/env bash
# Turn a kept golden run into a versioned, read-only full-size baseline.
#
#   scripts/regression/promote_baseline.sh KEPT_RUN_DIR NAME "REASON"
#
# KEPT_RUN_DIR is $MADIV28/regression/kept/<run> (a run made with GOLDEN_KEEP=1)
# and NAME the new baseline, e.g. v1-807e700f. The baseline is written to
# $MADIV28/baselines/NAME; point BASELINE_DIR in golden.env at it in the PR
# that supersedes the previous one.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=golden.env
source "$HERE/golden.env"
KEPT="${1:?usage: promote_baseline.sh KEPT_RUN_DIR NAME REASON}"
NAME="${2:?baseline name, e.g. v1-807e700f}"
REASON="${3:?reason}"
DEST="$MADIV28/baselines/$NAME"
[ -e "$DEST" ] && { echo "$DEST already exists" >&2; exit 1; }
for f in digests/raw.json digests/cal.json prefixes.env run.txt \
         "cubes/$GOLDEN_UID.ms.split/$GOLDEN_TASK"; do
    [ -e "$KEPT/$f" ] || { echo "missing $KEPT/$f" >&2; exit 1; }
done
umask 0002
mkdir -p "$DEST/cubes/$GOLDEN_UID.ms.split"
cp "$KEPT/digests/raw.json" "$KEPT/digests/cal.json" "$KEPT/prefixes.env" "$DEST/"
cp -r "$KEPT/cubes/$GOLDEN_UID.ms.split/$GOLDEN_TASK" \
      "$KEPT/cubes/$GOLDEN_UID.ms.split/$GOLDEN_TASK".{done,log} "$DEST/cubes/$GOLDEN_UID.ms.split/"
{
    cat "$KEPT/run.txt"
    echo "promoted=$(date --iso-8601=seconds)"
    echo "from=$KEPT"
    echo "reason=$REASON"
} > "$DEST/PROVENANCE"
chmod -R a-w "$DEST"
echo "baseline $NAME written to $DEST"
