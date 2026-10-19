#!/usr/bin/env bash
# Package a golden run's products as a baseline bundle (MADIV-28).
#
#   scripts/regression/package_baseline.sh RUN_DIR OUT_DIR NAME "PROVENANCE"
#
# Writes OUT_DIR/NAME.tar.gz (raw.json, cal.json, prefixes.env, PROVENANCE and
# cubes/<uid>.ms.split/<task>{,.done,.log}) and its .sha256. RUN_DIR must hold
# digests/ from compare_products.py --save-digests and the image products.
set -euo pipefail
RUN="$(cd "${1:?usage: package_baseline.sh RUN_DIR OUT_DIR NAME PROVENANCE}" && pwd)"
OUT="${2:?}"; NAME="${3:?}"; PROVENANCE="${4:?}"
UID_EB=uid___A002_X113d9b7_Xb670
TASK="$(cd "$RUN/cubes/$UID_EB.ms.split" && ls -d SPW-*/ | head -1 | tr -d /)"
STAGE="$(mktemp -d)/$NAME"
mkdir -p "$STAGE/cubes/$UID_EB.ms.split" "$OUT"
cp "$RUN/digests/raw.json" "$RUN/digests/cal.json" "$STAGE/"
cp -r "$RUN/cubes/$UID_EB.ms.split/$TASK" "$RUN/cubes/$UID_EB.ms.split/$TASK".{done,log} \
    "$STAGE/cubes/$UID_EB.ms.split/"
printf '%s\n' "BASELINE_PREFIX_RAW=$RUN/raw_ms" "BASELINE_PREFIX_CAL=$RUN/cal" \
    "BASELINE_PREFIX_CUBES=$RUN/cubes" "BASELINE_PREFIX_ASDM=$RUN/asdm" > "$STAGE/prefixes.env"
printf '%s\n' "$TASK" > "$STAGE/TASK"
printf '%s\n' "$PROVENANCE" > "$STAGE/PROVENANCE"
tar -czf "$OUT/$NAME.tar.gz" -C "$(dirname "$STAGE")" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "$OUT/$NAME.tar.gz"
