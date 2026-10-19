#!/usr/bin/env bash
# Seeded simulation golden run (MADIV-28 T0.17).
#
#   scripts/regression/sim_golden.sh OUT_DIR [SOURCE_TYPE ...]
#
# Runs one small simulation per source type (default: point gaussian diffuse
# molecular; the dataset-backed types need downloads) with NumPy and Python's
# random module seeded, so two runs of the same commit are bit-identical
# (checked on zwicky: 39 of 39 products each). OUT_DIR/<type>/ holds the
# products, OUT_DIR/logs/ the logs.
set -euo pipefail
OUT="${1:?usage: sim_golden.sh OUT_DIR [SOURCE_TYPE ...]}"; shift
TYPES=("$@"); [ ${#TYPES[@]} -gt 0 ] || TYPES=(point gaussian diffuse molecular)
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p "$OUT/logs"
for source_type in "${TYPES[@]}"; do
    start=$SECONDS
    "$PYTHON" -c "import random, sys; random.seed(0); sys.argv = ['almasim'] + sys.argv[1:]; from almasim.cli import main; main()" \
        simulation run --metadata-path "$REPO/data/qso_metadata.csv" --row-idx 0 \
        --main-dir "$REPO/src/almasim" --output-dir "$OUT/$source_type" --project-name golden \
        --source-type "$source_type" --n-pix 64 --n-channels 16 --seed 42 --background-seed 7 \
        --save-mode fits --backend sync > "$OUT/logs/$source_type.log" 2>&1 \
        || { echo "FAIL: $source_type"; tail -30 "$OUT/logs/$source_type.log"; exit 1; }
    echo "$source_type: $((SECONDS - start)) s, $(find "$OUT/$source_type" -type f | wc -l) files"
done
