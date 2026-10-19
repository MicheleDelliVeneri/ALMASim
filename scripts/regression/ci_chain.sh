#!/usr/bin/env bash
# Run the pipeline chain on the trimmed golden fixture (MADIV-28 CI tier 1).
#
#   scripts/regression/ci_chain.sh FIXTURE_DIR RUN_DIR
#
# FIXTURE_DIR is an extracted almasim-golden-ci bundle (raw/, auxiliary/).
# RUN_DIR receives raw_ms/, cal/, params.csv, cubes/ and logs/, laid out like a
# full-size golden run so compare_products.py golden can check it.
#
# Tools come from the environment (defaults suit the CI image):
#   PYTHON, WSCLEAN_BIN, CASA_DATA, CORES, TASK_MEMORY_GB, SELFCAL_MIN_SNR
set -euo pipefail
FIXTURE="$(cd "${1:?usage: ci_chain.sh FIXTURE_DIR RUN_DIR}" && pwd)"
RUN="${2:?usage: ci_chain.sh FIXTURE_DIR RUN_DIR}"
UID_EB=uid___A002_X113d9b7_Xb670
PYTHON="${PYTHON:-python}"
WSCLEAN_BIN="${WSCLEAN_BIN:-$(command -v wsclean)}"
CASA_DATA="${CASA_DATA:-/opt/casa-data}"
CORES="${CORES:-4}"
TASK_MEMORY_GB="${TASK_MEMORY_GB:-12}"
# The trimmed data has a much lower S/N than the full window; a low threshold
# keeps the self-calibration path exercised.
SELFCAL_MIN_SNR="${SELFCAL_MIN_SNR:-3}"
export OPENBLAS_NUM_THREADS=1

mkdir -p "$RUN/logs" "$RUN/archives"
RUN="$(cd "$RUN" && pwd)"
declare -A STAGE_SECONDS
stage() {  # stage NAME COMMAND...
    local name="$1"; shift
    echo "== $name $(date --iso-8601=seconds)"
    local start=$SECONDS
    if ! "$@" > "$RUN/logs/$name.log" 2>&1; then
        echo "FAIL: $name (log: $RUN/logs/$name.log)"
        tail -40 "$RUN/logs/$name.log"
        exit 1
    fi
    STAGE_SECONDS[$name]=$((SECONDS - start))
    echo "   $name finished in ${STAGE_SECONDS[$name]} s"
}

(cd "$FIXTURE" && sha256sum --quiet -c SHA256SUMS)

raw_tar="$(ls "$FIXTURE"/raw/*.tar)"
ln -sfn "$raw_tar" "$RUN/archives/$(basename "$raw_tar")"
if [[ "$raw_tar" == *.asdm.sdm.tar ]]; then
    stage extract "$PYTHON" -m almasim.cli visibilities extract \
        --source-root "$RUN/archives" --destination "$RUN/asdm"
    stage unpack "$PYTHON" -m almasim.cli visibilities unpack \
        --input-root "$RUN/asdm" --output-root "$RUN/raw_ms" --asdm-uid "$UID_EB" \
        --postprocess-backend sync --casa-data-root "$CASA_DATA" --skip-casa-data-update
else
    mkdir -p "$RUN/raw_ms/working"
    stage extract tar -xf "$raw_tar" -C "$RUN/raw_ms/working"
fi

stage calibrate "$PYTHON" -m almasim.cli visibilities calibrate --postprocess-backend sync \
    --input-root "$FIXTURE/auxiliary" --raw-ms-root "$RUN/raw_ms/working" \
    --output-root "$RUN/cal" --asdm-uid "$UID_EB" \
    --casa-data-root "$CASA_DATA" --skip-casa-data-update

stage compute-parameters "$PYTHON" -m almasim.cli image compute-parameters \
    "$RUN/cal" "$RUN/params.csv"

stage image "$PYTHON" -m almasim.cli image image-from-ms "$RUN/params.csv" "$RUN/cubes" \
    --postprocess-backend sync --wsclean-bin "$WSCLEAN_BIN" \
    --num-cores "$CORES" --task-memory-gb "$TASK_MEMORY_GB" --task-timeout 3600 \
    --cube --uv-contsub --selfcal inf,30s,ap:inf --selfcal-min-snr "$SELFCAL_MIN_SNR" \
    --noise-products --casa-data-root "$CASA_DATA"

stage clean-quality "$PYTHON" -m almasim.cli image clean-quality "$RUN/cubes"

echo "stage seconds: $(for s in "${!STAGE_SECONDS[@]}"; do printf '%s=%s ' "$s" "${STAGE_SECONDS[$s]}"; done)"
echo "total seconds: $SECONDS"
