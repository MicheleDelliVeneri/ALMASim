#!/bin/bash
#SBATCH --partition=zwickyA-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# MADIV-28 golden run: extract -> unpack -> calibrate -> image -> clean-quality
# on one ALMA window, then compare every product with the references and clean
# up. Submit with submit_golden.sh, which prepares $RUN; see the "Regression
# gate" section of the MADIV-28 plan.
#
# Imaging is single-threaded by default (GOLDEN_CORES=1): WSClean's
# multi-threaded gridding is not bit-reproducible, so two -j 10 runs of the same
# commit differ. MADIV-16 used --num-cores 10.
#
# If a stage fails the run directory is left in place. GOLDEN_FROM=compare
# (sbatch --export=ALL,GOLDEN_FROM=compare golden_run.sbatch RUN) then repeats
# only the digests, comparison and clean-up on the products already there.
set -euo pipefail
umask 0002
RUN="${1:?usage: golden_run.sbatch RUN_DIR}"
# shellcheck source=golden.env
source "$RUN/harness/golden.env"

export PYTHONPATH="$RUN/src/src"
export OPENBLAS_NUM_THREADS=1
cd "$RUN"
echo "golden run $RUN on $(hostname), job ${SLURM_JOB_ID:-local}, started $(date --iso-8601=seconds)"
cat "$RUN/run.txt"
"$PYTHON" -c "import almasim, sys; assert almasim.__file__.startswith('$RUN/src/'), almasim.__file__; print('almasim from', almasim.__file__)"

declare -A STAGE_SECONDS
stage() {  # stage NAME COMMAND...
    local name="$1"; shift
    echo "== $name $(date --iso-8601=seconds)"
    local start=$SECONDS
    "$@" > "$RUN/logs/$name.log" 2>&1 || { echo "FAIL: $name exited $? (log: $RUN/logs/$name.log)"; tail -20 "$RUN/logs/$name.log"; exit 1; }
    STAGE_SECONDS[$name]=$((SECONDS - start))
    echo "   $name finished in ${STAGE_SECONDS[$name]} s"
}
mkdir -p "$RUN/logs" "$RUN/archives"

if [ "${GOLDEN_FROM:-start}" = start ]; then
stage fixture-checksum bash -c "cd '$(dirname "$FIXTURE_TAR")' && sha256sum -c '$(basename "$FIXTURE_TAR").sha256'"
# extract writes <archive>.done beside the archive: point it at a per-run link.
ln -sfn "$FIXTURE_TAR" "$RUN/archives/$(basename "$FIXTURE_TAR")"

stage extract "$PYTHON" -m almasim.cli visibilities extract \
    --source-root "$RUN/archives" --destination "$RUN/asdm"

stage unpack "$PYTHON" -m almasim.cli visibilities unpack \
    --input-root "$RUN/asdm" --output-root "$RUN/raw_ms" --asdm-uid "$GOLDEN_UID" \
    --postprocess-backend sync --casa-data-root "$CASA_DATA" --skip-casa-data-update

stage calibrate "$PYTHON" -m almasim.cli visibilities calibrate --postprocess-backend sync \
    --input-root "$AUX_ROOT" --raw-ms-root "$RUN/raw_ms/working" --output-root "$RUN/cal" \
    --asdm-uid "$GOLDEN_UID" --casa-data-root "$CASA_DATA" --skip-casa-data-update

# Imaging parameters computed from this run's calibrated MS, like a user run, so
# changes to compute-parameters are gated too (they used to be the frozen
# MADIV-16 row). Only the golden task's row is kept.
stage compute-parameters "$PYTHON" -m almasim.cli image compute-parameters \
    "$RUN/cal" "$RUN/all-params.csv"
"$PYTHON" - "$RUN/all-params.csv" "$RUN/params.csv" "$GOLDEN_TASK" <<'PY'
import csv, re, sys
src, dst, task = sys.argv[1:]
spw, field = re.fullmatch(r"SPW-(\d+)_FIELD-(\d+)", task).groups()
rows = [
    r for r in csv.DictReader(open(src))
    if r["spectral_window_id"] == spw and r["target_field_ids"] == field
]
assert len(rows) == 1, f"expected one params row for {task}, found {len(rows)}"
with open(dst, "w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY

stage image "$PYTHON" -m almasim.cli image image-from-ms "$RUN/params.csv" "$RUN/cubes" \
    --postprocess-backend sync --wsclean-bin "$WSCLEAN_BIN" \
    --num-cores "${GOLDEN_CORES:-1}" --task-memory-gb 36 --task-timeout 28800 \
    --cube --uv-contsub --selfcal inf,30s,ap:inf --noise-products --casa-data-root "$CASA_DATA"

stage clean-quality "$PYTHON" -m almasim.cli image clean-quality "$RUN/cubes"
else
    echo "resuming at the comparison (GOLDEN_FROM=$GOLDEN_FROM)"
fi

# The baseline to compare with: a directory holding raw.json and cal.json (MS
# digests), cubes/<uid>.ms.split/<task> (image products) and prefixes.env (the
# paths its products were written under). GOLDEN_BASELINE overrides golden.env.
BASELINE_DIR="${GOLDEN_BASELINE:-$BASELINE_DIR}"
# shellcheck source=/dev/null
source "$BASELINE_DIR/prefixes.env"
echo "baseline: $BASELINE_DIR"

echo "== compare $(date --iso-8601=seconds)"
set +e
"$PYTHON" -I "$RUN/harness/compare_products.py" golden --run "$RUN" --uid "$GOLDEN_UID" \
    --task "$GOLDEN_TASK" --ref-raw "$BASELINE_DIR/raw.json" --ref-cal "$BASELINE_DIR/cal.json" \
    --ref-task "$BASELINE_DIR/cubes/$GOLDEN_UID.ms.split/$GOLDEN_TASK" \
    --map RAW "$RUN/raw_ms" "$BASELINE_PREFIX_RAW" \
    --map CAL "$RUN/cal" "$BASELINE_PREFIX_CAL" \
    --map CUBES "$RUN/cubes" "$BASELINE_DIR/cubes,$BASELINE_PREFIX_CUBES" \
    --map ASDM "$RUN/asdm" "$BASELINE_PREFIX_ASDM" \
    --save-digests "$RUN/digests" \
    --report "$RUN/report.md"
verdict=$?
set -e
{
    echo
    echo "baseline: $BASELINE_DIR"
    echo "stage seconds: $(for s in "${!STAGE_SECONDS[@]}"; do printf '%s=%s ' "$s" "${STAGE_SECONDS[$s]}"; done)"
    echo "total seconds: $SECONDS"
} >> "$RUN/report.md"

# Clean up. MS copies are always removed. On PASS nothing else is kept unless
# GOLDEN_KEEP=1 (a candidate baseline, see promote_baseline.sh); on FAIL the
# report, logs, digests and image products are kept for inspection.
cp "$RUN/report.md" "$MADIV28/regression/logs/$(basename "$RUN").report.md"
if [ "$verdict" -eq 0 ] && [ "${GOLDEN_KEEP:-0}" != 1 ]; then
    echo "PASS: removing $RUN"
    rm -rf "$RUN"
else
    [ "$verdict" -eq 0 ] && KEPT="$MADIV28/regression/kept/$(basename "$RUN")" \
        || KEPT="$MADIV28/regression/failed/$(basename "$RUN")"
    mkdir -p "$KEPT"
    cp -r "$RUN/report.md" "$RUN/run.txt" "$RUN/logs" "$RUN/params.csv" "$KEPT/"
    for d in digests cubes; do [ -d "$RUN/$d" ] && cp -r "$RUN/$d" "$KEPT/"; done
    printf '%s\n' \
        "BASELINE_PREFIX_RAW=$RUN/raw_ms" \
        "BASELINE_PREFIX_CAL=$RUN/cal" \
        "BASELINE_PREFIX_CUBES=$RUN/cubes" \
        "BASELINE_PREFIX_ASDM=$RUN/asdm" > "$KEPT/prefixes.env"
    rm -rf "$RUN"
    echo "$([ "$verdict" -eq 0 ] && echo PASS || echo FAIL): kept report, logs, digests and image products in $KEPT"
fi
echo "finished $(date --iso-8601=seconds)"
exit "$verdict"
