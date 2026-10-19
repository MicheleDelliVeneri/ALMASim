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

# The reference imaging parameters, pointed at this run's calibrated MS.
"$PYTHON" - "$REF_PARAMS" "$RUN/params.csv" "$GOLDEN_UID" "$RUN/cal" <<'PY'
import csv, sys
src, dst, uid, cal_root = sys.argv[1:]
rows = [r for r in csv.DictReader(open(src)) if f"/{uid}.ms" in r["filename"]]
assert len(rows) == 1, f"expected one params row for {uid}, found {len(rows)}"
rows[0]["filename"] = f"{cal_root}/{uid}.ms.split.cal"
with open(dst, "w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
PY

stage image "$PYTHON" -m almasim.cli image image-from-ms "$RUN/params.csv" "$RUN/cubes" \
    --postprocess-backend sync --wsclean-bin "$WSCLEAN_BIN" \
    --num-cores 10 --task-memory-gb 36 --task-timeout 28800 \
    --cube --uv-contsub --selfcal inf,30s,ap:inf --noise-products --casa-data-root "$CASA_DATA"

stage clean-quality "$PYTHON" -m almasim.cli image clean-quality "$RUN/cubes"
else
    echo "resuming at the comparison (GOLDEN_FROM=$GOLDEN_FROM)"
fi

# Reference digests are computed once (reading 20 GB) and cached read-only.
DIGESTS="$MADIV28/fixture-digests"
mkdir -p "$DIGESTS"
MAPS=(--map RAW "" "$REF_RAW_ROOT" --map CAL "" "$REF_CAL_ROOT" --map CUBES "" "$REF_CUBES" --map ASDM "" "$REF_ASDM_ROOT")
for kind in raw cal; do
    ms="$REF_RAW_MS"; [ "$kind" = cal ] && ms="$REF_CAL_MS"
    if [ ! -s "$DIGESTS/$kind.json" ]; then
        tmp="$DIGESTS/.$kind.json.$$"
        stage "digest-ref-$kind" "$PYTHON" -I "$RUN/harness/compare_products.py" digest "$ms" "$tmp" "${MAPS[@]}"
        mv -n "$tmp" "$DIGESTS/$kind.json" && chmod a-w "$DIGESTS/$kind.json"
        rm -f "$tmp"
    fi
done

echo "== compare $(date --iso-8601=seconds)"
set +e
"$PYTHON" -I "$RUN/harness/compare_products.py" golden --run "$RUN" --uid "$GOLDEN_UID" \
    --task "$GOLDEN_TASK" --ref-raw "$DIGESTS/raw.json" --ref-cal "$DIGESTS/cal.json" \
    --ref-task "$REF_TASK" \
    --map RAW "$RUN/raw_ms" "$REF_RAW_ROOT" \
    --map CAL "$RUN/cal" "$REF_CAL_ROOT" \
    --map CUBES "$RUN/cubes" "$REF_CUBES,$CYCLE11/MADIV-16/noise/cubes" \
    --map ASDM "$RUN/asdm" "$REF_ASDM_ROOT" \
    --report "$RUN/report.md"
verdict=$?
set -e
{
    echo
    echo "stage seconds: $(for s in "${!STAGE_SECONDS[@]}"; do printf '%s=%s ' "$s" "${STAGE_SECONDS[$s]}"; done)"
    echo "total seconds: $SECONDS"
} >> "$RUN/report.md"

# Clean up: everything on PASS; on FAIL keep the report, logs and image products.
if [ "$verdict" -eq 0 ]; then
    echo "PASS: removing $RUN"
    cp "$RUN/report.md" "$MADIV28/regression/logs/$(basename "$RUN").report.md"
    rm -rf "$RUN"
else
    FAILED="$MADIV28/regression/failed/$(basename "$RUN")"
    mkdir -p "$FAILED"
    cp -r "$RUN/report.md" "$RUN/run.txt" "$RUN/logs" "$RUN/params.csv" "$FAILED/"
    [ -d "$RUN/cubes" ] && cp -r "$RUN/cubes" "$FAILED/"
    rm -rf "$RUN"
    echo "FAIL: kept report, logs and image products in $FAILED"
fi
echo "finished $(date --iso-8601=seconds)"
exit "$verdict"
