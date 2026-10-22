#!/bin/bash
#SBATCH -J noise-uv -c 8 --mem=48G -t 02:00:00 -p zwickyA-CPU
# The visibility comparison of one window (uvcompare.py) for several error models:
#   sbatch --array=0-19 -o OUT/logs/%a.log uvwindow.sh OUT CONFIG.json [CONFIG.json ...]
# (NOISE_SET=all indexes all 100 windows; the window's base is made first if missing.)
set -euo pipefail
OUT=$1; shift
I=${SLURM_ARRAY_TASK_ID}
export OPENBLAS_NUM_THREADS=1
R=/home1/mdelliveneri/ALMASim; PY=$R/.venv/bin/python; N=$R/research/noise_realism
cd "$N"
mkdir -p "$OUT"
eval "$($PY window.py info "$I")"
BASE=/SKAO_data/cycle11/MADIV-28/noise/base/$NAME.ms
(
    flock 9
    [ -f "$BASE/noise.json" ] || $PY counterpart.py base "$MS" "$SPW" "$FIELD" "$BASE"
) 9> "/SKAO_data/cycle11/MADIV-28/noise/base/$NAME.lock"
$PY uvcompare.py window "$I" "$@" "$OUT/$I.json"
echo "DONE $I"
