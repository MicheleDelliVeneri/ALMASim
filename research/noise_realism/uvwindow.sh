#!/bin/bash
#SBATCH -J noise-uv -c 8 --mem=48G -t 02:00:00 -p zwickyA-CPU
# The visibility comparison of one window (uvcompare.py) for several error models:
#   sbatch --array=0-19 -o OUT/logs/%a.log uvwindow.sh OUT CONFIG.json [CONFIG.json ...]
set -euo pipefail
OUT=$1; shift
I=${SLURM_ARRAY_TASK_ID}
export OPENBLAS_NUM_THREADS=1
cd /home1/mdelliveneri/ALMASim/research/noise_realism
mkdir -p "$OUT"
/home1/mdelliveneri/ALMASim/.venv/bin/python uvcompare.py window "$I" "$@" "$OUT/$I.json"
echo "DONE $I"
