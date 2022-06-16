#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-16
MAIN_PATH="$PWD"

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate casa6.5


LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p sims_param.csv)
IFS=',' read INDEX INPUT_DIR OUTPUT_DIR <<< "$LINE"
INPUT_DIR="$MAIN_PATH/$INPUT_DIR"
OUTPUT_DIR="$MAIN_PATH/$OUTPUT_DIR"
mkdir "$MAIN_PATH/sim_$INDEX"
cp alma.cycle9.3.cfg $MAIN_PATH/sim_$INDEX
cd "$MAIN_PATH/sim_$INDEX"
echo $CONDA_PREFIX
conda run -n casa6.5 python $MAIN_PATH/alma_simulator.py $INDEX $INPUT_DIR $OUTPUT_DIR
cd ..
rm -r "$MAIN_PATH/sim_$INDEX"
conda deactivate


