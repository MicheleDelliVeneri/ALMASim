#! /bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
MAIN_PATH="$PWD"
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate casa6.5
INPUT_DIR="$MAIN_PATH/$1"
OUTPUT_DIR="$MAIN_PATH/$2"
CSV_NAME="$3"
N="$4"
conda run -n casa6.5 python $MAIN_PATH/generate_models.py $INPUT_DIR $OUTPUT_DIR $CSV_NAME $N
conda deactivate

