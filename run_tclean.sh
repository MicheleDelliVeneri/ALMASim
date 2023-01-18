#! /bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-4
MAIN_PATH="$PWD"

source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda init
conda activate casa6.5

START=$SLURM_ARRAY_TASK_ID
NUMLINES=2
STOP=$((SLURM_ARRAY_TASK_ID*NUMLINES))
START="$(($STOP - $(($NUMLINES - 1))))"
echo "START=$START"
echo "STOP=$STOP"

for (( N = $START; N <= $STOP; N++))
do
    LINE=$(sed -n "$N"p sims_param.csv)
    IFS=',' read INDEX INPUT_DIR OUTPUT_DIR ANTENNA_CONFIG<<< "$LINE"
    #INPUT_DIR="$MAIN_PATH/$INPUT_DIR"
    #OUTPUT_DIR="$MAIN_PATH/$OUTPUT_DIR"
    #mkdir "$MAIN_PATH/sim_$INDEX"
    #cp antenna_config/alma.cycle9.3.cfg $MAIN_PATH/sim_$INDEX
    cd "$MAIN_PATH/sim_$INDEX"
    echo $CONDA_PREFIX
    conda run -n casa6.5 python $MAIN_PATH/clean_simulator.py $INDEX $INPUT_DIR $OUTPUT_DIR $ANTENNA_CONFIG
    cd ..
    #rm -r "$MAIN_PATH/sim_$INDEX"
    conda deactivate
done