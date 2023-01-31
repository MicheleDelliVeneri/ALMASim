#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
MAIN_PATH="$PWD"
END=10
for i in $(seq 1 $END);
do  
    echo $i
    LINE=$(sed -n "$i"p sims_param.csv)
    IFS=',' read INDEX INPUT_DIR OUTPUT_DIR ANTENNA_CONFIG COORDINATES SPATIAL_RESOLUTION CENTRAL_FREQUENCY FREQUENCY_RESOLUTION INTEGRATION_TIME MAP_SIZE N_PX<<< "$LINE"
    echo "Sampling Sky model: $INDEX"
    echo "From Directory: $INPUT_DIR"
    echo "Saving Sky Model and Dirty Cube into $OUTPUT_DIR"
    echo "Using the Antenna Configuration: $ANTENNA_CONFIG"
    echo "Object Simulated at coordinates: $COORDINATES"
    echo "Setting Spatial Resolution to: $SPATIAL_RESOLUTION"
    echo "Setting Central Frequency to: $CENTRAL_FREQUENCY"
    echo "Setting Frequency Resolution to: $FREQUENCY_RESOLUTION"
    echo "Setting an total integration time of: $INTEGRATION_TIME"
    echo "Setting a Map Size of $MAP_SIZE"
    echo "Final cube spatial dimensions are [$N_PX, $N_PX]"

    mkdir "$MAIN_PATH/sim_$INDEX"
    cd "$MAIN_PATH/sim_$INDEX"
    conda run -n casa6.5 python $MAIN_PATH/alma_simulator.py $INDEX "$MAIN_PATH/$INPUT_DIR" "$MAIN_PATH/$OUTPUT_DIR" "$MAIN_PATH/$ANTENNA_CONFIG" "$COORDINATES" $SPATIAL_RESOLUTION $CENTRAL_FREQUENCY $FREQUENCY_RESOLUTION $INTEGRATION_TIME $MAP_SIZE $N_PX 
    cd ..
done