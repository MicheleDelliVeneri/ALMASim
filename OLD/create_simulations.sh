#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
MAIN_PATH="ALMASim"
MASTER_PATH="$PWD"
END=10
PARAM_PATH="$MAIN_PATH/sims_param.csv"

Help()
{
    echo 'Hello welcome to the ALMASim Create Simulations help function.'
    echo 
    echo 'Syntax: create_simulations.sh -p pathto/sims_param.csv'
    echo 'options:'
    echo 'd     The directory in which the mock data is stored'
    echo 'h     Print this Help.'
}
while getopts ":h;d:" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        d) # Path to the .csv file containing the parameters of the simulations
            MAIN_PATH=$OPTARG
            ;;
        \?) # incorrect option
            echo "Error: Invalid option"
            exit;;
    esac
done
PARAM_PATH="$MAIN_PATH/sims_param.csv"
echo $PARAM_PATH
for i in $(seq 1 $END);
do  
    LINE=$(sed -n "$i"p $PARAM_PATH)
    IFS=',' read INDEX INPUT_DIR OUTPUT_DIR ANTENNA_CONFIG COORDINATES SPATIAL_RESOLUTION CENTRAL_FREQUENCY FREQUENCY_RESOLUTION INTEGRATION_TIME MAP_SIZE N_PX VELOCITY_RESOLUTION RA DEC DISTANCE NOISE_LEVEL X_ROT Y_ROT N_CHANNELS SUBHALO_ID SNAPSHOT_ID<<< "$LINE"
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
    echo "Setting a Velocity Resolution of $VELOCITY_RESOLUTION"
    echo "Setting RA to $RA"
    echo "Setting DEC to $DEC"
    echo "Setting Distance to $DISTANCE"
    echo "Setting Noise Level to $NOISE_LEVEL"
    echo "Setting X Rotation to $X_ROT"
    echo "Setting Y Rotation to $Y_ROT"
    echo "Setting Number of Channels to $N_CHANNELS"
    echo "Setting Subhalo ID to $SUBHALO_ID"
    echo "Setting Snapshot ID to $SNAPSHOT_ID"

    mkdir "$MAIN_PATH/sim_$INDEX"
    cd "$MAIN_PATH/sim_$INDEX"
    conda run -n casa6.5 python "$MASTER_PATH/alma_simulator.py" $INDEX "$INPUT_DIR" "$OUTPUT_DIR" "$ANTENNA_CONFIG" "$COORDINATES" "$SPATIAL_RESOLUTION" "$CENTRAL_FREQUENCY" "$FREQUENCY_RESOLUTION" "$INTEGRATION_TIME" "$MAP_SIZE" "$N_PX" 
    cd ..
    rm -r "$MAIN_PATH/sim_$INDEX"
done