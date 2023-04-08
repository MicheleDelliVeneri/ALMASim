#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
MAIN_PATH="$PWD"

# Get the options
Help()
{
   # Display Help
   echo "Hello welcome to the ALMASim help function ."
   echo
   echo "Syntax: create_models.sh -d models -p plots [-o options]"
   echo "options:"
   echo "d     The directory in which the mock data is stored"
   echo "p     The directory in which to store the plots."
   echo "c     The name of the .csv file in which to store the simulated source parameters."
   echo "m     The mode in which to run the simulation. Options are: gauss, extended."
   echo "n     The number of models to simulate."
   echo "a     The antenna configuration to use."
   echo "s     The spatial resolution of the simulation."
   echo "i     The integration time of the simulation."
   echo "C     The coordinates of source in the sky."
   echo "b     The bandwidth of the simulation in MHz, default 1000 Mhz."
   echo "B     The ALMA band to simulate. Options are the following bands: 1, 3, 4, 5, 6, 7, 8, 9, 10."
   echo "f     The frequency resolution of the simulation."
   echo "v     The velocity resolution of the simulation."
   echo "t     The path to the TNGbase directory."
   echo "S     The snapshot number of the TNGbase simulation."
   echo "I     The subhalo ID of the TNGbase simulation."
   echo "P     The number of pixels in the simulation."
   echo "N     The number of channels in the simulation."
   echo "R     The right ascension of the source."
   echo "D     The declination of the source."
   echo "T     The distance of the source."
   echo "l     The noise level of the simulation."
   echo "e     Whether to sample the parameters of the simulation or use input parameters / default ones."
   echo "w     Flags controlling which parameters to sample. See -V for more details."
   echo "k     Whether to save the plots of the simulation."
   echo "h     Print this Help."
   echo "V     Detailed parameters description and default values     "
   echo
   echo "Never forget to have fun! otherwise go get a job in a cozy office in industry."
   echo "For any bug please open an issue on the github page."

   echo 
}

ShowParams()
{
    # Display Help
    conda run -n casa6.5 python $MAIN_PATH/generate_models.py -h
}

MASTER_DIR="ALMASim";
PLOT_DIR="plots";
PARAM_NAME="params.csv";
MODE="gauss";
N="10";
ANTENNA_CONFIG="antenna_config/alma.cycle9.3.1.cfg"
SPATIAL_RESOLUTION="0.1"
INTEGRATION_TIME="2400"
COORDINATES="J2000 03h59m59.96s -34d59m59.50s"
BANDWIDTH="1000"
ALMA_BAND="6"
FREQUENCY_RESOLUTION="10"
VELOCITY_RESOLUTION="10"
TNGBASE_PATH="TNG100-1/output/"
TNG_SNAP="99"
TNG_SUBHALO_ID="385350"
N_PIXELS="256"
N_CHANNELS="128"
RA="0.0"
DEC="0.0"
DISTANCE="0.0"
NOISE_LEVEL="0.3"
SAMPLE_PARAMS="False"
SAVE_PLOTS="False"
SAMPLE_PARAMS_FLAGS="e"

while getopts ":h;:V;d:;p:;c:;m:;n:;a:;s:;i:;C:;b:;B:;f:;v:;t:;S:;I:;P:;N:;R:;D:;T:;l:;e:;k:;w:" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      d) # Enter a input directory
         MASTER_DIR="$OPTARG";;
      p) # Enter a plot directory
          PLOT_DIR="$OPTARG";;
      c) # Enter a csv name
          PARAM_NAME="$OPTARG";;
      m) # Enter a mode
          MODE="$OPTARG";;
      n) # Enter a number of models
          N="$OPTARG";;
      a) # Enter a antenna config
          ANTENNA_CONFIG="$OPTARG";;
      s) # Enter a spatial resolution
          SPATIAL_RESOLUTION="$OPTARG";;
      i) # Enter a integration time
          INTEGRATION_TIME="$OPTARG";;
      C) # Enter coordinates
          COORDINATES="$OPTARG";;
      b) # Enter a bandwidth
          BANDWIDTH="$OPTARG";;
      B) # Enter an ALMA band
          ALMA_BAND="$OPTARG";;
      f) # Enter a frequency resolution
          FREQUENCY_RESOLUTION="$OPTARG";;
      v) # Enter a velocity resolution
          VELOCITY_RESOLUTION="$OPTARG";;
      t) # Enter a TNGbase path
          TNGBASE_PATH="$OPTARG";;
      S) # Enter a TNG snapshot
          TNG_SNAP="$OPTARG";;
      I) # Enter a TNG subhalo ID
          TNG_SUBHALO_ID="$OPTARG";;
      P) # Enter a number of pixels
          N_PIXELS="$OPTARG";;
      N) # Enter a number of channels
          N_CHANNELS="$OPTARG";;
      R) # Enter a RA
          RA="$OPTARG";;
      D) # Enter a DEC
          DEC="$OPTARG";;
      T) # Enter a distance
          DISTANCE="$OPTARG";;
      l) # Enter a noise level
          NOISE_LEVEL="$OPTARG";;
      e) # Enter a sample params
          SAMPLE_PARAMS="$OPTARG";;
      k) # Enter a save plots
          SAVE_PLOTS="$OPTARG";;
      w) # Enter sample params flags
          SAMPLE_PARAMS_FLAGS="$OPTARG";;
      \?) # incorrect option
          echo "Error: Invalid option"
          exit;;
      V) # Default values
          ShowParams
          exit;;
      
   esac
done


conda run -n casa6.5 python $MAIN_PATH/generate_models.py  --coordinates "$COORDINATES" --data_dir $MASTER_DIR --plot_dir $PLOT_DIR --csv_name $PARAM_NAME --mode $MODE --n_simulations $N --antenna_config $ANTENNA_CONFIG --spatial_resolution $SPATIAL_RESOLUTION --integration_time $INTEGRATION_TIME  --bandwidth $BANDWIDTH --band $ALMA_BAND --frequency_resolution $FREQUENCY_RESOLUTION --velocity_resolution $VELOCITY_RESOLUTION --TNGBasePath $TNGBASE_PATH --TNGSnap $TNG_SNAP --TNGSubhaloID $TNG_SUBHALO_ID --n_px $N_PIXELS --n_chan $N_CHANNELS --ra $RA --dec $DEC --distance $DISTANCE --noise_level $NOISE_LEVEL --sample_params $SAMPLE_PARAMS --sample_selection $SAMPLE_PARAMS_FLAGS --save_plots $SAVE_PLOTS  
echo "Finished generating models, please use the create_simulation.sh script to create the simulations."
