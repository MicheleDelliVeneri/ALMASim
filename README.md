# ALMASim
[![arXiv](https://img.shields.io/badge/arXiv-2211.11462-00ff00.svg)](https://arxiv.org/abs/2211.11462) 

![](images/Icon.png)

A python package to make realistic simulations of ALMA observations of galaxies and point sources. 
The project, at its current status, is able to generate both point like and extended line emission sources, and to simulate the ALMA observations of these sources.
This can be done by both using user defined simulation and observational parameters or by sampling from known ALMA observations of galaxies and point sources. The package generates 
simulations using the CASA6.5 framework, the TNG100-1 simulation of the IllustrisTNG project, the ALMA archive and the Martini HI simulation package and it procedes as follows:
- Sky Models are Generated;
- Dirty Cubes counterparts are generated based on the sky models and the chosen observational parameters;
- A a .csv file containing all source positions and morphological properties is generated.

The main scope of this repository is to let scientistis generate their own simple ALMA simulions on which to train and test their deconvolution and source detection and characterization models.

## Pre-requisites
You should install the conda package manager available at https://docs.conda.io/en/latest/miniconda.html and the following libraries. Those provided are for CentOS, for other systems please refer to the official documentation:
<pre><code>sudo yum install ImageMagick*</code></pre>
<pre><code>sudo yum install xorg-x11-server-Xvfb</code></pre>
<pre><code>sudo yum install compat-libgfortran-48</code></pre>
<pre><code>sudo yum install libnsl</code></pre>
<pre><code>sudo yum install openmpi-devel</code></pre>
<pre><code>sudo yum install mpich-devel</code></pre>
<pre><code>sudo yum install parallel</code></pre>

## Installation
1 Clone the GitHub repository and move into it:
<pre><code>git clone https://github.com/MicheleDelliVeneri/ALMASim.git</code></pre>
<pre><code>cd ALMASim</code></pre>

2 Create a conda environment from the provided requirements and activate it:
<pre><code>conda create --name casa6.5 --file requirements.txt </code></pre>
<pre><code>conda activate casa6.5 </code></pre>

3 Install the Hdecompose package:
<pre><code> git clone https://github.com/kyleaoman/Hdecompose.git </code></pre>
<pre><code> cd Hdecompose </code></pre>
<pre><code> python setup.py install </code></pre>

4 Install the MARTINI package:
<pre><code> git clone https://github.com/kyleaoman/martini.git</code></pre>

5 If you are interested in simulating Extended sources, you need to download the 99 snapshot of the TNG100-1 simulation from the IllustrisTNG project.
<pre><code> wget https://www.illustris-project.org/data/TNG100-1/output/snapdir_099/snap_099.0.hdf5 </code></pre>
also download 
## Usage
Both the model creation and the simulation creation are performed through the exectution bash scripts which can be executed both through bash or though the slurm workload manager with sbatch. All simulation
parameters are set in the model simuation, so that is the tricky part, the rest is just about running another bash script. The model creation is performed by the script create_model.sh, which can be executed as follows:
<pre><code>bash create_model.sh -option value</code></pre>
The options are:
* -d (data directory) the directory in which the simulated models cubes are temporarily stored;
* -o (output directory) the directory in which the simulated sky model cubes, dirty cubes, and measurement sets will be stored stored;
* -p (plot directory) the directory in which the plots of the simulated sky model cubes and dirty cubes will be stored;
* -c (csv name) the name of the .csv file containing the source positions and morphological properties;
* -m (mode) the mode of the simulation, it can be either "gauss" or "extended";
* -n (number of sources) the number of output simulated cubes;
* -a (Antenna Configuration) the antenna configuration file path, default value is antenna_config/alma.cycle9.3.1.cfg;
* -s (Spatial Resolution) the spatial resolution of the simulated observations in arcseconds, default value is 0.1;
* -i (Total Integration Time): the total integration time of the simulated observations, default value is 2400 seconds;
* -C (Coordinates) the coordinates of the simulated observations, default value is J2000 03h59m59.96s -34d59m59.50s;
* -B (Band) the ALMA observation band, which determines the central frequency of observation and thus the beam size, default value is 6;
* -b (Bandwidth) the bandwidth of the simulated observations in MHz, default value is 10 MHz;
* -f (Frequency Resolution) the frequency resolution of the simulated observations in MHz, default value is 10 MHz; 
* -v (Velocity Resolution) the velocity resolution of the simulated observations in km/s, default value is 10 km/s;
* -t (TNG Base Path) the path to the TNG100-1 snapshot, default value is  TNG100-1/output;
* -S (TNG Snapshot) the TNG100-1 snapshot number, default value is 99;
* -I (TNG Subhalo ID) the TNG100-1 subhalo ID, default value is 385350, this parameter can be set as a list if multiple subhalos are to be used as extended models for simulations;
* -P (Number of Pixels) the number of pixels of the simualated observations, default value is 256. The final output cubes will have a size roughly equal to 1.5 times the number of pixels. This is done in order to ensure that the primary beam fits the spatial dimensions of the cube;
* -N (Number of Channels) the number of channels in the frequency dimension of the cube, default value is 128;
* -R (Right Ascension) the right ascension of the simulated observation in degrees, default value is 0;
* -D (Declination) the declination of the simulated observation in degrees, default value is 0;
* -T (Distance) the distance of the simulated observation in Mpc, default value is 30;
* -l (Noise Level) the noise level of the simulated observations as a fraction of the primary peak max flux, default value is 0.3;
* -e (Sample Parameter Flag): if set to True, some of the parameters can be sampled from ALMA real observations. Must be combined with the sample selection flags -w;
* -w (Sample Selection Flag) : flags that determine which parameters to sample from real observations and which to set from the user defined or default values. The flags must be set as a continuous string, as an example [ -w acNbf] will sample for each observation the antenna, the coordinates, the number of channels, the ALMA band and the frequency resolution. The flags are:
    - -a sample antenna configuration;
    - -r sample spatial resolution;
    - -t sample total integration time;
    - -c sample coordinates;
    - -N sample number of channels;
    - -b sample ALMA band;
    - -B sample bandwidth;
    - -f sample frequency resolution;
    - -v sample velocity resolution;
    - -s sample TNG Snapshot;
    - -i sample TNG Subhalo ID;
    - -C sample RA and DEC;
    - -D sample Distance;
    - -N sample noise level;
    - -p sample the number of pixels;
    - -e sample all parameters;

After the script has been executed you will see that the data directory contains the simulated models, plus two .csv. 
The first one named sims_params.csv contains the parameters that must be fed to the create_simulations.sh script, the second one named sims_info.csv contains information about the sims_params.csv file such as the name of the columns and the units of the parameters.

Generate the dirsty models by modifying the create_simulations.sh script to reflect your hardware and then by running it through slurm:
<pre><code>sbatch create_simulations.sh</code></pre>

The create_simulations.sh script will generate the dirty cubes and the measurement sets, which will be stored in the output directory. The plots of the dirty cubes and the sky model cubes will be stored in the plot directory. To clean up the temporary files, run the clean.sh script:
You are set, enjoy your simulations!

 ## Work in progress
 - Introduce multi-line emissions / absorptions;
 - Introduce source classes;


### Cite us

Michele Delli Veneri, Łukasz Tychoniec, Fabrizia Guglielmetti, Giuseppe Longo, Eric Villard, 3D Detection and Characterisation of ALMA Sources through Deep Learning, Monthly Notices of the Royal Astronomical Society, 2022;, stac3314, https://doi.org/10.1093/mnras/stac3314

@article{10.1093/mnras/stac3314,
    author = {Delli Veneri, Michele and Tychoniec, Łukasz and Guglielmetti, Fabrizia and Longo, Giuseppe and Villard, Eric},
    title = "{3D Detection and Characterisation of ALMA Sources through Deep Learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2022},
    month = {11},
    issn = {0035-8711},
    doi = {10.1093/mnras/stac3314},
    url = {https://doi.org/10.1093/mnras/stac3314},
    note = {stac3314},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stac3314/47014718/stac3314.pdf},
}
