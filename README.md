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
1. -d : (data directory) the directory in which the simulated models cubes are temporarily stored;
2. -o : (output directory) the directory in which the simulated sky model cubes, dirty cubes, and measurement sets will be stored stored;
3. -p : (plot directory) the directory in which the plots of the simulated sky model cubes and dirty cubes will be stored;
4. -c : (csv name) the name of the .csv file containing the source positions and morphological properties;
5. -m : (mode) the mode of the simulation, it can be either "gauss" or "extended";
6. -n : (number of sources) the number of output simulated cubes;
7. -a : (Antenna Configuration) the antenna configuration file path, default value is antenna_config/alma.cycle9.3.1.cfg;
8. -s : (Spatial Resolution) the spatial resolution of the simulated observations in arcseconds, default value is 0.1;
9. -i : (Total Integration Time): the total integration time of the simulated observations, default value is 2400 seconds;
10. -C : (Coordinates) the coordinates of the simulated observations, default value is J2000 03h59m59.96s -34d59m59.50s;
11. -B : (Band) the ALMA observation band, which determines the central frequency of observation and thus the beam size, default value is 6;
12. -b : (Bandwidth) the bandwidth of the simulated observations in MHz, default value is 10 MHz;
13. -f ; (Frequency Resolution) the frequency resolution of the simulated observations in MHz, default value is 10 MHz; 
14. -v : (Velocity Resolution) the velocity resolution of the simulated observations in km/s, default value is 10 km/s;
15. -t : (TNG Base Path) the path to the TNG100-1 snapshot, default value is  TNG100-1/output;
16. -S : (TNG Snapshot) the TNG100-1 snapshot number, default value is 99;
17. -I : (TNG Subhalo ID) the TNG100-1 subhalo ID, default value is 385350, this parameter can be set as a list if multiple subhalos are to be used as extended models for simulations;
18. -P : (Number of Pixels) the number of pixels of the simualated observations, default value is 256. The final output cubes will have a size roughly equal to 1.5 times the number of pixels. This is done in order to ensure that the primary beam fits the spatial dimensions of the cube;
19. -N : (Number of Channels) the number of channels in the frequency dimension of the cube, default value is 128;
20. -R : (Right Ascension) the right ascension of the simulated observation in degrees, default value is 0;
21. -D : (Declination) the declination of the simulated observation in degrees, default value is 0;
22. -T : (Distance) the distance of the simulated observation in Mpc, default value is 30;
23. -l : (Noise Level) the noise level of the simulated observations as a fraction of the primary peak max flux, default value is 0.3;
24. -e : (Sample Parameter Flag): if set to True, some of the parameters can be sampled from ALMA real observations. Must be combined with the sample selection flags -w;
25. -w : (Sample Selection Flag) : flags that determine which parameters to sample from real observations and which to set from the user defined or default values. The flags must be set as a continuous string, as an example [ -w acNbf] will sample for each observation the antenna, the coordinates, the number of channels, the ALMA band and the frequency resolution. The flags are:
    - a sample antenna configuration;
    - r sample spatial resolution;
    -t sample total integration time;
    -c 


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
