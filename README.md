# ALMASim
[![arXiv](https://img.shields.io/badge/arXiv-2211.11462-00ff00.svg)](https://arxiv.org/abs/2211.11462) 

![](images/Icon.png)

A python package to make realistic simulations of ALMA observations of galaxies and point sources. 
For now, only simple point-like sources are generated, but soon more complex Galaxy models will be added. The project, at its current status, is able to:
- Create Sky Model Cubes of randomly scattered point-like sources;
- Generate corresponding calibrated Dirty Cubes;
- Generate tCLEAN cleaned counterparts to the Dirty Cubes;
- Generate a .csv file containing all source positions and morphological properties.

The main scope of this repository is to let scientistis generate their own simple ALMA simulions on which to train and test their models.

## Generating ALMA simulations for the ML imaging purposes

Instructions:

1 Create a conda environment and activate it:

<pre><code>conda create --name casa6.5 python=3.8 </code></pre>

<pre><code>conda activate casa6.5 </code></pre>


2 Clone the GitHub repository:
<pre><code>git clone https://github.com/MicheleDelliVeneri/ALMASim.git</code></pre>

3 Move to the repository:
<pre><code>cd ALMASim</code></pre>

4 Make sure that the required libraries are installed, we are supposing to be on a centos system:

<pre><code>sudo yum install ImageMagick*</code></pre>
<pre><code>sudo yum install xorg-x11-server-Xvfb</code></pre>
<pre><code>sudo yum install compat-libgfortran-48</code></pre>
<pre><code>sudo yum install libnsl</code></pre>
<pre><code>sudo yum install openmpi-devel</code></pre>
<pre><code>sudo yum install mpich-devel</code></pre>
<pre><code>sudo yum install parallel</code></pre>

5 Install the required python libraries

<pre><code>pip install -r requirements.txt</code></pre>

activate the base casa environment

<pre><code> casa activate base </code></pre>

<pre><code> git clone https://github.com/kyleaoman/Hdecompose.git </code></pre>
<pre><code> cd Hdecompose </code></pre>
<pre><code> python setup.py install </code></pre>

<pre><code> git clone 


6 Generate the sky model cubes:
modify the create_models.sh script with the number of cpus-per-task you want to use, and the number of tasks you want to run in parallel.
<pre><code>sbatch create_models.sh inputs_dir outputs_dir params.csv n configuration_file</code></pre>

where the first parameter <b>models</b> is the name of the directory in which to store the <b>sky models</b> cubes, the second <b>sims</b> is the name of the directory in which to store the simulations, the third <b>params.csv</b> is the name of the .csv file which holds the sources parameters and the fourth <b>n</b> is the number of cubes to generate
8 Generate the ALMA simulations, and configuration_file is teh path of one of the .cfg files stored in the antenna_config folder. For example this creates 10,000 sky model cubes in the models folder using the 9.3.1 configuration file. 

<pre><code>sbatch create_models.sh models sims params.csv 10000 antenna_config/alma.cycle9.3.1.cfg</code></pre>

7 Generate the dirty cubes: 
In order to generate the simulations, we are going to run the <b>run_simulations.sh</b> script in parallel with sbatch.
To do so first modify the --array field with the number of parallel tasks you want to use and modify NUMLINES so that NUMLINES * array equals the number of .fits file in the models folder, and then run it with the following command:

<pre><code>sbatch run_simulations.sh
 </code></pre>

8 Generate tclean cleaned cubes:
also modify the --array field to be consistent with previous value
<pre><code>sbatch run_tclean.sh
 </code></pre>
 this function will take the dirty cubes generated and run them through tclean, it will also write in each sim_i folder a .txt called running_params.txt containin the time of execution. 

The script assumes that your conda environment is called conda6.5, otherwise, modify its name in the script at line 9.

9 Update the parameters in the <b>params.csv</b> file with the fluxes and continuum values:
To do so, run the following command:
<pre><code>conda activate casa6.5</code></pre>
<pre><code>python generate_gaussian_params.py models sims params.csv 0.</code></pre>

or if you want to run it via slurm: ß
<pre><code>conda activate casa6.5</code></pre>
<pre><code>srun -n 1 -c 32 python generate_params.py models sims params.csv 0.15</code></pre>
where 0. is an examplary value for the noise rms. In case you want to add additional white noise to the simulations, increase this value. 

10 If you plan to use a Machine Learning model and you need to split the data into train, validation and test sets, you can use the split_data.py script. 
<pre><code>python split_data.py data_folder tclean_flag train_size</code></pre>
where <b>data_folder</b> is the path to the directory containing the output_dir containing the fits (for example sims), <b>tclean_flag</b> is a boolean flag indicating if tclean cleaned fits were also created, and train_size is the training size as a float between 0 and 1. Validation is created from the 25% of the remaining training set. 
If you followed the default exaple, simply run:
<pre><code>python split_data.py "" True 0.8</code></pre>


11. If you do not want to store them for later use and you want to save space, now you can safely delete the models and sim_* and sims folders, and all the .out and .log files or use the provided cleanup.sh script which takes as input the input and output dir defined at the beginning.

<pre><code>sh cleanup.sh models sims</code></pre>



You are set, enjoy your simulations!

 ## Work in progress
 - Introduce Galaxy dynamic and complex spectral profiles;
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
