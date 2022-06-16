# PyALMASim

A python package to make realistic simulations of ALMA observations of galaxies and point sources.

## Generating ALMA simulations for the ML imaging purposes

Instructions:

1 Create a conda environment:

<pre><code>conda create --name casa6.5 python=3.8 </code></pre>

2 Activate it:

<pre><code>conda activate casa6.5</code></pre>

3 Move to the folder where you want to store the results
4 Clone the GitHub repository:

<pre><code>git clone https://github.com/MicheleDelliVeneri/PyALMASim.git </code></pre>

5 Make sure that the required libraries are installed, we are supposing to be on a centos system:

<pre><code>sudo yum install ImageMagick*</code></pre>
<pre><code>sudo yum install xorg-x11-server-Xvfb</code></pre>
<pre><code>sudo yum install compat-libgfortran-48</code></pre>
<pre><code>sudo yum install libnsl</code></pre>
<pre><code>sudo yum install openmpi-devel</code></pre>
<pre><code>sudo yum install mpich-devel</code></pre>
<pre><code>sudo yum install parallel</code></pre>

6 Install the required python libraries

<pre><code>pip install -r requirements.txt</code></pre>

7 Generate the sky model cubes:
modify the create_models.sh script with the number of cpus-per-task you want to use. This is the number of cubes that will be created in parallel. This could be set to the maximum number of cores on a given node.

<pre><code>sbatch create_models.sh models sims params.csv 10000 </code></pre>

where the first parameter <b>models</b> is the name of the directory in which to store the <b>sky models</b> cubes, the second <b>sims</b> is the name of the directory in which to store the simulations, the third <b>params.csv</b> is the name of the .csv file which holds the sources parameters and the fourth <b>n</b> is the number of cubes to generate
8 Generate the ALMA simulations:
In order to generate the simulations, we are going to run the <b>run_simulations.sh</b> script in parallel with sbatch.
First, after running the create_models script, you shoudl first see the models directory0 populated with sky models .fits files, but also a <b>sim_param.csv</b> file in the root folder.
This file is used by the <b>run_simulations.sh</b> bash script to generate the simulations in parallel through sbatch. To do so first modify the --aray field with the number of sky-models you have previously generated and then run it with the following command:

<pre><code>sbatch run_simulations.sh
 </code></pre>

The script assumes that your conda environment is called conda6.5, otherwise modify its name in the script.
9 Now that the simulations are concluded, we neet to update the parameters in the <b>params.csv</b> file with the fluxes and continuum values. To do so run the following command:

<pre><code>python generate_parameters.py models sims </code></pre>

10 You are all ready to train and test your models.
