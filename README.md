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
