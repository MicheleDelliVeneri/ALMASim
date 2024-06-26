![Alt text](pictures/ALMASimBanner.jpeg)

## Overview

ALMASim is a package to generate mock observations of HI line galaxies as observed by the Atacama Large Millimetre/Submillimetre Array (ALMA). ALMASim primary goal is to allow users to generate simulated datasets on which to test deconvolution and source detection models. ALMASim is intended to leverage MPI parallel computing on modern HPC clusters to generate thousands of ALMA data cubes, but can also work on laptopts.
ALMA database or a prepared catalogue is queried to sample observational 
metadata such as band, bandwidth, integration time, antenna configurations and so on. 
ALMASim employs the MARTINI Package (https://github.com/kyleaoman/martini), and the Illustris Python Package (https://github.com/illustristng/illustris_python) to be able to generate observations of extendend sources in all possible ALMA configurations.
For every observed target ALMASim generates:

## Citing ALMASim

   
If you use ALMASim in your research, please cite the following paper:


    @ARTICLE{10.1093/mnras/stac3314,
    author = {Delli Veneri, Michele and Tychoniec, Łukasz and Guglielmetti, Fabrizia and Longo, Giuseppe and Villard, Eric},
    title = "{3D Detection and Characterisation of ALMA Sources through Deep Learning}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    year = {2022},
    month = {11},
    issn = {0035-8711}, 
    doi = {10.1093/mnras/stac3314},
    url = {https://doi.org/10.1093/mnras/stac3314},
    note = {stac3314},
    eprint = {https://academic.oup.com/mnras/advance-article-pdf/doi/10.1093/mnras/stac3314/47014718/stac3314.pdf}
    }

ALMASim entry: https://doi.org/10.1093/mnras/stac3314

## Installation Notes


ALMASim works with ``python3`` (version ``3.12``), and does not support ``python2``.
First create a virtual environment with ``python3`` and activate it. Then install the required packages with ``pip``:

- Create the Python Environment:```python3.12 -m venv astro-env```
- Activate it: ```source astro-env/bin/activate``` (in case of your shell is Bash, otherwise check the other activations scripts within the bin folders)
- Clone the ALMASim Repository: ```git clone https://github.com/MicheleDelliVeneri/ALMASim.git```
- Install packages from the requirements file: ```pip install -r ALMASim/requirements.txt```
- Clone Illustris TNG Folder: ```git clone https://github.com/illustristng/illustris_python.git```
- Move to the Folder: ```cd illustris_python```
- Install the packages: ```pip install .```
- Login into Kaggle and go to: ```https://www.kaggle.com/settings```
- Click on ```create new token```, this will produce a kaggle.json file which must be saved in your home folder


## Getting started

To run the simulation, just navigate to the ALMASim folder and execute:

``` python main.py ```

## Notes

Cube size will dictate simulation speed and RAM usage. To gauge what you can affor to run, we advice to start with a single simulation of a 256 x 256 x 256 cube.
1.85 GHZ is the descriminator between line and continuum spectral window. 