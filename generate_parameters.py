import numpy as np 
from astropy.io import fits
from astropy.wcs import WCS
from astropy import constants as c
import pandas as pd 
import sys 
import dask.dataframe as dd 
import time
import argparse
import os

parser =argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, 
        help='The directory in wich the simulated model cubes are stored;')
parser.add_argument("output_dir", type=str, 
         help='The directory in wich to store the simulated dirty cubes and corresponding skymodels;')
parser.add_argument("catalogue_name", type=str,
          help="The name of the .csv file in which the sources parameters were stored.")
args = parser.parse_args()
input_dir = args.model_dir
output_dir = args.output_dir
catalogue_name = args.catalogue_name

input_dataframe = pd.read_csv(os.path.join(input_dir, catalogue_name))
n = len(list(os.listdir(input_dir))) - 1



