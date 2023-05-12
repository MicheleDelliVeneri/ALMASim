import dask
from dask.distributed import Client

import sys
import os
 
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import simulator as sm
import numpy as np
import pandas as pd
import os
import argparse
from random import choices

def test_imports():
    print('All imports completed')


test_imports()