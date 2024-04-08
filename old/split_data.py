import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from natsort import natsorted
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str,
                    help='The directory containing the sims subdirectory;', default='')
parser.add_argument('tclean_flag', type=bool, 
                    help='tCLEAN flag, if set to True, it expects to find tclea cleaned cubes within the sims folder', 
                    default=True)
parser.add_argument('train_size', type=float, help='Training size, float between 0 and 1.', default=0.8)
args = parser.parse_args()

data_dir = args.data_dir
tclean_flag = args.tclean_flag
train_size = args.train_size
input_dir = os.path.join(data_dir, 'sims')
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
valid_dir = os.path.join(data_dir, "Validation")
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)
if not os.path.exists(valid_dir):
    os.mkdir(valid_dir)
dlist = np.array(natsorted(list([file for file in os.listdir(input_dir) if 'dirty' in file])))
clist = np.array(natsorted(list([file for file in os.listdir(input_dir) if 'clean' in file])))
if tclean_flag is True:
    tlist = np.array(natsorted(list([file for file in os.listdir(input_dir) if 'tcleaned' in file])))
params = pd.read_csv(os.path.join(input_dir, "params.csv"))
indexes = np.arange(dlist.shape[0])
train_idxs, test_idxs = train_test_split(indexes, test_size=1 - train_size, random_state=42)
train_idxs, valid_idxs = train_test_split(train_idxs, test_size=0.25, random_state=42)
train_idxs = np.array(natsorted(train_idxs))
test_idxs = np.array(natsorted(test_idxs))
valid_idxs = np.array(natsorted(valid_idxs))
train_params = params[params['ID'].isin(train_idxs)]
valid_params = params[params['ID'].isin(valid_idxs)]
test_params = params[params['ID'].isin(test_idxs)]
train_params.to_csv(os.path.join(train_dir, 'train_params.csv'), index=False)
test_params.to_csv(os.path.join(test_dir, 'test_params.csv'), index=False)
valid_params.to_csv(os.path.join(valid_dir, 'valid_params.csv'), index=False)
print('Splitting fits cubes in Train, Test, and Validation')
for idx in tqdm(indexes):
    if idx in train_idxs:
        os.system("cp {} {}".format(os.path.join(input_dir, dlist[idx]), 
                    os.path.join(train_dir, dlist[idx])))
        os.system("cp {} {}".format(os.path.join(input_dir, clist[idx]), 
                    os.path.join(train_dir, clist[idx])))
        if tclean_flag is True:
            os.system("cp {} {}".format(os.path.join(input_dir, tlist[idx]), 
                    os.path.join(train_dir, tlist[idx])))
    elif idx in valid_idxs:
        os.system("cp {} {}".format(os.path.join(input_dir, dlist[idx]), 
                    os.path.join(valid_dir, dlist[idx])))
        os.system("cp {} {}".format(os.path.join(input_dir, clist[idx]), 
                    os.path.join(valid_dir, clist[idx])))
        if tclean_flag is True:
            os.system("cp {} {}".format(os.path.join(input_dir, tlist[idx]), 
                    os.path.join(valid_dir, tlist[idx])))
    
    else:
        os.system("cp {} {}".format(os.path.join(input_dir, dlist[idx]), 
                    os.path.join(test_dir, dlist[idx])))
        os.system("cp {} {}".format(os.path.join(input_dir, clist[idx]), 
                    os.path.join(test_dir, clist[idx])))
        if tclean_flag is True:
            os.system("cp {} {}".format(os.path.join(input_dir, tlist[idx]), 
                    os.path.join(test_dir, tlist[idx])))   