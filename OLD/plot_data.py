import numpy as np
import os 
import sys
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted
from astropy.io import fits
from astropy import units as u
import argparse
from tqdm import tqdm

def load_fits(inFile):
    hdu_list = fits.open(inFile)
    data = hdu_list[0].data
    hdu_list.close()
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help='The directory containing the mock data;', default='ALMASim')
parser.add_argument("--plot_dir", type=str, help='The directory to save the plots;', default="plots")
parser.add_argument("--n_samples", type=int, help='Number of samples to plot', default=10)



args = parser.parse_args()
data_dir = args.data_dir + '/sims'
n_samples = args.n_samples
plot_dir = os.path.join(args.data_dir, args.plot_dir)

if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

clean_list = np.array(natsorted(list([file for file in os.listdir(data_dir) if 'clean' in file])))
dirty_list = np.array(natsorted(list([file for file in os.listdir(data_dir) if 'dirty' in file])))
idxs = np.random.randint(0, len(clean_list), n_samples)

for idx in tqdm(idxs, desc='Plotting samples', total=len(idxs)):
    clean = load_fits(os.path.join(data_dir, clean_list[idx]))
    dirty = load_fits(os.path.join(data_dir, dirty_list[idx]))
    print(clean.shape, dirty.shape)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(np.sum(clean[0, :, :, :], axis=0), origin='lower')
    plt.colorbar()
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title('Clean')
    plt.subplot(122)
    plt.imshow(np.sum(dirty[0, :, :, :], axis=0), origin='lower')
    plt.colorbar()
    plt.title('Dirty')
    plt.savefig(os.path.join(plot_dir, 'sample_{}.png'.format(idx)))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(np.sum(clean[0, :, :, :], axis=(1, 2)))
    plt.xlabel('Frequency (channels)')
    plt.ylabel('Flux (Jy)')
    plt.title('Clean Spectrum')
    plt.subplot(122)
    plt.plot(np.sum(dirty[0, :, :, :], axis=(1, 2)))
    plt.title('Dirty Spectrum')
    plt.savefig(os.path.join(plot_dir, 'sample_spectrum_{}.png'.format(idx)))
    plt.close()