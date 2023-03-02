import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import units as U
from astropy import constants as const
from astropy.coordinates import SkyCoord
import astropy.units as U
import astropy 
from astropy.io import fits
import seaborn as sns
import os
import argparse



class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='Genarating observation metadata for simualtions',
                                 formatter_class=SmartFormatter)
parser.add_argument('--d', '--data_path', type=str, default='all_alma_data.csv', help='Path to the data file')
parser.add_argument('--o', '--output_path', type=str, default='obs_configurations.csv', help='Path to the output file')
args = parser.parse_args()
datapath = str(args.d)
outpath = str(args.o)
data = pd.read_csv(datapath, low_memory=False)
observation_metadata = data[['s_resolution', 't_exptime', 'em_min', 't_resolution',
                        'em_max', 'em_resolution', 'spatial_resolution', 'bandwidth', 
                        'frequency', 'frequency_support', 'velocity_resolution', 'sensitivity_10kms', 
                        'cont_sensitivity_bandwidth', 'scan_intent', 's_ra', 's_dec', 'antenna_arrays']]
observation_metadata = observation_metadata.dropna()
scan_intents = observation_metadata['scan_intent'].values
ids = []
for i, scan in enumerate(scan_intents):
    if scan == 'TARGET':
        ids.append(i)
ids = np.array(ids)
observation_metadata = observation_metadata.iloc[ids, :]
observation_metadata = observation_metadata[(observation_metadata['s_resolution'] == observation_metadata['spatial_resolution'])]
observation_metadata = observation_metadata.drop(columns=['s_resolution'])     
observation_metadata = observation_metadata[['spatial_resolution', 't_exptime', 'em_min','em_max', 'em_resolution', 
            'bandwidth', 'frequency', 'frequency_support', 'velocity_resolution', 's_ra', 's_dec', 't_resolution', 'antenna_arrays']]

observation_metadata['velocity_resolution'] = observation_metadata['velocity_resolution'] * 10**(-3)
observation_metadata = observation_metadata.rename(columns={'spatial_resolution': 'spatial_resolution [arcsec]',
                                                            't_exptime': 'integration_time [s]',
                                                            'em_resolution': 'frequency_resolution [m]',
                                                            'bandwidth': 'bandwidth [MHz]',
                                                            'frequency': 'frequency [GHz]',
                                                            'velocity_resolution': 'velocity_resolution [Km/s]',
                                                            'em_min': 'frequency_min [m]', 'em_max': 'frequency_max [m]',
                                                            's_ra': 'ra [deg]', 's_dec': 'dec [deg]'})
observation_metadata['bandwidth [MHz]'] = observation_metadata['bandwidth [MHz]'].values * 10**(-6)
coords = SkyCoord(ra=observation_metadata['ra [deg]'], dec=observation_metadata['dec [deg]'], unit='deg').to_string('hmsdms')
freq_sup = observation_metadata['frequency_support'].values
freq_res = [int(float(f.split(',')[1][:-3]))* 10 ** (-3)  for f in freq_sup]
observation_metadata['frequency_resolution [MHz]'] = freq_res
b = 'J2000 '
coords = [b + i for i in coords]
observation_metadata['coords'] = coords
observation_metadata.rename(columns={'coords': 'coords [J2000]'}, inplace=True)
antenna_arrays = observation_metadata['antenna_arrays'].values
antenna_arrays = [[a.split(':')[0] for a in antenna_arrays[i].split(' ')] for i in range(len(antenna_arrays))]
observation_metadata['pads'] = antenna_arrays
to_save = observation_metadata[['spatial_resolution [arcsec]', 'integration_time [s]', 
                                'velocity_resolution [Km/s]', 'ra [deg]', 'dec [deg]', 
                                'bandwidth [MHz]', 'frequency [GHz]', 'pads', 
                                'frequency_resolution [MHz]', 'coords [J2000]']]
to_save.to_csv(outpath, index=False)