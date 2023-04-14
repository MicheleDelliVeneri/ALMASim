import dask
from dask.distributed import Client, progress
import simulator as sm
import numpy as np
import pandas as pd
import os
import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

parser = argparse.ArgumentParser(description='Simulate ALMA data cubes from TNG data cubes and Gaussian Simulations',
                                 formatter_class=SmartFormatter)

parser.add_argument('--n_sims', type=int, default=10, help='R|Number of simulations to perform.')
parser.add_argument('--data_dir', type=str, default='/media/storage', help='R|Directory where the all the simulations outputs are stored.')
parser.add_argument('--main_path', type=str, default='/home/deepfocus/ALMASim', help='R|Directory where the ALMASim package is stored.')
parser.add_argument('--output_dir', type=str, default='sims', help='R|Directory where the simulation fits outputs are within the data_dir.')
parser.add_argument('--project_name', type=str, default='sim', help='R|Name of the simalma project, leave it to default.')
parser.add_argument('--bands', type=list, default=[6], help='R|Bands to simulate, if a single band is given all simulations will be performed with \
                    the given band, otherwise bands are randomly extracted from the provided bands list.')
parser.add_argument('--antenna_config', type=list, default=[3], help='R|Antenna configurations as a list of integers in the interval [1, 10]. \
                    if a single antenna configuration is given all simulations will be performed with the given configuration, otherwise \
                     configurations are randomly extracted from the provided configurations list.')
parser.add_argument('--inbright', type=list, default=[0.001], help='R|Input brightness in Jy/beam, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--bandwidth', type=list, default=[1280], help='R|Bandwidth in MHz, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--inwidth', type=list, default=[10], help='R|Input width in km/s, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--integration', type=list, default=[10], help='R|Integration time in seconds, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--totaltime', type=list, default=[3600], help='R|Total time in seconds, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--pwv', type=list, default=[0.5], help='R|PWV in mm between 0 and 1, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--snr', type=list, default=[30], help='R|SNR, if a single value is given all simulations will be performed with \
                    the given value, otherwise values are randomly extracted from the provided values list.')
parser.add_argument('--get_skymodel', type=bool, default=False, help='R|If True, the skymodel is laoded from the data_path.')
parser.add_argument('--extended', type=bool, default=False, help='R|If True, extended skymodel using the TNG simulations are used, otherwise point like gaussians.')
parser.add_argument('--plot', type=bool, default=False, help='R|If True, the simulation results are plotted.')
parser.add_argument('--n_px', type=int, default=None, help='R|Number of pixels in the simulation.')
parser.add_argument('--n_channels', type=int, default=None, help='R|Number of channels in the simulation.')
parser.add_argument('--n_workers', type=int, default=10, help='R|Number of workers to use.')
parser.add_argument('--threads_per_worker', type=int, default=4, help='R|Number of threads per worker to use.')



if __name__ == '__main__':
    args = parser.parse_args()

    dask.config.set(scheduler='threads')
    client = Client(threads_per_worker=args.threads_per_worker, n_workers=args.n_workers)
    idxs = np.arange(0, args.n_sims)
    data_dir = [args.data_dir for i in idxs]
    main_path = [args.main_path for i in idxs]
    output_dir = [args.output_dir for i in idxs]
    project_name = [args.project for i in idxs]
    bands = np.random.randint(3, 6, size=len(idxs))
    antenna_name = ['alma.cycle9.3.3' for i in idxs]
    inbright = [0.001 for i in idxs]
    bandwidth  = [1280 for i in idxs]
    inwidth = [10 for i in idxs]
    integration = [10 for i in idxs]
    totaltime = [2500 for i in idxs]
    pwv = 0.5 * np.random.sample(size=len(idxs))
    snr = np.random.randint(10, 30, size=len(idxs))
    get_skymodel = [False for i in idxs]
    extended = [False for i in idxs]
    plot = [True for i in idxs]
    
    input_params = pd.DataFrame(zip(idxs, data_dir, main_path, project_name, output_dir,  bands, antenna_name, inbright, bandwidth, inwidth, integration, totaltime, pwv, snr, get_skymodel, extended, plot), columns=['idx', 'data_dir', 'main_path', 'project_name', 'output_dir', 'bands', 'antenna_name', 'inbright', 'bandwidth', 'inwidth', 'integration', 'totaltime', 'pwv', 'snr', 'get_skymodel', 'extended', 'plot'])
    input_params.info()
    futures = client.map(sm.simulator, *input_params.values.T)
    client.gather(futures)
    client.close()
    os.remove('*log')