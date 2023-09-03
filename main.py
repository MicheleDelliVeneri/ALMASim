import dask
from dask.distributed import Client
import simulator as sm
import numpy as np
import pandas as pd
import os
import argparse
from random import choices

MALLOC_TRIM_THRESHOLD_ = 0

class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Welcome to ALMASim, the ALMA simulation package.\
                                 This is the main script to run the simulations. To use it you need to provide the following required arguments:\
                                --data_dir /full/path/to/where/you/want/to/store/the/simulations/outputs/ \
                                --main_path /full/path/to/ALMASim/ \
                                 The rest are set to default as follows:' ,
                                 formatter_class=SmartFormatter)

parser.add_argument('--n_sims', type=int, default=10, help='R|Number of simulations to perform, default 10.')
parser.add_argument('--data_dir', type=str, required=True, help='R|Directory where the all the simulations outputs are stored.')
parser.add_argument('--main_path', type=str, required=True,  help='R|Directory where the ALMASim package is stored.')
parser.add_argument('--output_dir', type=str, default='sims', help='R|Directory where the simulation fits outputs are within the data_dir, default sims.')
parser.add_argument('--project_name', type=str, default='sim', help='R|Name of the simalma project, leave it to default.')
parser.add_argument('--bands', type=int, default=[6],  nargs='+',  help='R|Bands to simulate, if a single band is given all simulations will be performed with the given band, otherwise bands are randomly extracted from the provided bands list. Default 6')
parser.add_argument('--antenna_config', type=int, default=[3],  nargs='+', help='R|Antenna configurations as a list of integers in the interval [1, 10]. If a single antenna configuration is given all simulations will be performed with the given configuration, otherwise configurations are randomly extracted from the provided configurations list. Default 3')
parser.add_argument('--inbright', type=float, default=[0.01], nargs='+', help='R|Input brightness in Jy/beam, if a single value is given all simulations will be performed with the given value, otherwise values are randomly sampled from a uniform distribution between the min and max values in the list. Defaylt 0.01')
parser.add_argument('--bandwidth', type=int, default=[1280], nargs='+', help='R|Bandwidth in MHz, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 1280')
parser.add_argument('--inwidth', type=int, default=[10], nargs='+', help='R|Input width in km/s, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 10')
parser.add_argument('--integration', type=int, default=[10], nargs='+', help='R|Integration time in seconds, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 10')
parser.add_argument('--totaltime', type=int, default=[4500], nargs='+', help='R|Total time in seconds, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 4500')
parser.add_argument('--pwv', type=float, default=[0.3], nargs='+', help='R|PWV in mm between 0 and 1, if a single value is given all simulations will be performed with the given value, otherwise alues are randomly sampled from a uniform distribution between the min and max values in the list. Default 0.3')
parser.add_argument('--snr', type=int, default=[30], nargs='+', help='R|SNR, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 30')
parser.add_argument('--get_skymodel', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the skymodel is laoded from the data_path. Default False.')
parser.add_argument('--source_type', type=str, default='point', nargs='?', help='R|SOURCE_TYPE, type of source to generate: "point", "diffuse" or "extended"')
parser.add_argument('--TNGBasePath', type=str, default=None, help='R|Path to the TNG data on your folder. Default /media/storage/TNG100-1/output')
parser.add_argument('--TNGSnapID', type=int, default=[99], nargs='+', help='R|Snapshot ID of the TNG data.  Default 99')
parser.add_argument('--TNGSubhaloID', type=int, default=[0], nargs='+', help='R|Subhalo ID of the TNG data. Default 0')
parser.add_argument('--plot', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the simulation results are plotted. Default False.')
parser.add_argument('--save_ms', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the measurement sets are preserved and stored as numpy arrays. Default False.')
parser.add_argument('--crop', type=str2bool, default=False, const=True, nargs='?',  help='R|If True, the simulation results are cropped to the size of the beam times 1.5. Default False.')
parser.add_argument('--n_px', type=int, default=None, help='R|Number of pixels in the simulation. Default None, if set simulations are spatially cropped to the given number of pixels.')
parser.add_argument('--n_channels', type=int, default=None, help='R|Number of channels in the simulation. Default None, if set simulations are spectrally cropped to the given number of channels.')
parser.add_argument('--n_workers', type=int, default=10, help='R|Number of workers to use. Default 10.')
parser.add_argument('--threads_per_worker', type=int, default=4, help='R|Number of threads per worker to use ~ 1  per physical CPU. Default 4.')



if __name__ == '__main__':
    args = parser.parse_args()
    dask.config.set(scheduler='threads')
    dask.config.set({'temporary_directory': args.data_dir})
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    output_dir = os.path.join(args.data_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plot_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    idxs = np.arange(0, args.n_sims)
    data_dir = [args.data_dir for i in idxs]
    main_path = [args.main_path for i in idxs]
    output_dir = [output_dir for i in idxs]
    plot_dir = [plot_dir for i in idxs]
    project_name = [args.project_name for i in idxs]
    bands = choices(args.bands, k=len(idxs))
    antenna_ids = choices(args.antenna_config, k=len(idxs))
    antenna_names = ['alma.cycle9.3.{}'.format(k) for k in antenna_ids]
    min_inbright, max_inbright = np.min(args.inbright), np.max(args.inbright)
    inbrights = np.random.uniform(min_inbright, max_inbright, size=len(idxs))
    bandwidths = choices(args.bandwidth, k=len(idxs))
    inwidths = choices(args.inwidth, k=len(idxs))
    integrations = choices(args.integration, k=len(idxs))
    totaltimes = choices(args.totaltime, k=len(idxs))
    min_pwv, max_pwv = np.min(args.pwv), np.max(args.pwv)
    pwvs = np.random.uniform(min_pwv, max_pwv, size=len(idxs))
    min_snr, max_snr = np.min(args.snr), np.max(args.snr)
    snrs = np.random.uniform(min_snr, max_snr, size=len(idxs))
    get_skymodel = [args.get_skymodel for i in idxs]
    source_type = [args.source_type for i in idxs]
    tng_basepaths = [bool(args.TNGBasePath) for i in idxs]
    tng_snapids = choices(args.TNGSnapID, k=len(idxs))
    if len(args.TNGSubhaloID) > args.n_sims:
        tng_subhaloids = choices(args.TNGSubhaloID, k=len(idxs))
    else:
        tng_subhaloids = sm.get_subhaloids_from_db(args.n_sims)
    plot = [args.plot for i in idxs]
    save_ms = [args.save_ms for i in idxs]
    crop = [args.crop for i in idxs]
    n_pxs = [args.n_px for i in idxs]
    n_channels = [args.n_channels for i in idxs]

    input_params = pd.DataFrame(zip(idxs, 
                                    data_dir, 
                                    main_path,
                                    project_name, 
                                    output_dir,
                                    plot_dir, 
                                    bands, 
                                    antenna_names, 
                                    inbrights, 
                                    bandwidths, 
                                    inwidths, 
                                    integrations, 
                                    totaltimes, 
                                    pwvs, 
                                    snrs, 
                                    get_skymodel, 
                                    source_type, 
                                    tng_basepaths, 
                                    tng_snapids,
                                    tng_subhaloids,
                                    plot, 
                                    save_ms, 
                                    crop,
                                    n_pxs, 
                                    n_channels), 
                                    columns=['idx', 'data_dir', 'main_path', 
                                            'project_name', 'output_dir', 'plot_dir', 'band',
                                            'antenna_name', 'inbright', 'bandwidth',
                                            'inwidth', 'integration', 'totaltime', 
                                            'pwv', 'snr', 'get_skymodel', 'source_type',
                                            'tng_basepath', 'tng_snapid', 'tng_subhaloid',
                                            'plot', 'save_ms', 'crop',
                                            'n_px', 'n_channels'])
    input_params.info()
    dbs = np.array_split(input_params, len(input_params) / args.n_workers)
    for db in dbs:
        client = Client(threads_per_worker=args.threads_per_worker, 
                    n_workers=args.n_workers, memory_limit='50GB' )
        futures = client.map(sm.simulator, *db.values.T)
        client.gather(futures)
        client.close()
    files = os.listdir(args.main_path)
    for item in files:
        if item.endswith(".log"):
            os.remove(os.path.join(args.main_path, item))