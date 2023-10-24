import dask
from dask.distributed import Client
import simulator as sm
import numpy as np
import pandas as pd
import os
import argparse
from random import choices
from natsort import natsorted
import math

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

parser.add_argument('--n_sims', type=int, default=10, help='R|Number of simulations to perform, default 10. If reference images are provided, this is the number of simulations will be performed for each reference image.')
parser.add_argument('--data_dir', type=str, required=True, help='R|Directory where the all the simulations outputs are stored.')
parser.add_argument('--main_path', type=str, required=True,  help='R|Directory where the ALMASim package is stored.')
parser.add_argument('--output_dir', type=str, default='sims', help='R|Directory where the simulation fits outputs are within the data_dir, default sims.')
parser.add_argument('--project_name', type=str, default='sim', help='R|Name of the simalma project, leave it to default.')
parser.add_argument('--bands', type=int, default=[6],  nargs='+',  help='R|Bands to simulate, if a single band is given all simulations will be performed with the given band, otherwise bands are randomly extracted from the provided bands list. Default 6')
parser.add_argument('--cycle', type=int, default=[9], nargs='+', help='R|Cycle of the ALMA project, if a single cycle is given all simulations will be performed with the given cycle, otherwise cycles are randomly extracted from the provided cycles list. Default 9')
parser.add_argument('--antenna_config', type=int, default=[3],  nargs='+', help='R|Antenna configurations as a list of integers in the interval [1, 10]. If a single antenna configuration is given all simulations will be performed with the given configuration, otherwise configurations are randomly extracted from the provided configurations list. Default 3')
parser.add_argument('--inbright', type=float, default=[0.01], nargs='+', help='R|Input brightness in Jy/beam, if a single value is given all simulations will be performed with the given value, otherwise values are randomly sampled from a uniform distribution between the min and max values in the list. Default 0.01')
parser.add_argument('--bandwidth', type=int, default=[1280], nargs='+', help='R|Bandwidth in MHz, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 1280')
parser.add_argument('--inwidth', type=int, default=[10], nargs='+', help='R|Input width in km/s, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 10')
parser.add_argument('--integration', type=int, default=[10], nargs='+', help='R|Integration time in seconds, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 10')
parser.add_argument('--totaltime', type=int, default=[4500], nargs='+', help='R|Total time in seconds, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 4500')
parser.add_argument('--pwv', type=float, default=[0.3], nargs='+', help='R|PWV in mm between 0 and 1, if a single value is given all simulations will be performed with the given value, otherwise alues are randomly sampled from a uniform distribution between the min and max values in the list. Default 0.3')
parser.add_argument('--snr', type=int, default=[30], nargs='+', help='R|SNR, if a single value is given all simulations will be performed with the given value, otherwise values are randomly extracted from the provided values list. Default 30')
parser.add_argument('--get_skymodel', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the skymodel is laoded from the data_path. Default False.')
parser.add_argument('--reference_source', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the reference sources are searched for within the reference folder from which to sample the observing parameters. This is performed in order to generate similar sources. Default False.')
parser.add_argument('--reference_dir', type=str, default=None, help='R|Path to the reference sources folder. Default None.')
parser.add_argument('--source_type', type=str, default='point', nargs='?', help='R|SOURCE_TYPE, type of source to generate: "point", "gaussian", "diffuse" or "extended"')
parser.add_argument('--TNGBasePath', type=str, default=None, help='R|Path to the TNG data on your folder. Default /media/storage/TNG100-1/output')
parser.add_argument('--TNGSnapID', type=int, default=[99], nargs='+', help='R|Snapshot ID of the TNG data.  Default 99')
parser.add_argument('--TNGSubhaloID', type=int, default=[0], nargs='+', help='R|Subhalo ID of the TNG data. Default 0')
parser.add_argument('--TNGAPIKey', type=str, default='8f578b92e700fae3266931f4d785f82c', help='R|API Key to download the TNG data. Default None')
parser.add_argument('--insert_serendipitous', type=str2bool, default=True, const=True, nargs='?', help='R|If True, serendipitous sources are injected in the simulation. Default True.')
parser.add_argument('--plot', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the simulation results are plotted. Default False.')
parser.add_argument('--save_ms', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the measurement sets are preserved and stored as numpy arrays. Default False.')
parser.add_argument('--save_psf', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the PSF is stored as a numpy array. Default False.')
parser.add_argument('--save_pb', type=str2bool, default=False, const=True, nargs='?', help='R|If True, the primary beam is stored as a numpy array. Default False.')
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
    
    if args.reference_source == True:
        idxs = np.arange(0, args.n_sims * len(os.listdir(args.reference_dir)))
    else:
        idxs = np.arange(0, args.n_sims)
    data_dir = [args.data_dir for i in idxs]
    main_path = [args.main_path for i in idxs]
    output_dir = [output_dir for i in idxs]
    plot_dir = [plot_dir for i in idxs]
    project_name = [args.project_name for i in idxs]

    # Getting the observing bands
    if args.reference_source == False:
        bands = choices(args.bands, k=len(idxs))
    else:
        if len(args.bands) == len(os.listdir(args.reference_dir)):
            bands = np.repeat(args.bands, args.n_sims)
        else:   
            bands = choices(args.bands, k=len(idxs))
    
    # Getting the parameters from the reference sources
    if args.reference_source == True:
        files = natsorted(os.listdir(args.reference_dir))
        reference_params = []
        for id, file_ in enumerate(files):
            for i in range(args.n_sims):
                reference_params.append(sm.get_info_from_reference(os.path.join(args.reference_dir, file_), plot_dir[0], id))
        reference_params = np.array(reference_params)   
        ras = reference_params[:, 0]
        decs = reference_params[:, 1]
        n_pxs = reference_params[:, 2]
        n_channels = reference_params[:, 3]
        inbrights = reference_params[:, 4]
        rest_frequencies = reference_params[:, 5]
        cycles = reference_params[:, 6]
        antenna_ids = reference_params[:, 7]
    else:
        ras = [0.0 for i in idxs]
        decs = [0.0 for i in idxs]
        n_pxs = [args.n_px for i in idxs]
        n_channels = [args.n_channels for i in idxs]
        min_inbright, max_inbright = np.min(args.inbright), np.max(args.inbright)
        inbrights = np.random.uniform(min_inbright, max_inbright, size=len(idxs))
        rest_frequencies = [1420.4 for i in idxs]
        cycles = choices(args.cycle, k=len(idxs))
        antenna_ids = choices(args.antenna_config, k=len(idxs))
    

    antenna_names = [os.path.join('cycle{}'.format(int(j)), 'alma.cycle{}.0.{}'.format(int(j), int(k))) for j, k in zip(cycles, antenna_ids)]

    if len(args.bandwidth) == len(idxs):
        bandwidths = args.bandwidth
    elif len(args.bandwidth) != len(idxs) and len(args.bandwidth) == len(os.listdir(args.reference_dir)) and args.reference_source == True:
        bandwidths = [np.repeat(args.bandwidth, args.n_sims)]
    else:
        bandwidths = choices(args.bandwidth, k=len(idxs))
    if args.n_channels != None and args.n_channels != 1:
        inwidths = [bw / args.n_channels for bw in bandwidths]
    else:
        if len(args.inwidth) == len(idxs):
            inwidths = args.inwidth
        elif len(args.inwidth) != len(idxs) and len(args.inwidth) ==  len(os.listdir(args.reference_dir)) and args.reference_source == True:
            inwidths = [np.repeat(args.inwidth, args.n_sims)]
        else:
            inwidths = choices(args.inwidth, k=len(idxs))
    if len(args.integration) == len(idxs):
        integrations = args.integration
    else:
        integrations = choices(args.integration, k=len(idxs))
    if len(args.totaltime) == len(idxs):
        totaltimes = args.totaltime
    elif len(args.totaltime) != len(idxs) and args.reference_source == True:
        totaltimes = [np.repeat(args.totaltime, args.n_sims)]
    else:
        totaltimes = choices(args.totaltime, k=len(idxs))
    min_pwv, max_pwv = np.min(args.pwv), np.max(args.pwv)
    pwvs = np.random.uniform(min_pwv, max_pwv, size=len(idxs))
    min_snr, max_snr = np.min(args.snr), np.max(args.snr)
    snrs = np.random.uniform(min_snr, max_snr, size=len(idxs))
    get_skymodel = [args.get_skymodel for i in idxs]
    source_type = [args.source_type for i in idxs]
    tng_basepaths = [args.TNGBasePath for i in idxs]
    tng_snapids = choices(args.TNGSnapID, k=len(idxs))
    if args.source_type == 'extended':
        for snapID in args.TNGSnapID:
            if sm.check_TNGBasePath(TNGBasePath=args.TNGBasePath, 
                                TNGSnapshotID=snapID, 
                                TNGSubhaloID=args.TNGSubhaloID) == False:
                print('TNG Data not found, downloading...')
                sm.download_TNG_data(path=args.TNGBasePath, TNGSnapshotID=snapID, 
                                    TNGSubhaloID=args.TNGSubhaloID, 
                                    api_key=args.TNGAPIKey)
            elif sm.check_TNGBasePath(TNGBasePath=args.TNGBasePath, 
                                  TNGSnapshotID=snapID, 
                                  TNGSubhaloID=args.TNGSubhaloID) == None:
                print('Warning: if source_type is extended, TNGBasePath must be provided.')
                exit()
        # setting the working directory to the ALMASim directory, 
        # needed if the TNG data is downloaded
        os.chdir(args.main_path)
        subhalo_limit = sm.get_subhalorange(args.TNGBasePath, args.TNGSnapID[0], args.TNGSubhaloID)
        if len(args.TNGSubhaloID) > args.n_sims:
            tng_subhaloids = np.random.randint(0, subhalo_limit, size=args.n_sims).tolist()
        else:
            tng_subhaloids = sm.get_subhaloids_from_db(args.n_sims, subhalo_limit)
    else:
        tng_subhaloids = [0 for i in idxs]
    insert_serendipitous = [args.insert_serendipitous for i in idxs]
    plot = [args.plot for i in idxs]
    save_ms = [args.save_ms for i in idxs]
    save_psf = [args.save_psf for i in idxs]
    save_pb = [args.save_pb for i in idxs]
    crop = [args.crop for i in idxs]
    
    print('Data directory: {}'.format(data_dir[0]))
    print('Main path: {}'.format(main_path[0]))
    print('Output directory: {}'.format(output_dir[0]))
    print('Plot directory: {}'.format(plot_dir[0]))
    print('Project name: {}'.format(project_name[0]))
    print('Bands: {}'.format(bands))
    print('Antenna names: {}'.format(antenna_names))
    print('Input brightness: {}'.format(inbrights))
    print('Bandwidths: {}'.format(bandwidths))
    print('Frequency Channel Widths: {}'.format(inwidths))
    print('Integration Times: {}'.format(integrations))
    print('Total Times: {}'.format(totaltimes))
    print('Right Ascensions: {}'.format(ras))
    print('Declinations: {}'.format(decs))
    print('PWVs: {}'.format(pwvs))
    print('Rest Frequencies: {}'.format(rest_frequencies))
    print('SNRs: {}'.format(snrs))
    print('Get Skymodel: {}'.format(get_skymodel[0]))
    print('Source Type: {}'.format(source_type[0]))
    print('TNG Basepaths: {}'.format(tng_basepaths[0]))
    print('TNG Snapshots: {}'.format(tng_snapids))
    print('TNG Subhaloids: {}'.format(tng_subhaloids))
    print('Plot: {}'.format(plot[0]))
    print('Save MS: {}'.format(save_ms[0]))
    print('Save PSF: {}'.format(save_psf[0]))
    print('Save PB: {}'.format(save_pb[0]))
    print('Crop: {}'.format(crop[0]))
    print('Insert Serendipitous: {}'.format(insert_serendipitous[0]))
    print('Number of pixels: {}'.format(n_pxs))
    print('Number of channels: {}'.format(n_channels))
    print('Number of workers: {}'.format(args.n_workers))
    print('Threads per worker: {}'.format(args.threads_per_worker))
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
                                    ras,
                                    decs,
                                    pwvs,
                                    rest_frequencies,
                                    snrs, 
                                    get_skymodel, 
                                    source_type, 
                                    tng_basepaths, 
                                    tng_snapids,
                                    tng_subhaloids,
                                    plot, 
                                    save_ms,
                                    save_psf,
                                    save_pb,
                                    crop,
                                    insert_serendipitous,
                                    n_pxs, 
                                    n_channels
                                    ), 
                                    columns=['idx', 'data_dir', 'main_path', 
                                            'project_name', 'output_dir', 'plot_dir', 'band',
                                            'antenna_name', 'inbright', 'bandwidth',
                                            'inwidth', 'integration', 'totaltime', 'ra', 'dec',
                                            'pwv', 'rest_frequency', 'snr', 'get_skymodel', 
                                            'source_type', 'tng_basepath', 'tng_snapid', 'tng_subhaloid',
                                            'plot', 'save_ms', 'save_psf', 'save_pb', 'crop', 'serendipitous',
                                            'n_px', 'n_channels'])
    input_params.info()
    dbs = np.array_split(input_params, math.ceil(len(input_params) / args.n_workers))
    for db in dbs:
        if len(db) > 1:
            client = Client(threads_per_worker=args.threads_per_worker, 
                    n_workers=args.n_workers, memory_limit='{}GB'.format(sm.get_mem_gb()) )
            futures = client.map(sm.simulator, *db.values.T)
            client.gather(futures)
            client.close()
        else:
            sm.simulator(*db.values.T) 
    files = os.listdir(args.main_path)
    for item in files:
        if item.endswith(".log"):
            os.remove(os.path.join(args.main_path, item))
    for dir in os.listdir(output_dir[0]):
        if os.path.isdir(os.path.join(output_dir[0], dir)) and dir != 'plots':
            os.system('rm -rf {}'.format(os.path.join(output_dir[0], dir)))