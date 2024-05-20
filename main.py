import dask
from distributed import Client, LocalCluster, WorkerPlugin
import multiprocessing
import dask.dataframe as dd
import numpy as np
import pandas as pd
import os
from random import choices
from natsort import natsorted
import math
from itertools import product
import random
import psutil
import sys
#current_path = os.getcwd()
#parent_dir = os.path.join(current_path, "..")
#sys.path.append(parent_dir)
import utility.alma as ual
import utility.astro as uas
import utility.compute as uc
import warnings
from os.path import isfile, expanduser
import subprocess

RED = '\033[31m'
BLUE = '\033[34m'
YELLOW = '\033[33m'
RESET = '\033[0m'

warnings.simplefilter(action="ignore", category=UserWarning)
MALLOC_TRIM_THRESHOLD_ = 0
class MemoryMonitor(WorkerPlugin):
    def __init__(self, memory_limit):
        self.memory_limit = memory_limit

    def setup(self, worker):
        self.worker = worker
        self.process = psutil.Process()
        self.process_memory_limit = self.memory_limit * 1024 * 1024 * 1024  # Convert GB to bytes

    async def monitor_memory(self):
        while True:
            memory_usage = self.process.memory_info().rss
            if memory_usage > self.process_memory_limit:
                print("Memory limit exceeded. Closing worker.")
                await self.worker.close(close_workers=True)
                break
            await asyncio.sleep(1)

class MemoryLimitPlugin(WorkerPlugin):
    def __init__(self, memory_limit):
        self.memory_limit = memory_limit

    def setup(self, worker):
        pass

    def teardown(self, worker):
        pass

    def transition(self, key, start, finish, *args, **kwargs):
        if finish == 'memory' and psutil.virtual_memory().percent > self.memory_limit:
            # If memory usage exceeds the limit, skip the task
            return 'erred'

if __name__ == '__main__':
    # Creating Working directories
    main_path = os.getcwd()
    #output_dir = input("Insert absolute path of the output directory, if this is the first time running ALMASim this directory will be created: ")
    #output_dir = "/srv/Fast01/delliven/almasim-test-24-5-14"
    output_dir = '/Users/michele/Documents/almasim-test-24-5-18'
    #tng_dir = input("Insert absolute path of the TNG directory, if this is the firt time running ALMASim this directory will be created: ")
    tng_dir = "/Users/michele/Documents/TNGData"
    #project_name = input(f"{RED}Insert the name of the project: {RESET}")
    project_name = 'test-int'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tng_dir):
        os.makedirs(tng_dir)
    output_path = os.path.join(output_dir, project_name)
    if not os.path.exists(os.path.join(output_dir, project_name)):
        os.makedirs(output_path)
    plot_dir = os.path.join(output_path, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    
    # Getting Sims Configuration
    #n_sims = input(f"{BLUE}Insert number of simulations to run: {RESET}")
    n_sims = 1
    try:
        n_sims = int(n_sims)
    except ValueError:
        print(f"{YELLOW}Invalid input. Please insert an integer.{RESET}")
        n_sims = input(f"{BLUE}Insert number of simulations to run: {RESET}")
        n_sims = int(n_sims)
    
    sim_idxs = np.arange(n_sims)
    #ncpu = input(f"{RED}Insert total number of CPUs to use: {RESET}")
    ncpu = 10
    try:
        ncpu = int(ncpu)
    except ValueError:
        print(f"{YELLOW}Invalid input. Please insert an integer.{RESET}")
        ncpu = input(f"{BLUE}Insert total number of CPUs to use: {RESET}")
        ncpu = int(ncpu)
    #query = input(f'{BLUE}Do you want to query for metadata or get an available file stored in the metadata directory? (query/get) {RESET}')
    query = 'get'
    if query != 'query' and query != 'get':
        print(f"{YELLOW}Invalid input. Please insert query or get.{RESET}")
        query = input(f'{RED}Do you want to query for metadata or get an available file stored in the metadata directory? (query/get) {RESET}')
    if query == 'query':
        query_mode = input(f"{BLUE}Do you have a target list for the ALMA Database or do you want to query by science case? (target/science): {RESET}")
        if query_mode != "target" and query_mode != "science":
            print(f"{YELLOW}Invalid input. Please insert target or science.{RESET}")
            query_mode = input(f"{RED}Do you have a target list for the ALMA Database or do you want to query by science case? (target/science): {RESET}")
        if query_mode == "target":
            target_list = input(f"{BLUE}Insert the absolute path of the target list .csv file. This file should contain two columns with the target name and the target uid: {RESET}")
            if not isfile(target_list):
                print(f"{YELLOW}File not found.{RESET}")
                target_list = input(f"{RED}File not found. Please provide the correct path: {RESET}")
            target_list = pd.read_csv(target_list).values
            target_list = target_list.tolist()
            metadata_name = input(f"{BLUE}Queried metadata will be saved as a .csv file in the metadata folder: {RESET}")
            metadata = ual.query_for_metadata_by_targets(target_list, os.path.join(main_path, "metadata", metadata_name))
        else:
            metadata_name = input(f"{RED}Queried metadata will be saved as a .csv file in the metadata folder: {RESET}")
            if '.csv' not in metadata_name:
                metadata_name = metadata_name.split('.')[0]
                metadata_name = metadata_name + '.csv'
            metadata = ual.query_for_metadata_by_science_type(metadata_name, main_path)
    else:
        #metadata_name = input(f"{BLUE}Insert the name of the metadata file you want to use. Make sure to add .csv: {RESET}")
        metadata_name = 'AGN_all_bands'
        if '.csv' not in metadata_name:
            metadata_name = metadata_name.split('.')[0]
            metadata_name = metadata_name + '.csv'
        metadata = uc.load_metadata(main_path, metadata_name)
    #line_mode = input(f"{RED}Do you want to simulate a specific line/s? (y/n) {RESET}")
    line_mode = 'n'
    if line_mode != "y" and line_mode != "n":
        print(f"{YELLOW}Invalid input. Please insert y or n.{RESET}")
        line_mode = input(f"{BLUE}Do you want to simulate a specific line/s? (y/n) {RESET}")
    if line_mode == "y":
        uas.line_display(main_path)
        line_idxs = input(f"{RED}Select the line/s you want to simulate, separated by a space: {RESET}")
        line_idxs = [int(ix) for ix in line_idxs.split(' ')]
        rest_freq, line_names = uas.get_line_info(main_path, line_idxs)
        if len(rest_freq) == 1:
            rest_freq = rest_freq[0]
        rest_freqs = np.array([rest_freq]*n_sims)
        redshifts = np.array([None]*n_sims)
        n_lines = np.array([None]*n_sims)
        line_names = np.array([line_names]*n_sims)
        z1 = None
    else:
        #redshifts = input(f'{BLUE}Please provide the boundaries of the redshift interval you want to simulate as two float or integers separated by a space. If a single value is given, all simualtions will be performed at the same redshift: {RESET}')
        redshifts = '0.2'
        redshifts = redshifts.split()
        if len(redshifts) == 1:
            redshifts = np.array([float(redshifts[0])] * n_sims)
            z0, z1 = float(redshifts[0]), float(redshifts[0])
        else:
            z0, z1 = float(redshifts[0]), float(redshifts[1])
            redshifts = np.random.uniform(z0, z1, n_sims)
        #n_lines = input(f'{RED}Please provide the number of lines you want to simulate as an integer: {RESET}')
        n_lines = '1'
        n_lines = np.array([int(n_lines)]*n_sims)
        rest_freq, _ = uas.get_line_info(main_path)
        line_names = np.array([None]*n_sims)
        rest_freqs = np.array([None]*n_sims)
    
    #set_infrared = input(f'{BLUE}Do you want to provide infrared luminosities for SED normalization? (y/n), if not provided, they will be automatically computed based on the minimum continuum flux observable by the ALMA configuration: {RESET}')
    set_infrared  = 'y'
    if set_infrared != "y" and set_infrared != "n":
        print(f"{YELLOW}Invalid input. Please insert y or n.{RESET}")
        set_infrared = input(f'{BLUE}Do you want to provide infrared luminosities for SED normalization? (y/n), if not provided, they will be automatically computed based on the minimum continuum flux observable by the ALMA configuration: {RESET}')
    if set_infrared == "y":
        #lum_infrared = input(f'{RED}Insert infrared luminosity (in solar masses), you can input a single value, or an interval as two floats (es. 1e10) separated by a space: {RESET}')
        lum_infrared = '1e12'
        lum_infrared = [float(lum) for lum in lum_infrared.split()]
        if len(lum_infrared) == 1:
            lum_ir = np.array([lum_infrared[0]]*n_sims)
        else:
            lum_ir = np.random.uniform(lum_infrared[0], lum_infrared[1], n_sims)
    else:
        lum_ir = np.array([None]*n_sims)
    #set_snr = input(f'{RED}Do you want to provide a desired SNR for the simulated observations? (y/n) {RESET}')
    set_snr = 'n'
    if set_snr != "y" and set_snr != "n":
        print(f"{YELLOW}Invalid input. Please insert y or n.{RESET}")
        set_snr = input(f'{BLUE}Do you want to provide a desired SNR for the simulated observations? (y/n) {RESET}')
    if set_snr == "y":
        snr = input(f'{RED}Please provide the desired SNR as a float or an interval as two floats separated by a space: {RESET}')
        snr = [float(snr) for snr in snr.split()]
        if len(snr) == 1:
            snr = np.array([snr[0]]*n_sims)
        else:
            snr = np.random.uniform(snr[0], snr[1], n_sims)
    else:
        snr = np.ones(n_sims)
    
    #fix_spatial = input(f'{BLUE}Do you want to fix cube spatial dimensions? (y/n) {RESET}')
    fix_spatial = 'n'
    if fix_spatial != 'y' and fix_spatial != 'n':
        print(f"{YELLOW}Invalid input. Please insert y or n.{RESET}")
        fix_spatial = input(f'{RED}Do you want to fix cube spatial dimensions? (y/n) {RESET}')
    
    if fix_spatial == 'y':
        #n_pix = input(f'{BLUE}Insert the desired cube dimension in pixels: {RESET}')
        n_pix = '256'
        n_pix = int(n_pix)
    else:
        n_pix = None
    #fix_spectral = input(f'{RED}Do you want to fix cube spectral dimensions? (y/n) {RESET}')
    fix_spectral = 'y'
    if fix_spectral != 'y' and fix_spectral != 'n':
        print(f"{YELLOW}Invalid input. Please insert y or n.{RESET}")
        fix_spectral = input(f'{BLUE}Do you want to fix cube spectral dimensions? (y/n) {RESET}')
    if fix_spectral == 'y':
        #n_channels = input(f'{RED}Insert the desired number of channels: {RESET}')
        n_channels = '256'
        n_channels = int(n_channels)
    else:
        n_channels = None
    #source_type = input(f'{BLUE}Insert source type you want to simulate (point, gaussian, extended, diffuse): {RESET}')
    source_type = 'gaussian'
    if source_type != 'point' and source_type != 'gaussian' and source_type != 'extended' and source_type != 'diffuse':
        print(f"{YELLOW}Invalid input. Please insert point, gaussian, extended or diffuse.{RESET}")
        source_type = input(f'{RED}Insert source type you want to simulate (point, gaussian, extended, diffuse): {RESET}')
    if source_type == 'extended':
        print('Checking TNG Folders')
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1'))
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1', 'output')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1', 'output'))
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1', 'postprocessing')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1', 'postprocessing'))
        if not os.path.exists(os.path.join(tng_dir, 'TNG100-1', 'postprocessing', 'offsets')):
            os.makedirs(os.path.join(tng_dir, 'TNG100-1', 'postprocessing', 'offsets'))
    
        print('Checking simulation file')
        tng_api_key = '8f578b92e700fae3266931f4d785f82c'
        if not isfile(os.path.join(tng_dir, 'TNG100-1', 'simulation.hdf5')):
            print('Downloading simulation file')
            url = "http://www.tng-project.org/api/TNG100-1/files/simulation.hdf5"
            cmd = "wget -nv --content-disposition --header=API-Key:{} -O {} {}".format(tng_api_key, os.path.join(tng_dir, 'TNG100-1', 'simulation.hdf5'), url)
            subprocess.check_call(cmd, shell=True)
            print('Done.')
    
        tng_apis = [str(tng_api_key)*n_sims]
    else:
        tng_apis = np.array([None]*n_sims)    
    
    if source_type == 'extended': 
        metadata = uas.sample_given_redshift(metadata, n_sims, rest_freq, True, z1)
    else:
        metadata = uas.sample_given_redshift(metadata, n_sims, rest_freq, False, z1)
    print('\nMetadata retrieved\n')
    #inject_ser = input(f'{RED}Do you want to inject serendipitous sources? (y/n) {RESET}')
    inject_ser = 'n'
    if inject_ser != 'y' and inject_ser != 'n':
        print(f"{YELLOW}Invalid input. Please insert y or n.{RESET}")
        inject_ser = input(f'{BLUE}Do you want to inject serendipitous sources? (y/n) {RESET}')
    if inject_ser == 'y':
        inject_serendipitous = np.array([True] * n_sims)
    else:
        inject_serendipitous = np.array([False] * n_sims)
    ras = metadata['RA'].values
    decs = metadata['Dec'].values
    bands = metadata['Band'].values
    ang_ress = metadata['Ang.res.'].values
    vel_ress = metadata['Vel.res.'].values
    fovs = metadata['FOV'].values
    obs_dates = metadata['Obs.date'].values
    pwvs = metadata['PWV'].values
    int_times = metadata['Int.Time'].values
    total_times = metadata['Total.Time'].values
    bandwidths = metadata['Bandwidth'].values
    freqs = metadata['Freq'].values
    freq_supports = metadata['Freq.sup.'].values
    antenna_arrays = metadata['antenna_arrays'].values
    cont_sens = metadata['Cont_sens_mJybeam'].values
    n_pixs = np.array([n_pix]*n_sims)
    n_channels = np.array([n_channels]*n_sims)
    source_types = np.array([source_type]*n_sims)
    output_paths = np.array([output_path]*n_sims)
    tng_paths = np.array([tng_dir]*n_sims)
    main_paths = np.array([main_path]*n_sims)
    ncpus = np.array([ncpu]*n_sims)
    project_names = np.array([project_name]*n_sims)
    save_secondary = 'y'
    if save_secondary == 'y':
        save_secondary = True
    else:
        save_secondary = False
    save_secondary = np.array([save_secondary]*n_sims)
    
    input_params = pd.DataFrame(zip(
        sim_idxs, main_paths, output_paths, tng_paths, project_names, ras, decs, bands, ang_ress, vel_ress, fovs, 
        obs_dates, pwvs, int_times, total_times, bandwidths, freqs, freq_supports, cont_sens,
        antenna_arrays, n_pixs, n_channels, source_types,
        tng_apis, ncpus, rest_freqs, redshifts, lum_ir, snr,
        n_lines, line_names, save_secondary, inject_serendipitous), 
        columns = ['idx', 'main_path', 'output_dir', 'tng_dir', 'project_name', 'ra', 'dec', 'band', 
        'ang_res', 'vel_res', 'fov', 'obs_date', 'pwv', 'int_time', 'total_time', 'bandwidth', 
        'freq', 'freq_support', 'cont_sens', 'antenna_array', 'n_pix', 'n_channels', 'source_type',
        'tng_api_key', 'ncpu', 'rest_frequency', 'redshift', 'lum_infrared', 'snr',
        'n_lines', 'line_names', 'save_secondary', 'inject_serendipitous'])

    # Dask utils
    #dask.config.set({'temporary_directory': output_path})
    total_memory = psutil.virtual_memory().total
    num_processes = multiprocessing.cpu_count() // 4
    memory_limit = int(0.9 * total_memory / num_processes)
    #ddf = dd.from_pandas(input_params, npartitions=multiprocessing.cpu_count() // 4)
    #cluster = LocalCluster(n_workers=num_processes, threads_per_worker=4, dashboard_address=':8787')
    #output_type = "object"
    #client = Client(cluster)
    #client.register_worker_plugin(MemoryLimitPlugin(memory_limit))
    #results =  ddf.map_partitions(lambda df: df.apply(lambda row: uc.simulator(*row), axis=1), meta=output_type).compute()
    #client.close()
    #cluster.close()
    uc.simulator2(*input_params.iloc[0])
    uc.remove_logs(main_path)
    
