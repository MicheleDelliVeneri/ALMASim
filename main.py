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
    output_dir = "/mnt/storage/astro/almasim-test"
    #tng_dir = input("Insert absolute path of the TNG directory, if this is the firt time running ALMASim this directory will be created: ")
    tng_dir = "/mnt/storage/astro/TNGData"
    #project_name = input("Insert the name of the project: ")
    project_name = 'test'
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
    #n_sims = int(input("Insert number of simulations to run: "))
    n_sims = 1
    sim_idxs = np.arange(n_sims)
    #ncpu = input("Insert total number of CPUs to use: ")
    ncpu = 10
    #query_mode = input("Do you have a target list for the ALMA Database or do you want to query by science case? (target/science):")
    query_mode = "science"
    if query_mode == "target":
        target_list = input("Insert the absolute path of the target list .csv file. This file should contain two columns with the target name and the target uid: ")
        target_list = pd.read_csv(target_list).values
        target_list = target_list.tolist()
        metadata_name = input("Queried metadata will be saved as a .csv file. Insert the name of the file, make sure to add .csv: ")
        metadata = ual.query_for_metadata_by_targets(target_list, os.path.join(main_path, "metadata", metadata_name))
    else:
        #metadata_name = input("Queried metadata will be saved as a .csv file. Insert the name of the file, make sure to add .csv: ")
        metadata_name = "test.csv"
        metadata = ual.query_for_metadata_by_science_type(os.path.join(main_path, "metadata", metadata_name))
    #line_mode = input("Do you want to simulate a specific line? (y/n) ")
    line_mode = "y"
    if line_mode == "y":
        #line_name = input("Insert the name of the line you want to simulate: ")
        line_name = "CO(1-0)"
        rest_freq = uas.get_line_rest_frequency(line_name)
        rest_freqs = np.array([rest_freq]*n_sims)
        redshifts = np.array([None]*n_sims)
    else:
        #redshifts = input('Please provide the boundaries of the redshift interval you want to simulate as two float or integers separated by a space: ')
        redshifts = '1 2'
        z0, z1 = redshifts.split()
        z0, z1 = float(z0), float(z1)
        redshifts = np.random.uniform(z0, z1, n_sims)
        rest_freqs = np.array([None]*n_sims)

    #fix_spatial = input('Do you want to fix cube spatial dimensions? (y/n) ')
    fix_spatial = 'y'
    if fix_spatial == 'y':
        #n_pix = input('Insert the desired cube dimension in pixels: ')
        n_pix = 256
        n_pix = int(n_pix)
    else:
        n_pix = None
    #fix_spectral = input('Do you want to fix cube spectral dimensions? (y/n) ')
    fix_spectral = 'y'
    if fix_spectral == 'y':
        #n_channels = input('Insert the desired number of channels: ')
        n_channels = 256
        n_channels = int(n_channels)
    else:
        n_channels = None
    #source_type = input('Insert source type you want to simulate (point, gaussian, extended, diffuse): ')
    source_type = 'extended'
    if source_type == 'extended':
        tng_subhaloids = uas.get_subhaloids_from_db(n_sims, main_path)
    
    else:
        tng_subhaloids = np.array([None]*n_sims)    
    
    metadata = metadata.sample(n = n_sims)
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
    n_pixs = np.array([n_pix]*n_sims)
    n_channels = np.array([n_channels]*n_sims)
    source_types = np.array([source_type]*n_sims)
    output_paths = np.array([output_path]*n_sims)
    tng_paths = np.array([tng_dir]*n_sims)
    main_paths = np.array([main_path]*n_sims)
    ncpus = np.array([ncpu]*n_sims)
    project_names = np.array([project_name]*n_sims)
    input_params = pd.DataFrame(zip(
        sim_idxs, main_paths, output_paths, tng_paths, project_names, ras, decs, bands, ang_ress, vel_ress, fovs, 
        obs_dates, pwvs, int_times, total_times, bandwidths, freqs, freq_supports, 
        antenna_arrays, n_pixs, n_channels, source_types, 
        tng_subhaloids, ncpus, rest_freqs, redshifts), 
        columns = ['idx', 'main_path', 'output_dir', 'tng_dir', 'project_name', 'ra', 'dec', 'band', 
        'ang_res', 'vel_res', 'fov', 'obs_date', 'pwv', 'int_time', 'total_time', 'bandwidth', 
        'freq', 'freq_support', 'antenna_array', 'n_pix', 'n_channels', 'source_type', 'tng_subhaloid', 'ncpu', 
        'rest_freq', 'redshift'])
    
    
        # Dask utils
    dask.config.set({'temporary_directory': output_path})
    total_memory = psutil.virtual_memory().total
    num_processes = multiprocessing.cpu_count() // 4
    memory_limit = int(0.9 * total_memory / num_processes)
    ddf = dd.from_pandas(input_params, npartitions=multiprocessing.cpu_count() // 4)
    cluster = LocalCluster(n_workers=num_processes, threads_per_worker=4, dashboard_address=':8787')
    output_type = "object"
    client = Client(cluster)
    client.register_worker_plugin(MemoryLimitPlugin(memory_limit))
    results =  ddf.map_partitions(lambda df: df.apply(lambda row: uc.simulator(*row), axis=1), meta=output_type).compute()
    client.close()
    cluster.close()    