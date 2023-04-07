import dask
from dask.distributed import Client, progress
import simulator as sm
import numpy as np
import pandas as pd 
client = Client(threads_per_worker=4, n_workers=10)
client

idxs = np.arange(0, 10)
data_dir = ['/media/storage' for i in idxs]
main_path = ['/home/deepfocus/ALMASim' for i in idxs]
output_dir = ['sims' for i in idxs]
project_name = ['sim' for i in idxs]
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

input_params = pd.DataFrame(zip(idxs, data_dir, main_path, output_dir, project_name, bands, antenna_name, inbright, bandwidth, inwidth, integration, totaltime, pwv, snr, get_skymodel, extended, plot), columns=['idx', 'data_dir', 'main_path', 'output_dir', 'project_name', 'bands', 'antenna_name', 'inbright', 'bandwidth', 'inwidth', 'integration', 'totaltime', 'pwv', 'snr', 'get_skymodel', 'extended', 'plot'])
input_params.info()
futures = client.map(sm.simulator, input_params.values)