import os
import argparse
import tempfile
import time
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name

from casatasks import simobserve, tclean, exportfits

def run_tclean(i, input_dir, output_dir, antennalist):
    filename = os.path.join(input_dir, "gauss_cube_" + str(i) + ".fits")
    antenna_name = '.'.join(antennalist.split("/")[-1].split(".")[0:-1])
    project = 'sim'
    start_time = time.time()
    tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=project+'/{}.{}'.format(project, antenna_name),
        imsize=[360, 360],
        cell=["0.1arcsec"],
        phasecenter="",
        specmode="cube",
        niter=200,
        fastnoise=False,
        calcpsf=True,
        pbcor=False
        )
    exportfits(imagename=project+'/{}.{}.image'.format(project, antenna_name), 
           fitsimage=output_dir + "/tcleaned_cube_" + str(i) + ".fits")
    time_of_execution = time.time() - start_time
    with open('running_params.txt', 'w') as f:
        f.write('Execution took {} seconds'.format(time_of_execution))
    f.close()
parser =argparse.ArgumentParser()
parser.add_argument("i", type=str, 
        help='the index of the simulation to be run;')
parser.add_argument("model_dir", type=str, 
        help='The directory in wich the simulated model cubes are stored;')
parser.add_argument("output_dir", type=str, 
         help='The directory in wich to store the tcleaned cubes;')
parser.add_argument('antenna_config', type=str, 
        help="The antenna configuration file")
args = parser.parse_args()
input_dir = args.model_dir
output_dir = args.output_dir
i = args.i
antenna_config = args.antenna_config
run_tclean(i, input_dir, output_dir, antenna_config)