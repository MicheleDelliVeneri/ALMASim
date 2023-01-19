import os
import argparse
import tempfile
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name

from casatasks import simobserve, tclean, exportfits

def generate_sims(i, input_dir, output_dir, antennalist):
    filename = os.path.join(input_dir, "gauss_cube_" + str(i) + ".fits")
    antenna_name = '.'.join(antennalist.split("/")[-1].split(".")[0:-1])
    project = 'sim'
    simobserve(
        project=project, 
        skymodel=filename,
        inbright="0.001Jy/pix",
        indirection="J2000 03h59m59.96s -34d59m59.50s",
        incell="0.1arcsec",
        incenter="230GHz",
        inwidth="10MHz",
        setpointings=True,
        integration="10s",
        direction="J2000 03h59m59.96s -34d59m59.50s",
        mapsize=["10arcsec"],
        maptype="hexagonal",
        obsmode="int",
        antennalist=antennalist,
        totaltime="2400s",
        thermalnoise="tsys-atm",
        user_pwv=0.8,
        seed=11111,
        graphics="none",
        verbose=False,
        overwrite=True)

    tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=project+'/{}.{}'.format(project, antenna_name),
        imsize=[360, 360],
        cell=["0.1arcsec"],
        phasecenter="",
        specmode="cube",
        niter=0,
        fastnoise=False,
        calcpsf=True,
        pbcor=False
        )
    exportfits(imagename=project+'/{}.{}.image'.format(project, antenna_name), 
           fitsimage=output_dir + "/dirty_cube_" + str(i) +".fits", overwrite=True)
    exportfits(imagename=project+'/{}.{}.skymodel'.format(project, antenna_name), 
           fitsimage=output_dir + "/clean_cube_" + str(i) +".fits", overwrite=True)
    #os.system('rm -r {}'.format(project))

parser = argparse.ArgumentParser()
parser.add_argument("i", type=str, 
        help='the index of the simulation to be run;')
parser.add_argument("model_dir", type=str, 
        help='The directory in wich the simulated model cubes are stored;')
parser.add_argument("output_dir", type=str, 
         help='The directory in wich to store the simulated dirty cubes and corresponding skymodels;')
parser.add_argument('antenna_config', type=str, 
        help="The antenna configuration file")
args = parser.parse_args()
input_dir = args.model_dir
output_dir = args.output_dir
i = args.i
antenna_config = args.antenna_config

generate_sims(i, input_dir, output_dir, antenna_config)
