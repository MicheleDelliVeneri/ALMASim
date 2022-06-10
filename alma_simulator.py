import os
import argparse
from casatasks import simobserve, tclean, exportfits

def generate_sims(i, input_dir, output_dir):
    filename = os.path.join(input_dir, "gauss_cube_" + str(i) + ".fits")
    project = 'sim'
    antennalist = "alma.cycle9.3.cfg"
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
        vis=os.path.join(project, "{}.alma.cycle9.3.noisy.ms".format(project)),
        imagename=project+'/{}.alma.cycle9.3'.format(project),
        imsize=[360, 360],
        cell=["0.1arcsec"],
        phasecenter="",
        specmode="cube",
        niter=0,
        fastnoise=False,
        calcpsf=True,
        pbcor=False
        )
    exportfits(imagename=project+'/{}.alma.cycle9.3.image'.format(project), 
           fitsimage=output_dir + "/dirty_cube_" + str(i) +".fits")
    exportfits(imagename=project+'/{}.alma.cycle9.3.skymodel'.format(project), 
           fitsimage=output_dir + "/clean_cube_" + str(i) +".fits")
    os.system('rm -r {}'.format(project))

parser =argparse.ArgumentParser()
parser.add_argument("i", type=str, 
        help='the index of the simulation to be run;')
parser.add_argument("model_dir", type=str, 
        help='The directory in wich the simulated model cubes are stored;')
parser.add_argument("output_dir", type=str, 
         help='The directory in wich to store the simulated dirty cubes and corresponding skymodels;')

args = parser.parse_args()
input_dir = args.model_dir
output_dir = args.output_dir
i = args.i
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

generate_sims(i, input_dir, output_dir)