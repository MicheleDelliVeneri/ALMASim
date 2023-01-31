import os
import argparse
import tempfile
temp_dir = tempfile.TemporaryDirectory()
os.environ['MPLCONFIGDIR'] = temp_dir.name

from casatasks import simobserve, tclean, exportfits

def generate_sims(i, input_dir, output_dir, antennalist, coordinates, spatial_resolution, central_frequency, 
                  frequency_resolution, integration_time, map_size, n_px):
    filename = os.path.join(input_dir, "skymodel_cube_" + str(i) + ".fits")
    antenna_name = '.'.join(antennalist.split("/")[-1].split(".")[0:-1])
    project = 'sim'
    simobserve(
        project=project, 
        skymodel=filename,
        inbright="0.001Jy/pix",
        indirection=coordinates,
        incell=spatial_resolution,
        incenter=central_frequency,
        inwidth=frequency_resolution,
        setpointings=True,
        integration="10s",
        direction=coordinates,
        mapsize=map_size,
        maptype="hexagonal",
        obsmode="int",
        antennalist=antennalist,
        totaltime=integration_time,
        thermalnoise="tsys-atm",
        user_pwv=0.8,
        seed=11111,
        graphics="none",
        verbose=False,
        overwrite=True)

    tclean(
        vis=os.path.join(project, "{}.{}.noisy.ms".format(project, antenna_name)),
        imagename=project+'/{}.{}'.format(project, antenna_name),
        imsize=[int(n_px), int(n_px)],
        cell=[spatial_resolution],
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
parser.add_argument('coordinates', type=str, 
                    help="The coordinates of the simulated source")
parser.add_argument('spatial_resolution', type=str,
                    help="The spatial resolution of the simulated cube")
parser.add_argument('central_frequency', type=str,
                    help="The central frequency of the simulated cube")
parser.add_argument('frequency_resolution', type=str,
                    help="The frequency resolution of the simulated cube")
parser.add_argument('integration_time', type=str,
                    help="The integration time of the simulated cube")
parser.add_argument('map_size', type=str,
                    help="The map size of the simulated cube")
parser.add_argument('n_px', type=str,
                    help="The number of pixels of the simulated cube")

args = parser.parse_args()
input_dir = args.model_dir
output_dir = args.output_dir
i = args.i
antenna_config = args.antenna_config
coordinates = args.coordinates
spatial_resolution = args.spatial_resolution
central_frequency = args.central_frequency
frequency_resolution = args.frequency_resolution
integration_time = args.integration_time
map_size = args.map_size
n_px = args.n_px

generate_sims(i, input_dir, output_dir, antenna_config, coordinates, spatial_resolution, central_frequency, frequency_resolution, 
              integration_time, map_size, n_px)
