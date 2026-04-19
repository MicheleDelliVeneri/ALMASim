import numpy as np
import astropy.coordinates as coord
from astropy.coordinates import EarthLocation
import astropy.time
import astropy.units as u
from astropy.time import Time as _Time
from pathlib import Path

import numpy  as _np

def generate_via_astropy(antenna_positions, ra, dec, time: _Time):
    n_antennas = antenna_positions.shape[0]
    antenna_positions = EarthLocation(x=antenna_positions[:,0],
                                      y=antenna_positions[:,1],
                                      z=antenna_positions[:,2])
    
    alma_site = EarthLocation.of_site("ALMA")

    # Convert antenna pos terrestrial to celestial.  For astropy use 
    # get_gcrs_posvel(t)[0] rather than get_gcrs(t) because if a velocity 
    # is attached to the coordinate astropy will not allow us to do additional 
    # transformations with it (https://github.com/astropy/astropy/issues/6280)
    alma_p, alma_v = alma_site.get_gcrs_posvel(time)
    antenna_positions_gcrs = coord.GCRS(antenna_positions.get_gcrs_posvel(time)[0], 
                                        obstime=time, obsgeoloc=alma_p, obsgeovel=alma_v)

    # Define the UVW frame relative to a certain point on the sky.  There are
    # two versions, depending on whether the sky offset is done in ICRS 
    # or GCRS:
    pnt = coord.SkyCoord(ra, dec, frame='icrs')
    #frame_uvw = pnt.skyoffset_frame() # ICRS
    frame_uvw = pnt.transform_to(antenna_positions_gcrs).skyoffset_frame() # GCRS

    # Rotate antenna positions into UVW frame.
    antenna_positions_uvw = antenna_positions_gcrs.transform_to(frame_uvw).cartesian

    # Full set of baselines would be differences between all pairs of 
    # antenna positions, but we'll just do relative to the first antenna
    # for simplicity.
    baselines = np.zeros((n_antennas * (n_antennas-1)//2, 3), dtype=np.float64)
    count = 0
    for i in range(n_antennas):
        for j in range(i):
            values =  (antenna_positions_uvw[i] - antenna_positions_uvw[j]).xyz
            baselines[count, :] = values
            count += 1
    return baselines



ANTENNA_CONFIG = Path(__file__).parents[3] / "src" / "almasim" / "antenna_config" / "antenna_coordinates.csv"


def read_coordinates(coordinates_file):
    coordinates = _np.loadtxt(coordinates_file, delimiter=",", skiprows=1, usecols=(1,2,3))
    return coordinates

def test_uvw_casacore():
    coordinates = read_coordinates(ANTENNA_CONFIG) * u.m
    baselines = generate_via_astropy(coordinates, 3.26 * u.rad, -1.05 * u.rad, _Time.now())
    

"""
# Rest of script does the same(?) thing in CASA for comparison.
# Requires casatools from CASA 6 which can be installed via:
# pip install --index-url https://casa-pip.nrao.edu/repository/pypi-casa-release/simple casatools==6.0.0.27
try:
    import casatools
except ImportError:
    casatools = None

if casatools is not None:

    def casa_to_astropy(c):
        Convert CASA spherical coords to astropy CartesianRepresentation
        sph = coord.SphericalRepresentation(
                lon=c['m0']['value']*u.Unit(c['m0']['unit']), 
                lat=c['m1']['value']*u.Unit(c['m1']['unit']), 
                distance=c['m2']['value']*u.Unit(c['m2']['unit']))
        return sph.represent_as(coord.CartesianRepresentation)

    # The necessary interfaces:
    me = casatools.measures()
    qa = casatools.quanta()
    qq = qa.quantity

    # Init CASA frame info:
    me.doframe(me.observatory('VLA'))
    me.doframe(me.epoch('UTC',qq(t.mjd,'d')))
    me.doframe(me.direction('J2000',
        qq(pnt.ra.to(u.rad).value, 'rad'),
        qq(pnt.dec.to(u.rad).value, 'rad')))

    # Format antenna positions for CASA:
    antpos_casa = me.position('ITRF',
            qq(antpos[:,0].to(u.m).value,'m'),
            qq(antpos[:,1].to(u.m).value,'m'),
            qq(antpos[:,2].to(u.m).value,'m'))

    # Converts from ITRF to "J2000":
    antpos_c_casa = me.asbaseline(antpos_casa)

    # Rotate into UVW frame
    antpos_uvw_casa = me.touvw(antpos_c_casa)[0]

    # me.expand would compute all pairs of baselines but here we convert
    # to astropy CartesianRepresentation, and only do baselines to the
    # first antenna for easier comparison
    bl_uvw_casa = casa_to_astropy(antpos_uvw_casa)
    bl_uvw_casa -= bl_uvw_casa[0]

"""