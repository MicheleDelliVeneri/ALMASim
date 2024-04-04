from astropy.constants import c
import astropy.units as U
import math

def estimate_alma_beam_size(central_frequency_ghz, max_baseline_km):
  """
  Estimates the beam size of the Atacama Large Millimeter/submillimeter Array (ALMA) in arcseconds.

  This function provides an approximation based on the theoretical relationship between
  observing frequency and maximum baseline. The formula used is:
  beam_size = (speed_of_light / central_frequency) / max_baseline * (180 / pi) * 3600 arcseconds
  [km]/[s] * [s] / [km] = [radians] * [arcsec /radian] * [arcseconds/degree]

  Args:
      central_frequency_ghz: Central frequency of the observing band in GHz (float).
      max_baseline_km: Maximum baseline of the antenna array in kilometers (float).

  Returns:
      Estimated beam size in arcseconds (float).

  Raises:
      ValueError: If either input argument is non-positive.
  """

  # Input validation
  if central_frequency_ghz <= 0 or max_baseline_km <= 0:
    raise ValueError("Central frequency and maximum baseline must be positive values.")

  # Speed of light in meters per second
  light_speed = c.to(U.m / U.s).value

  # Convert frequency to Hz
  central_frequency_hz = central_frequency_ghz.to(U.Hz).value

  # Convert baseline to meters
  max_baseline_meters = max_baseline_km.to(U.m).value


  # Theoretical estimate of beam size (radians)
  theta_radians = (light_speed / central_frequency_hz) / max_baseline_meters

  # Convert theta from radians to arcseconds
  beam_size_arcsec = theta_radians * (180 / math.pi) * 3600 * U.arcsec

  return beam_size_arcsec

def get_fov_from_band(band, antenna_diameter: int = 12):
    """
    This function returns the field of view of an ALMA band in arcseconds
    input: 
        band number (int): the band number of the ALMA band, between 1 and 10
        antenna_diameter (int): the diameter of the antenna in meters
    output:
        fov (astropy unit): the field of view in arcseconds

    """
    light_speed = c.to(U.m / U.s).value
    if band == 1:
        central_freq = 43 * U.GHz  
    elif band == 2:
        central_freq = 67 * U.GHz
    elif band == 3:
        central_freq = 100 * U.GHz
    elif band == 4:
        central_freq = 150 * U.GHz
    elif band == 5:
        central_freq = 217 * U.GHz
    elif band == 6:
        central_freq = 250 * U.GHz
    elif band == 7:
        central_freq = 353 * U.GHz
    elif band == 8:
        central_freq = 545 * U.GHz
    elif band == 9:
        central_freq = 650 * U.GHz    
    elif band == 10:
        central_freq = 868.5 * U.GHz
    central_freq = central_freq.to(U.Hz).value
    central_freq_s = 1 / central_freq
    wavelength = light_speed * central_freq_s
    # this is the field of view in Radians
    fov = 1.22 * wavelength / antenna_diameter
    # fov in arcsec
    fov = fov * (180 / math.pi) * 3600 * U.arcsec
    return fov

def get_max_baseline_from_antenna_config(antenna_config):
    """
    takes an antenna configuration .cfg file as input and outputs
    """
    positions = []
    with open(antenna_config, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip().startswith('#'):
                if '\t' in line:
                    row = [x for x in line.split("\t")][:3]
                else:
                    row = [x for x in line.split(" ")][:3]
                positions.append([float(x) for x in row])  
    positions = np.array(positions)
    max_baseline = 2 * np.max(np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2])) / 1000
    return max_baseline