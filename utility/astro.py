import astropy.units as U
from collections import OrderedDict
import h5py
import illustris_python as il

def convert_to_j2000_string(ra_deg, dec_deg):
  """Converts RA and Dec in degrees to J2000 notation string format (e.g., "J2000 19h30m00 -40d00m00").

  Args:
      ra_deg (float): Right Ascension in degrees (0 to 360).
      dec_deg (float): Declination in degrees (-90 to 90).

  Returns:
      str: J2000 RA and Dec string ("J2000 hh:mm:ss.sss +/- dd:mm:ss.sss").

  Raises:
      ValueError: If RA or Dec values are outside valid ranges.
  """

  # Validate input values
  if not (0 <= ra_deg < 360):
      raise ValueError("Right Ascension (RA) must be between 0 and 360 degrees.")
  if not (-90 <= dec_deg <= 90):
      raise ValueError("Declination (Dec) must be between -90 and 90 degrees.")

  # Convert RA to sexagesimal string (hours:minutes:seconds)
  hours = int(ra_deg / 15)
  minutes = int((ra_deg % 15) * 60)
  seconds = (ra_deg % 15 - minutes / 60) * 3600  # Ensure higher precision

  ra_string = f"{hours:02d}h{minutes:02d}m{seconds:06.3f}"

  # Convert Dec to sexagesimal string (degrees:minutes:seconds)
  dec_dir = "+" if dec_deg >= 0 else "-"
  dec_deg = abs(dec_deg)
  degrees = int(dec_deg)
  minutes = int((dec_deg % 1) * 60)
  seconds = int((dec_deg % 1 - minutes / 60) * 3600)  # Ensure higher precision
  dec_string = f"{dec_dir}{degrees:02d}d{minutes:02d}m{seconds:02d}"

  # Combine RA and Dec strings in J2000 format
  return f"J2000 {ra_string} {dec_string}"

def convert_range_from_GHz_to_km_s(central_freq, central_velocity, freq_range):
    """
    Converts a frequency range in GHz to a velocity range in km/s.
    Args:
        central_freq (astropy Unit): Central frequency of the range in GHz.
        central_velocity (astropy Unit): Central velocity of the range in km/s.
        freq_range (astropy Unit): Frequency range in GHz.
    """
    freq_band = freq_range[1] - freq_range[0]
    freq_band = freq_band 
    dv = (c * freq_band  / central_freq).to(U.km / U.s)
    return dv

def compute_redshift(rest_frequency, observed_frequency):
    """
    Computes the redshift of a source given the rest frequency and the observed frequency.

    Args:
        rest_frequency (astropy Unit): Rest frequency of the source in GHz.
        observed_frequency (astropy Unit): Observed frequency of the source in GHz.

    Returns:
        float: Redshift of the source.

    Raises:
        ValueError: If either input argument is non-positive.
    """
    # Input validation
    if rest_frequency <= 0 or observed_frequency <= 0:
        raise ValueError("Rest and observed frequencies must be positive values.")

    # Compute redshift
    redshift = (rest_frequency.value - observed_frequency.value) / rest_frequency.value
    return redshift

def luminosity_to_jy(velocity, data, rest_frequency: float = 115.27):
        """
        This function takes as input a pandas db containing luminosities in K km s-1 pc2, redshifts, and luminosity distances in Mpc, 
        and returns the brightness values in Jy.
        
        Parameters:
        velocity (float): The velocity dispersion assumed for the line (Km s-1).
        data (pandas.DataFrame): A pandas DataFrame containing the data.
        rest_frequency (float): The rest frequency of the line in GHz. Defaults to 115.27 GHz for CO(1-0).

        Output:
        sigma: numpy.ndarray: An array of brightness values in Jy.

        """
        alpha = 3.255 * 10**7
        sigma = (data['Luminosity(K km s-1 pc2)'] * ( (1 + data['#redshift']) * rest_frequency **2)) / (alpha * velocity * (data['luminosity distance(Mpc)']**2))
        return sigma
        
def exponential_func(x, a, b):
        """
        Exponential function used to fit the data.
        """
        return a * np.exp(-b * x)

def redshift_to_snapshot(redshift):
    snap_db = {
        0: 20.05,
        1: 14.99,
        2: 11.98,
        3: 10.98,
        4: 10.00,
        5: 9.390,
        6: 9.000,
        7: 8.450,
        8: 8.010,
        9: 7.600,
        10: 7.24,
        11:	7.01,
        12:	6.49,
        13:	6.01,
        14:	5.85,
        15:	5.53,
        16:	5.23,
        17:	5.00,
        18:	4.66,
        19:	4.43,
        20:	4.18,
        21:	4.01,
        22:	3.71,
        23:	3.49,
        24:	3.28,
        25:	3.01,
        26:	2.90,
        27:	2.73,
        28:	2.58,
        29:	2.44,
        30:	2.32,
        31:	2.21,
        32:	2.10,
        33:	2.00,
        34:	1.90,
        35:	1.82,
        36:	1.74,
        37:	1.67,
        38:	1.60,
        39:	1.53,
        40:	1.50,
        41:	1.41,
        42:	1.36,
        43:	1.30,
        44:	1.25,
        45:	1.21,
        46:	1.15,
        47:	1.11,
        48:	1.07,
        49:	1.04,
        50:	1.00,
        51:	0.95,
        52:	0.92,
        53:	0.89,
        54:	0.85,
        55:	0.82,
        56:	0.79,
        57:	0.76,
        58:	0.73,
        59:	0.70,
        60:	0.68,
        61:	0.64,
        62:	0.62,
        63:	0.60,
        64:	0.58,
        65:	0.55,
        66:	0.52,
        67:	0.50,
        68:	0.48,
        69:	0.46,
        70:	0.44,
        71:	0.42,
        72:	0.40,
        73:	0.38,
        74:	0.36,
        75:	0.35,
        76:	0.33,
        77:	0.31,
        78:	0.30,
        79:	0.27,
        80:	0.26,
        81:	0.24,
        82:	0.23,
        83:	0.21,
        84:	0.20,
        85:	0.18,
        86:	0.17,
        87:	0.15,
        88:	0.14,
        89:	0.13,
        90:	0.11,
        91:	0.10,
        92:	0.08,
        93:	0.07,
        94:	0.06,
        95:	0.05,
        96:	0.03,
        97:	0.02,
        98:	0.01,
        99:	0,
    }
    snaps, redshifts = list(snap_db.keys())[::-1], list(snap_db.values())[::-1]
    for i in range(len(redshifts) - 1):
        if redshift >= redshifts[i] and redshift < redshifts[i + 1]:
            return snaps[i]