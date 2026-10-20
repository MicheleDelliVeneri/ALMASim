"""Physical and ALMA constants shared by the pipeline and the simulator (MADIV-28 T2.1).

Every copy now uses one exact value (maintainer decision, 2026-10-20): the
Gaussian FWHM-to-sigma and beam-area factors are computed, not rounded, and the
ALMA site is astropy's ``EarthLocation.of_site("ALMA")`` written out, so no
site registry or network is needed.
"""

from __future__ import annotations

import math

SPEED_OF_LIGHT_M_S = 299_792_458.0
SPEED_OF_LIGHT_KM_S = 299_792.458
ARCSEC_PER_RAD = 206_264.806_247_096_36  # 180 / pi * 3600

# Gaussian: FWHM = FWHM_PER_SIGMA * sigma; solid angle = GAUSSIAN_BEAM_AREA * FWHM_maj * FWHM_min.
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
GAUSSIAN_BEAM_AREA = math.pi / (4.0 * math.log(2.0))

# astropy's site registry entry "alma" (https://almascience.eso.org/about-alma/alma-site).
ALMA_LATITUDE_DEG = -23.029
ALMA_LONGITUDE_DEG = -67.755
ALMA_ELEVATION_M = 5000.0

DISH_DIAMETER_12M_M = 12.0
DISH_DIAMETER_7M_M = 7.0
