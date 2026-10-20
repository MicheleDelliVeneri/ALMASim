"""Physical and ALMA constants shared by the pipeline and the simulator (MADIV-28 T2.1).

Only values that every copy agreed on bit for bit live here so far. The
FWHM-to-sigma factor (2.3548, 2.35482, 2.354820045), the Gaussian beam-area
factor (1.1331) and the primary-beam/FOV factor (1.12, 1.13, 1.22) differ
between copies; choosing one changes products, so they wait for review.
"""

from __future__ import annotations

SPEED_OF_LIGHT_M_S = 299_792_458.0
SPEED_OF_LIGHT_KM_S = 299_792.458
ARCSEC_PER_RAD = 206_264.806_247_096_36  # 180 / pi * 3600

ALMA_LATITUDE_DEG = -23.028

DISH_DIAMETER_12M_M = 12.0
DISH_DIAMETER_7M_M = 7.0
