"""Utility functions for sky model generation."""
import numpy as np
import math
import time
from scipy.ndimage import zoom
from astropy.io import fits
from astropy.units import Quantity
from dask.distributed import Client


def interpolate_array(arr: np.ndarray, n_px: int) -> np.ndarray:
    """Interpolates a 2D array to have n_px pixels while preserving aspect ratio."""
    x_zoom_factor = n_px / arr.shape[0]
    y_zoom_factor = n_px / arr.shape[1]
    return zoom(arr, [x_zoom_factor, y_zoom_factor])


def track_progress(update_progress, futures):
    """Track progress of dask futures and emit progress updates."""
    if update_progress is not None:
        total_tasks = len(futures)
        completed_tasks = 0
        # Track progress
        while completed_tasks < total_tasks:
            completed_tasks = sum(f.done() for f in futures)
            progress_value = int((completed_tasks / total_tasks) * 100)
            update_progress.emit(progress_value)  # Emit progress signal
            time.sleep(1)  # Check progress every second


def gaussian(x: np.ndarray, amp: float, cen: float, fwhm: float) -> np.ndarray:
    """
    Generates a 1D Gaussian given the following input parameters:
    
    Parameters
    ----------
    x : np.ndarray
        Position array
    amp : float
        Amplitude
    cen : float
        Center position
    fwhm : float
        Full width at half maximum
        
    Returns
    -------
    np.ndarray
        Normalized Gaussian profile
    """
    gaussian = np.exp(-((x - cen) ** 2) / (2 * (fwhm / 2.35482) ** 2))
    if np.sum(gaussian) != 0:
        norm = amp / np.sum(gaussian)
    else:
        norm = amp
    result = norm * gaussian
    return result


def get_datacube_header(datacube, obs_date: str) -> fits.Header:
    """Generate FITS header for a datacube."""
    datacube.drop_pad()
    datacube.freq_channels()
    wcs_header = datacube.wcs.to_header()
    wcs_header.rename_keyword("WCSAXES", "NAXIS")
    header = fits.Header()
    header.append(("SIMPLE", "T"))
    header.append(("BITPIX", 16))
    header.append(("NAXIS", wcs_header["NAXIS"]))
    header.append(("NAXIS1", datacube.n_px_x))
    header.append(("NAXIS2", datacube.n_px_y))
    header.append(("NAXIS3", datacube.n_channels))
    header.append(("EXTEND", "T"))
    header.append(("CDELT1", wcs_header["CDELT1"]))
    header.append(("CRPIX1", wcs_header["CRPIX1"]))
    header.append(("CRVAL1", wcs_header["CRVAL1"]))
    header.append(("CTYPE1", wcs_header["CTYPE1"]))
    header.append(("CUNIT1", wcs_header["CUNIT1"]))
    header.append(("CDELT2", wcs_header["CDELT2"]))
    header.append(("CRPIX2", wcs_header["CRPIX2"]))
    header.append(("CRVAL2", wcs_header["CRVAL2"]))
    header.append(("CTYPE2", wcs_header["CTYPE2"]))
    header.append(("CUNIT2", wcs_header["CUNIT2"]))
    header.append(("CDELT3", wcs_header["CDELT3"]))
    header.append(("CRPIX3", wcs_header["CRPIX3"]))
    header.append(("CRVAL3", wcs_header["CRVAL3"]))
    header.append(("CTYPE3", wcs_header["CTYPE3"]))
    header.append(("CUNIT3", wcs_header["CUNIT3"]))
    header.append(("OBJECT", "MOCK"))
    header.append(("BUNIT", "Jy/beam"))
    header.append(("MJD-OBS", obs_date))
    header.append(("BTYPE", "Intensity"))
    header.append(("SPECSYS", wcs_header["SPECSYS"]))
    keywords_to_remove = ["BMAJ", "BMIN", "BPA"]
    # Iterate over the keywords in the header
    for key in keywords_to_remove:
        while key in header:
            del header[key]
    header.append(("BMIN", abs(wcs_header["CDELT1"]) * 5))
    header.append(("BMAJ", abs(wcs_header["CDELT2"]) * 5))
    return header


