from PyQt6.QtCore import QObject, pyqtSignal
from astropy.io import fits
from astropy.time import Time
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as U
from astropy.constants import c
from scipy.ndimage import zoom
import matplotlib.cm as cm
import numpy as np
import os
import matplotlib
import h5py
import time
from memory_profiler import memory_usage
from dask import delayed, compute
from dask.distributed import Client, LocalCluster, progress


def get_channel_wavelength(obs_wavelengths, channel):
    wavelength = list(obs_wavelengths[channel] * 1e-3)
    wavelength.append((wavelength[0] + wavelength[1]) / 2.0)
    return wavelength


def prepare_2d_arrays(Npix):
    beam = np.zeros((Npix, Npix), dtype=np.float32)
    totsampling = np.zeros((Npix, Npix), dtype=np.float32)
    dirtymap = np.zeros((Npix, Npix), dtype=np.float32)
    noisemap = np.zeros((Npix, Npix), dtype=np.complex64)
    robustsamp = np.zeros((Npix, Npix), dtype=np.float32)
    Gsampling = np.zeros((Npix, Npix), dtype=np.complex64)
    Grobustsamp = np.zeros((Npix, Npix), dtype=np.complex64)
    GrobustNoise = np.zeros((Npix, Npix), dtype=np.complex64)
    return (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    )


def prepare_baselines(
    Nant,
    nH,
    Hcov,
):
    """
    This method prepares the baselines for an interferometer array.
    The method first calculates the number of unique
    baselines in an array of N antennas.
    It then initializes several arrays to store baseline parameters and
    visibility data, including baseline vectors, baseline indices,
    antenna pair indices, complex gains, complex noise values, and hour angle values.
    The method also calculates the sine and cosine of the hour angles
    and stores them in `H`.
    It then iterates over all unique antenna pairs,
    assigning a baseline index to each pair
    and storing the antenna pair indices for each baseline.
    Finally, it initializes arrays to store u and v coordinates
    (in wavelengths) for each
    baseline at each hour angle, and sets `ravelDims`
    to the shape of these arrays.

    Attributes Set:
        Nbas (int): The number of unique baselines in the array.
        B (np.array): Baseline vectors for each baseline at each hour angle.
        basnum (np.array): A matrix storing the baseline index for each pair
                                of antennas.
        basidx (np.array): A square matrix storing the baseline index for each
                                pair of antennas.
        antnum (np.array): Stores the antenna pair indices for each baseline.
        Gains (np.array): Complex gains for each baseline at each hour angle.
        Noise (np.array): Complex noise values for each baseline at each
                               hour angle.
        Horig (np.array): Original hour angle values, evenly spaced over the
                               observation time.
        H (list): Trigonometric values (sine and cosine) of the hour angles.
        u (np.array): u coordinates (in wavelengths) for each baseline at
                           each hour angle.
        v (np.array): v coordinates (in wavelengths) for each baseline at
                           each hour angle.
        ravelDims (tuple): The shape of the u and v arrays.
    """
    # Calculate the number of unique baselines in an array of N antennas
    Nbas = Nant * (Nant - 1) // 2
    # Create a redundant variable for the number of baselines
    NBmax = Nbas
    # Initialize arrays to store baseline parameters and visibility data:
    # B: Baseline vectors (x, y, z components) for each baseline at each hour angle.
    B = np.zeros((NBmax, nH), dtype=np.float32)
    # basnum: A matrix storing the baseline index for each pair of antennas.
    basnum = np.zeros((Nant, Nant - 1), dtype=np.int8)
    # basidx: A square matrix storing the baseline index for each pair of
    # antennas (redundant storage).
    basidx = np.zeros((Nant, Nant), dtype=np.int8)
    # antnum: Stores the antenna pair indices for each baseline.
    antnum = np.zeros((NBmax, 2), dtype=np.int8)
    # Gains: Complex gains for each baseline at each hour angle (initialized to 1).
    Gains = np.ones((Nbas, nH), dtype=np.complex64)
    # Noise:  Complex noise values for each baseline at each hour angle
    # (initialized to 0).
    Noise = np.zeros((Nbas, nH), dtype=np.complex64)
    # Horig:  Original hour angle values, evenly spaced over the observation time.
    Horig = np.linspace(Hcov[0], Hcov[1], nH)
    H = Horig[np.newaxis, :]  # Add a new axis for broadcasting
    del Horig
    # Trigonometric values (sine and cosine) of the hour angles.
    H = [np.sin(H), np.cos(H)]
    bi = 0  # bi: Baseline index counter (starts at 0).
    # nii: List to keep track of the next available index in basnum for each antenna.
    nii = [0 for n in range(Nant)]
    # Iterate over all unique antenna pairs
    for n1 in range(Nant - 1):
        for n2 in range(n1 + 1, Nant):
            # Assign a baseline index to each antenna pair in both basnum and basidx.
            basnum[n1, nii[n1]] = np.int8(bi)
            basnum[n2, nii[n2]] = np.int8(bi)
            basidx[n1, n2] = np.int8(bi)
            # Store the antenna pair indices for the current baseline.
            antnum[bi] = [n1, n2]
            # Increment the next available index for each antenna in basnum.
            nii[n1] += 1
            nii[n2] += 1
            # Increment the baseline index counter
            bi += np.int16(1)
    # Initialize arrays to store u and v coordinates (in wavelengths)
    # for each baseline at each hour angle.
    u = np.zeros((NBmax, nH))
    v = np.zeros((NBmax, nH))
    ravelDims = (NBmax, nH)
    return Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims


def set_noise(noise, Noise):
    Noise[:] = np.random.normal(
        loc=0.0, scale=noise, size=np.shape(Noise)
    ) + 1.0j * np.random.normal(loc=0.0, scale=noise, size=np.shape(Noise))
    return Noise


def set_baselines(Nbas, antnum, B, u, v, antPos, trlat, trdec, H, wavelength):
    bas2change = range(Nbas)
    for currBas in bas2change:
        # Get the antenna indices that form the current baseline
        n1, n2 = antnum[currBas]
        # Calculate the baseline vector components (B_x, B_y, B_z) in wavelengths:
        # B_x: Projection of baseline onto the plane perpendicular to Earth's
        # rotation axis.
        B[currBas, 0] = -(antPos[n2][1] - antPos[n1][1]) * trlat[0] / wavelength[2]
        # B_y: Projection of baseline onto the East-West direction.
        B[currBas, 1] = (antPos[n2][0] - antPos[n1][0]) / wavelength[2]
        # B_z: Projection of baseline onto the North-South direction.
        B[currBas, 2] = (antPos[n2][1] - antPos[n1][1]) * trlat[1] / wavelength[2]
        # Calculate u and v coordinates (spatial frequencies) in wavelengths:
        # u: Projection of the baseline vector onto the UV plane (East-West
        # component).
        u[currBas, :] = -(B[currBas, 0] * H[0] + B[currBas, 1] * H[1])
        # v: Projection of the baseline vector onto the UV plane (North-South
        # component).
        v[currBas, :] = (
            -B[currBas, 0] * trdec[0] * H[1]
            + B[currBas, 1] * trdec[0] * H[0]
            + trdec[1] * B[currBas, 2]
        )
    return B, u, v


def _grid_uv(
    Nbas, totsampling, Gsampling, noisemap, u, v, Nphf, Gains, Noise, robust, nH, imsize
):
    """
    The main purpose of this method is to take the continuous visibility measurements
    collected by the interferometer (represented by u and v coordinates for each
     baseline) and "grid" them onto a discrete grid in the UV plane.
    Parameters:
        antidx (int): The index of the specific antenna for which to
                      grid the baselines.
                      If -1, all baselines are gridded. Default is -1.
    The method first determines which baselines to grid
     based on the provided antenna index.
    If no specific antenna is provided (antidx=-1), all
     baselines are gridded. If a specific antenna index
    is provided and it is less than the total number of antennas,
     only the baselines associated with
    that antenna are gridded. If the provided antenna index is invalid
     (greater than or equal to the total number
    of antennas), no baselines are gridded. The method then calculates
     the pixel size in the UV plane
    and initializes the baseline phases dictionary and the
    list of baselines to change.
    For each baseline in the list of baselines to change, the method calculates
     the pixel coordinates
    in the UV plane for each visibility sample of the current baseline.
     It then filters out visibility samples
    that fall outside the field of view (half-plane) and calculates the phase
     of the gains to introduce atmospheric errors.
    The method then calculates positive and negative pixel indices
     (accounting for the shift in the FFT)
    and updates the total sampling, gains, and noise at the
     corresponding pixel locations in the UV grid for
    the good pixels of the current baseline.
    Finally, the method calculates a robustness factor based on
     the total sampling and a user-defined parameter.
    The method modifies the following instance variables:
    - pixpos: A list of lists storing the pixel positions for each baseline.
    - totsampling: An array storing the total sampling for each pixel
                         in the UV grid.
    - Gsampling: An array storing the total gains for each pixel in the UV grid.
    - noisemap: An array storing the total noise for each pixel in the UV grid.
    - UVpixsize: The pixel size in the UV plane.
    - baseline_phases: A dictionary storing the phase of the gains
                            for each baseline.
    - bas2change: A list of the baselines to change.
    - robfac: A robustness factor based on the total
                     sampling and a user-defined parameter.
    """
    bas2change = range(Nbas)
    # Initialize lists to store pixel positions for each baseline.
    pixpos = [[] for nb in bas2change]
    # Reset sampling, gain, and noise arrays for a clean grid.
    totsampling[:] = 0.0
    Gsampling[:] = 0.0
    noisemap[:] = 0.0
    # set the pixsize in the UV plane
    UVpixsize = 2.0 / (imsize * np.pi / 180.0 / 3600.0)
    baseline_phases = {}
    bas2change = bas2change
    for nb in bas2change:
        # Calculate the pixel coordinates (pixU, pixV) in the UV plane
        # for each visibility sample of the current baseline.
        # Rounding to the nearest integer determines the UV pixel location
        pixU = np.rint(u[nb] / UVpixsize).flatten().astype(np.int32)
        pixV = np.rint(v[nb] / UVpixsize).flatten().astype(np.int32)
        # Filter out visibility samples that fall outside the field of view
        # (half-plane).
        goodpix = np.where(np.logical_and(np.abs(pixU) < Nphf, np.abs(pixV) < Nphf))[0]
        # added to introduce Atmospheric Errors
        phase_nb = np.angle(Gains[nb, goodpix])
        baseline_phases[nb] = phase_nb
        phase_rms = np.std(baseline_phases[nb])  # Standard deviation is RMS for phases
        random_phase_error = np.random.normal(scale=phase_rms)
        Gains[nb] *= np.exp(1j * random_phase_error)
        # Calculate positive and negative pixel indices
        # (accounting for the shift in the FFT).
        # Isolates positives and negative pixels
        pU = pixU[goodpix] + Nphf
        pV = pixV[goodpix] + Nphf
        mU = -pixU[goodpix] + Nphf
        mV = -pixV[goodpix] + Nphf
        # updated pixel positions for current baseline
        pixpos[nb] = [np.copy(pU), np.copy(pV), np.copy(mU), np.copy(mV)]
        # Iterate over the good pixels for the current baseline and update:
        for pi, gp in enumerate(goodpix):
            # computes the absolute gains for the current baseline
            gabs = np.abs(Gains[nb, gp])
            pVi = pV[pi]
            mUi = mU[pi]
            mVi = mV[pi]
            pUi = pU[pi]
            # Update the sampling counts, gains, and noise at the corresponding pixel
            # locations in the UV grid.
            totsampling[pVi, mUi] += 1.0
            totsampling[mVi, pUi] += 1.0
            Gsampling[pVi, mUi] += Gains[nb, gp]
            Gsampling[mVi, pUi] += np.conjugate(Gains[nb, gp])
            noisemap[pVi, mUi] += Noise[nb, gp] * gabs
            noisemap[mVi, pUi] += np.conjugate(Noise[nb, gp]) * gabs
    # Calculate a robustness factor based on the total sampling and a
    # user-defined parameter.
    robfac = (
        (5.0 * 10.0 ** (-robust)) ** 2.0 * (2.0 * Nbas * nH) / np.sum(totsampling**2.0)
    )
    return (
        pixpos,
        totsampling,
        Gsampling,
        noisemap,
        UVpixsize,
        baseline_phases,
        bas2change,
        robfac,
    )


def set_beam(
    robfac,
    totsampling,
    robustsamp,
    Gsampling,
    GrobustNoise,
    Grobustsamp,
    noisemap,
    beam,
    Nphf,
):
    # 1. Robust Weighting Calculation:
    #   - denom: Denominator used for robust weighting to balance data
    #      points with varying noise and sampling density.
    denom = 1.0 + robfac * totsampling
    # 2. Apply Robust Weighting to Sampling, Gains, and Noise:
    #   - robustsamp: Weighted sampling distribution in the UV plane.
    robustsamp[:] = totsampling / denom
    Grobustsamp[:] = Gsampling / denom
    GrobustNoise[:] = noisemap / denom
    # 3. Dirty Beam Calculation:
    #   - np.fft.fftshift(robustsamp): Shift the zero-frequency component
    #       of the weighted sampling to the center for FFT.
    #   - np.fft.ifft2(...): Perform the 2D inverse Fourier Transform to
    #       get the dirty beam in the image domain.
    #   - np.fft.ifftshift(...): Shift the zero-frequency component back
    #       to the original corner.
    #   - .real: Extract the real part of the complex result, as the beam
    #       is a real-valued function.
    #   -  / (1. + W2W1): Normalize the beam by a factor determined by `W2W1`.
    beam[:] = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(robustsamp))).real
    # 4. Beam Scaling and Normalization:
    #   - Find the maximum value of the beam within a central region
    #       (likely to avoid edge effects).
    beamScale = np.max(beam[Nphf : Nphf + 1, Nphf : Nphf + 1])
    beam[:] /= beamScale
    return beam, beamScale, robustsamp, GrobustNoise, Grobustsamp


def check_lfac(Xmax, wavelength, lfac):
    mw = 2.0 * Xmax / wavelength[2] / lfac
    if mw < 0.1 and lfac == 1.0e6:
        lfac = 1.0e3
        ulab = r"U (k$\lambda$)"
        vlab = r"V (k$\lambda$)"
    elif mw >= 100.0 and lfac == 1.0e3:
        lfac = 1.0e6
        ulab = r"U (M$\lambda$)"
        vlab = r"V (M$\lambda$)"
    return lfac, ulab, vlab


def _prepare_model(Npix, img, Nphf, Np4, zooming):
    modelim = [np.zeros((Npix, Npix), dtype=np.float32) for i in [0, 1]]
    modelimTrue = np.zeros((Npix, Npix), dtype=np.float32)
    dims = np.shape(img)
    d1 = img.shape[0]
    if d1 == Nphf:
        sh0 = (Nphf - dims[0]) // 2
        sh1 = (Nphf - dims[1]) // 2
        modelimTrue[
            sh0 + Np4 : sh0 + Np4 + dims[0],
            sh1 + Np4 : sh1 + Np4 + dims[1],
        ] += zoomimg
    else:
        zoomimg = zoom(img, float(Nphf) / d1)
        zdims = np.shape(zoomimg)
        zd0 = min(zdims[0], Nphf)
        zd1 = min(zdims[1], Nphf)
        sh0 = (Nphf - zdims[0]) // 2
        sh1 = (Nphf - zdims[1]) // 2
        modelimTrue[
            sh0 + Np4 : sh0 + Np4 + zd0,
            sh1 + Np4 : sh1 + Np4 + zd1,
        ] += zoomimg[:zd0, :zd1]
    modelimTrue[modelimTrue < 0.0] = 0.0
    return modelim, modelimTrue


def add_therman_noise(img, noise):
    return img + np.random.normal(loc=0.0, scale=noise, size=np.shape(img))


def set_primary_beam(header, distmat, wavelength, Diameters, modelim, modelimTrue):
    PB = (
        2.0
        * (1220.0 * 180.0 / np.pi * 3600.0 * wavelength[2] / Diameters[0] / 2.3548)
        ** 2.0
    )
    header.append(
        ("BMAJ", 180.0 * np.sqrt(PB) / np.pi, "Beam FWHM along major axis [deg]")
    )
    header.append(
        ("BMIN", 180.0 * np.sqrt(PB) / np.pi, "Beam FWHM along minor axis [deg]")
    )
    header.append(("BPA", 0.0, "Beam position angle [deg]"))
    # 2. Create Primary Beam Image:
    #   - Creates a 2D Gaussian image (`beamImg`) representing the primary beam.
    #   - distmat: Pre-calculated matrix of squared
    #           distances from the image center.
    #   - np.exp(distmat / PB): Calculates
    #       the Gaussian profile of the beam based on
    #            the distances and primary beam width.
    beamImg = np.exp(distmat / PB)
    # 3. Apply Primary Beam to Model:
    #   - Multiplies the original model image (`modelimTrue`)
    #       by the primary beam image (`beamImg`).
    #   - This effectively attenuates the model image towards the edges,
    #       simulating the telescope's sensitivity pattern.
    #   - The result is stored in the first element of the `modelim` list
    modelim[0][:] = modelimTrue * beamImg
    # 4. Calculate Model FFT:
    #   - Computes the 2D Fast Fourier Transform (FFT) of the primary
    #       beam-corrected model image.
    #   - np.fft.fftshift(...): Shifts the zero-frequency component to the
    #  center of the array before the FFT, as required for correct FFT interpretation.
    modelfft = np.fft.fft2(np.fft.fftshift(modelim[0]))
    return modelfft, modelim


def observe(dirtymap, GrobustNoise, modelfft, Grobustsamp, beamScale):
    # 1. Calculate Dirty Map:
    #   - np.fft.ifftshift(GrobustNoise), np.fft.ifftshift(Grobustsamp):
    #  Shift the zero-frequency components to the corners before inverse FFT.
    #   - modelfft * np.fft.ifftshift(Grobustsamp): Element-wise
    #       multiplication of the model FFT and the shifted weighted sampling to
    #       incorporate the effect of the instrument.
    #   - ... + modelfft * ... : Add the complex noise to the model's
    #            visibility after scaling by the robust sampling
    #           to obtain the observed visibilities.
    #   - np.fft.ifft2(...): Perform the 2D inverse Fast Fourier Transform (IFFT)
    #           on the combined visibilities (shifted noise + weighted model).
    #   - np.fft.fftshift(...): Shift the zero-frequency component back to the center.
    #   - .real: Extract the real part of the IFFT result to get the dirty map, which
    #       is a real-valued image.
    #   - / (1. +
    #  weighting scheme (`W2W1`).
    dirtymap[:] = (
        np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(GrobustNoise)
                + modelfft * np.fft.ifftshift(Grobustsamp)
            )
        )
    ).real
    # 2. Normalize Dirty Map:
    #   - Divide the dirty map by the beam scale factor (`beamScale`)
    #       calculated earlier in `_set_beam`.
    #   - This normalization ensures that the peak brightness in the dirty map
    #        is consistent with the beam's peak intensity.
    dirtymap /= beamScale
    # 3. Correct Negative Values in Dirty Map (Optional):
    #   - Find the minimum value in the dirty map.
    # min_dirty = np.min(dirtymap)
    #   - If there are negative values, shift the whole dirty map
    #        upwards to make all values non-negative.
    #   - This step might be necessary to avoid issues with certain
    #            image display or processing algorithms.
    # if min_dirty < 0.0:
    #    dirtymap += np.abs(min_dirty)
    # else:
    #    dirtymap -= min_dirty
    # 4. Calculate Model and Dirty Visibilities:
    #   - modelvis: Shift the zero-frequency component of the
    #        model's Fourier transform to the center.
    modelvis = np.fft.fftshift(modelfft)  # Already calculated in _set_primary_beam
    #   - dirtyvis: Shift the zero-frequency component of the dirty
    #         visibilities (shifted noise + weighted model) to the center.
    dirtyvis = np.fft.fftshift(
        np.fft.ifftshift(GrobustNoise) + modelfft * np.fft.ifftshift(Grobustsamp)
    )
    return dirtymap, modelvis, dirtyvis


@delayed
def image_channel(
    img,
    wavelength,
    Npix,
    Nant,
    Hcov,
    nH,
    noise,
    antPos,
    robfac,
    trlat,
    trdec,
    Diameters,
    imsize,
    Xmax,
    lfac,
    distmat,
    Nphf,
    Np4,
    zooming,
    header,
):
    (
        beam,
        totsampling,
        dirtymap,
        noisemap,
        robustsamp,
        Gsampling,
        Grobustsamp,
        GrobustNoise,
    ) = prepare_2d_arrays(Npix)
    Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims = (
        prepare_baselines(Nant, nH, Hcov)
    )
    Noise = set_noise(noise, Noise)
    B, u, v = set_baselines(Nbas, antnum, B, u, v, antPos, trlat, trdec, H, wavelength)
    (
        pixpos,
        totsampling,
        Gsampling,
        noisemap,
        UVpixsize,
        baseline_phases,
        bas2change,
        robfac,
    ) = _grid_uv(
        Nbas,
        totsampling,
        Gsampling,
        noisemap,
        u,
        v,
        Nphf,
        Gains,
        Noise,
        robust,
        nH,
        imsize,
    )
    beam, beamScale, robustsamp, GrobustNoise, Grobustsamp = set_beam(
        robfac,
        totsampling,
        robustsamp,
        Gsampling,
        GrobustNoise,
        Grobustsamp,
        noisemap,
        beam,
        Nphf,
    )
    lfac, ulab, vlab = check_lfac(Xmax, wavelength, lfac)
    img = add_therman_noise(img, noise)
    modelim, modelimTrue = _prepare_model(Npix, img, Nphf, Np4, zooming)
    modelfft, modelim = set_primary_beam(
        header, distmat, wavelength, Diameters, modelim, modelimTrue
    )
    dirtymap, modelvis, dirtyvis = observe(
        dirtymap, GrobustNoise, modelfft, Grobustsamp, beamScale
    )
    del noisemap, robustsamp, Gsampling, Grobustsamp, GrobustNoise
    del Nbas, B, basnum, basidx, antnum, Gains, Noise, H, ravelDims
    del pixpos, baseline_phases, bas2change
    return modelim[0], dirtymap, modelvis, dirtyvis, u, v, beam, totsampling


class Interferometer(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(
        self,
        idx,
        skymodel,
        client,
        main_dir,
        output_dir,
        ra,
        dec,
        central_freq,
        bandwidth,
        fov,
        antenna_array,
        noise,
        snr,
        integration_time,
        observation_date,
        header,
        save_mode,
        robust,
        terminal,
    ):
        super(Interferometer, self).__init__()
        print('Hello')
        self.idx = idx
        self.skymodel = skymodel
        self.client = client
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        self.ra = ra
        self.dec = dec
        self.central_freq = central_freq
        self.bandwidth = bandwidth
        self.fov = fov
        self.antenna_array = antenna_array
        self.noise = noise
        self.snr = snr
        self.integration_time = integration_time
        self.observation_date = observation_date
        self.header = header
        self.save_mode = save_mode
        self.robust = robust
        self.terminal = terminal
        # Initialize variables
        self._init_variables()
        # Get the observing location
        self._get_observing_location()
        # Ger Coverage and Antennas
        self._get_Hcov()
        self._read_antennas()
        # Get the observing wavelengths for each channel
        self._get_wavelengths()
        msg = f"Performing {self.nH} scans with a scan time of {self.scan_time} seconds"
        print('Hello2')
        if self.terminal is not None:
            self.terminal.add_log(msg)
        else:
            print(msg)

    def _hz_to_m(self, freq):
        return self.c_ms / freq

    def _init_variables(self):
        self.Hfac = np.pi / 180.0 * 15.0
        self.deg2rad = np.pi / 180.0
        self.rad2deg = 180.0 / np.pi
        self.deg2arcsec = 3600.0
        self.arcsec2deg = 1.0 / 3600.0
        self.second2hour = 1.0 / 3600.0
        self.curzoom = [0, 0, 0, 0]
        self.deltaAng = 1.0 * self.deg2rad
        self.gamma = 0.5
        self.lfac = 1.0e6
        self._get_nH()
        self.Hmax = np.pi
        self.lat = -23.028 * self.deg2rad
        self.trlat = [np.sin(self.lat), np.cos(self.lat)]
        self.Diameters = [12.0, 0]
        self.ra = self.ra.value * self.deg2rad
        self.dec = self.dec.value * self.deg2rad
        self.trdec = [np.sin(self.dec), np.cos(self.dec)]
        self.central_freq = self.central_freq.to(U.Hz).value
        self.bandwidth = self.bandwidth.to(U.Hz).value
        self.imsize = 3 * self.fov
        self.Npix = self.skymodel.shape[1]
        self.Nchan = self.skymodel.shape[0]
        self.Np4 = self.Npix // 4
        self.Nphf = self.Npix // 2
        self.pixsize = self.imsize / self.Npix
        self.UVpixsize = 2.0 / (self.imsize * np.pi / 180.0 / 3600.0)
        self.Xaxmax = self.imsize / 2
        self.c_ms = c.to(U.m / U.s).value
        self.xx = np.linspace(-self.Xaxmax, self.Xaxmax, self.Npix)
        self.yy = np.ones(self.Npix, dtype=np.float32)
        self.distmat = (
            -np.outer(self.xx**2.0, self.yy) - np.outer(self.yy, self.xx**2.0)
        ) * self.pixsize**2.0
        self.robfac = 0.0
        self.currcmap = cm.jet
        self.zooming = 0
        self.terminal.add_log("Number of Epochs", self.nH)

    def _get_observing_location(self):
        self.observing_location = EarthLocation.of_site("ALMA")

    def _get_Hcov(self):
        self.integration_time = self.integration_time * U.s
        start_time = Time(
            self.observation_date + "T00:00:00", format="isot", scale="utc"
        )
        middle_time = start_time + self.integration_time / 2
        end_time = start_time + self.integration_time
        ha_start = self._get_hour_angle(start_time)
        ha_middle = self._get_hour_angle(middle_time)
        ha_end = self._get_hour_angle(end_time)
        start = ha_start - ha_middle
        end = ha_end - ha_middle
        self.Hcov = [start, end]

    def _get_hour_angle(self, time):
        lst = time.sidereal_time("apparent", longitude=self.observing_location.lon)
        ha = lst.deg - self.ra
        if ha < 0:
            ha += 360
        return ha

    def _get_az_el(self):
        self._get_observing_location()
        self._get_middle_time()
        sky_coords = SkyCoord(
            ra=self.ra * self.rad2deg, dec=self.dec * self.rad2deg, unit="deg"
        )
        aa = AltAz(location=self.observing_location, obstime=self.middle_time)
        sky_coords.transform_to(aa)
        self.az = sky_coords.az
        self.el = sky_coords.alt

    def _get_nH(self):
        self.scan_time = 6
        self.nH = int(self.integration_time / (self.scan_time * self.second2hour))
        if self.nH > 200:
            while self.nH > 200:
                self.scan_time *= 1.5
                self.nH = int(
                    self.integration_time / (self.scan_time * self.second2hour)
                )
        self.header.append(("EPOCH", self.nH))

    def _read_antennas(self):
        antenna_coordinates = pd.read_csv(
            os.path.join(self.main_dir, "antenna_config", "antenna_coordinates.csv")
        )
        obs_antennas = self.antenna_array.split(" ")
        obs_antennas = [antenna.split(":")[0] for antenna in obs_antennas]
        obs_coordinates = antenna_coordinates[
            antenna_coordinates["name"].isin(obs_antennas)
        ]
        # Read Antenna coordinates from the antenna array
        antenna_coordinates = obs_coordinates[["x", "y"]].values
        antPos = []
        Xmax = 0.0
        for line in antenna_coordinates:
            # Convert them in meters
            antPos.append([line[0] * 1e-3, line[1] * 1e-3])
            # Get the maximum distance between any two antennas to be used in the
            # covariance matrix
            Xmax = np.max(np.abs(antPos[-1] + [Xmax]))
        self.Xmax = Xmax
        self.antPos = antPos
        # Computes the sine of the difference between lat and dec and checks that
        # is less then 1 which means that the angle of observation is valid
        cosW = -np.tan(self.lat) * np.tan(self.dec)
        if np.abs(cosW) < 1.0:
            Hhor = np.arccos(cosW)
        # if the difference
        elif np.abs(self.lat - self.dec) > np.pi / 2.0:
            Hhor = 0
        else:
            Hhor = np.pi

        if Hhor > 0.0:
            if self.Hcov[0] < -Hhor:
                self.Hcov[0] = -Hhor
            if self.Hcov[1] > Hhor:
                self.Hcov[1] = Hhor

        self.Hmax = Hhor
        self.Xmax = self.Xmax * 1.5
        self.Nant = len(self.antPos)

    def _get_wavelengths(self):
        self.w_max, self.w_min = [
            self._hz_to_m(freq)
            for freq in [
                self.central_freq - self.bandwidth / 2,
                self.central_freq + self.bandwidth / 2,
            ]
        ]
        waves = np.linspace(self.w_min, self.w_max, self.Nchan + 1)
        obs_wavelengths = np.array(
            [[waves[i], waves[i + 1]] for i in range(len(waves) - 1)]
        )
        self.obs_wavelengths = obs_wavelengths

    def get_channel_wavelength(self, channel):
        wavelength = list(self.obs_wavelengths[channel] * 1e-3)
        wavelength.append((wavelength[0] + wavelength[1]) / 2.0)
        fmtB1 = r"$\lambda = $ %4.1fmm  " % (wavelength[2] * 1.0e6)
        fmtB = (
            fmtB1
            + "\n"
            + r"% 4.2f Jy/beam"
            + "\n"
            + r"$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f "
        )
        return wavelength, fmtB

    def run_interferometric_sim(self):

        print(f"Dask dashboard link: {client.dashboard_link}")
        # Scatter input data to workers
        scattered_channels = [
            self.client.scatter(self.skymodel[i]) for i in range(self.skymodel.shape[0])
        ]
        modelcube = []
        modelvis = []
        dirtycube = []
        dirtyvis = []
        u = []
        v = []
        beam = []
        totsampling = []
        for i in range(self.Nchan):
            wavelength, _ = self.get_channel_wavelength(i)
            delayed_result = image_channel(
                scattered_channels[i],
                wavelength,
                self.Npix,
                self.Nant,
                self.Hcov,
                self.nH,
                self.noise,
                self.antPos,
                self.robfac,
                self.trlat,
                self.trdec,
                self.Diameters,
                self.imsize,
                self.Xmax,
                self.lfac,
                self.distmat,
                self.Nphf,
                self.Np4,
                self.zooming,
                self.header,
            )
            modelcube.append(delayed_result[0])
            modelvis.append(delayed_result[1])
            dirtycube.append(delayed_result[2])
            dirtyvis.append(delayed_result[3])
            u.append(delayed_result[4])
            v.append(delayed_result[5])
            beam.append(delayed_result[6])
            totsampling.append(delayed_result[7])
        # Stack the results for each array back into 3D arrays
        delayed_model = delayed(np.stack)(modelcube, axis=0)
        delayed_modelvis = delayed(np.stack)(modelvis, axis=0)
        delayed_dirty = delayed(np.stack)(dirtycube, axis=0)
        delayed_dirtyvis = delayed(np.stack)(dirtyvis, axis=0)
        delayed_u = delayed(np.stack)(u, axis=0)
        delayed_v = delayed(np.stack)(v, axis=0)
        delayed_beam = delayed(np.stack)(beam, axis=0)
        delayed_totsampling = delayed(np.stack)(totsampling, axis=0)
        futures = self.client.compute(
            [
                delayed_model,
                delayed_modelvis,
                delayed_dirty,
                delayed_dirtyvis,
                delayed_u,
                delayed_v,
                delayed_beam,
                delayed_totsampling,
            ]
        )
        self.track_progress(futures)  # Start tracking progress
        # Final result gathering after completion
        modelCube, visCube, dirtyCube, dirtyvisCube, u, v, beam, totsampling = (
            self.client.gather(futures)
        )
        # self._savez_compressed_cubes(modelCube, visCube,
        #                            dirtyCube, dirtyvisCube)
        self.s_wavelength, self.s_fmtB = self.get_channel_wavelength(self.Nchan // 2)
        simulation_results = {
            "model_cube": modelCube,
            "model_vis": visCube,
            "dirty_cube": dirtyCube,
            "dirty_vis": dirtyvisCube,
            "beam": beam[self.Nchan // 2],
            "totsampling": totsampling[self.Nchan // 2],
            "u": u[self.Nchan // 2],
            "v": v[self.Nchan // 2],
            "Npix": self.Npix,
            "Np4": self.Np4,
            "Nchan": self.Nchan,
            "gamma": self.gamma,
            "currcmap": self.currcmap,
            "Xaxmax": self.Xaxmax,
            "lfac": self.lfac,
            "UVpixsize": self.UVpixsize,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "plot_dir": self.plot_dir,
            "idx": self.idx,
            "fmtB": self.s_fmtB,
            "wavelength": self.s_wavelength,
            "curzoom": self.curzoom,
            "Nphf": self.Nphf,
            "Xmax": self.Xmax,
            "antPos": self.antPos,
            "Nant": self.Nant,
        }
        del modelCube, visCube, dirtyCube, dirtyvisCube, u, v, beam, totsampling
        return simulation_results

    def track_progress(self, futures):
        total_tasks = len(futures)
        completed_tasks = 0
        # Track progress
        while completed_tasks < total_tasks:
            completed_tasks = sum(f.done() for f in futures)
            progress_value = int((completed_tasks / total_tasks) * 100)
            self.progress_signal.emit(progress_value)  # Emit progress signal
            time.sleep(1)  # Check progress every second

    def _savez_compressed_cubes(self, modelCube, visCube, dirtyCube, dirtyvisCube):

        if self.save_mode == "npz":
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "clean-cube_{}.npz".format(str(self.idx))
                ),
                modelCube,
            )
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "dirty-cube_{}.npz".format(str(self.idx))
                ),
                dirtyCube,
            )
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "dirty-vis-cube_{}.npz".format(str(self.idx))
                ),
                dirtyvisCube,
            )
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "clean-vis-cube_{}.npz".format(str(self.idx))
                ),
                visCube,
            )
        elif self.save_mode == "h5":
            with h5py.File(
                os.path.join(self.output_dir, "clean-cube_{}.h5".format(str(self.idx))),
                "w",
            ) as f:
                f.create_dataset("clean_cube", data=modelCube)
            with h5py.File(
                os.path.join(self.output_dir, "dirty-cube_{}.h5".format(str(self.idx))),
                "w",
            ) as f:
                f.create_dataset("dirty_cube", data=dirtyCube)
            with h5py.File(
                os.path.join(
                    self.output_dir, "dirty-vis-cube_{}.h5".format(str(self.idx))
                ),
                "w",
            ) as f:
                f.create_dataset("dirty_vis_cube", data=dirtyvisCube)
            with h5py.File(
                os.path.join(
                    self.output_dir, "clean-vis-cube_{}.h5".format(str(self.idx))
                ),
                "w",
            ) as f:
                f.create_dataset("clean_vis_cube", data=visCube)
        elif self.save_mode == "fits":
            self.clean_header = self.header
            self.clean_header.append(("DATAMAX", np.max(modelCube)))
            self.clean_header.append(("DATAMIN", np.min(modelCube)))
            hdu = fits.PrimaryHDU(header=self.clean_header, data=modelCube)
            hdu.writeto(
                os.path.join(
                    self.output_dir, "clean-cube_{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            self.dirty_header = self.header
            self.dirty_header.append(("DATAMAX", np.max(dirtyCube)))
            self.dirty_header.append(("DATAMIN", np.min(dirtyCube)))
            hdu = fits.PrimaryHDU(header=self.dirty_header, data=dirtyCube)
            hdu.writeto(
                os.path.join(
                    self.output_dir, "dirty-cube_{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            real_part = np.real(dirtyvisCube)
            imag_part = np.imag(dirtyvisCube)
            hdu_real = fits.PrimaryHDU(real_part)
            hdu_imag = fits.PrimaryHDU(imag_part)
            # hdu = fits.HDUList(hdus=[hdu_real, hdu_imag])
            hdu_real.writeto(
                os.path.join(
                    self.output_dir, "dirty-vis-cube_real{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            hdu_imag.writeto(
                os.path.join(
                    self.output_dir, "dirty-vis-cube_imag{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            real_part = np.real(visCube)
            imag_part = np.imag(visCube)
            hdu_real = fits.PrimaryHDU(real_part)
            hdu_imag = fits.PrimaryHDU(imag_part)
            hdu_real.writeto(
                os.path.join(
                    self.output_dir, "clean-vis-cube_real{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            hdu_imag.writeto(
                os.path.join(
                    self.output_dir, "clean-vis-cube_imag{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            del real_part
            del imag_part
        if self.terminal is not None:
            self.terminal.add_log(
                f"Total Flux detected in model cube: {round(np.sum(modelCube), 2)}Jy"
            )
            self.terminal.add_log(
                f"Total Flux detected in dirty cube: {round(np.sum(dirtyCube), 2)}Jy"
            )
        else:
            print(f"Total Flux detected in model cube: {round(np.sum(modelCube), 2)}Jy")
            print(f"Total Flux detected in dirty cube: {round(np.sum(dirtyCube), 2)}Jy")

"""
if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster, timeout=60, heartbeat_interval=10)
    skymodel = np.ones((256, 256, 256))
    ra = 120
    dec = 30
    central_freq = 142
    band_range = 1
    fov = 12
    idx = 0
    noise = 0.2
    snr = 3
    int_time = 10
    obs_date = "2022-01-01"
    header = fits.header.Header()
    save_mode = "npz"
    robust = 0.5
    antenna_array = "A001:DA59 A002:DA49 A003:DV23 A004:DA41 A005:DA54 A006:DA61 A008:DV08 A009:DV18 A010:DV25 A011:DV09 A013:DA48 A014:DV22 A015:DV04 A017:DV12 A018:DA52 A019:DA64 A020:DV03 A021:DA58 A023:DA51 A025:DA57 A027:DV07 A029:DV10 A030:DA65 A031:DV17 A035:DA62 A036:DV16 A037:DV19 A038:DA50 A042:DV01 A046:DA53 A047:DA55 A048:DV15 A049:DV11 A050:DV24 A052:DA47 A060:DA60 A062:DA56 A063:DA44 A064:DV13 A065:DV06 A067:DA42 A068:DA45 A069:DV14 A070:DV21 A071:DA43 T701:PM03 T702:PM02 T703:PM04"
    interferometer = Interferometer(
        idx=idx,
        client=client,
        skymodel=skymodel,
        main_dir=os.curdir,
        output_dir=os.curdir,
        ra=ra * U.deg,
        dec=dec * U.deg,
        central_freq=central_freq * U.MHz,
        bandwidth=band_range * U.MHz,
        fov=fov,
        antenna_array=antenna_array,
        noise=noise,
        snr=snr,
        integration_time=int_time,
        observation_date=obs_date,
        header=header,
        save_mode=save_mode,
        robust=robust,
        terminal=None,
    )
    t1 = time.time()
    simulation_results = interferometer.run_interferometric_sim()
    t2 = time.time()
    print(f"Time taken: {t2-t1}")
"""