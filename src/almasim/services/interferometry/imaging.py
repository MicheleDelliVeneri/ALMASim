"""Image processing functions for interferometry."""
import numpy as np
from scipy.ndimage import zoom
from astropy.io import fits
from dask import delayed

from .baselines import prepare_baselines, set_baselines, set_noise


def prepare_2d_arrays(Npix: int) -> tuple:
    """Prepare 2D arrays for interferometric processing."""
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


def _grid_uv(
    Nbas: int,
    totsampling: np.ndarray,
    Gsampling: np.ndarray,
    noisemap: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    Nphf: int,
    Gains: np.ndarray,
    Noise: np.ndarray,
    robust: float,
    nH: int,
    imsize: float,
) -> tuple:
    """Grid visibility measurements onto a discrete UV plane."""
    bas2change = range(Nbas)
    pixpos = [[] for nb in bas2change]
    totsampling[:] = 0.0
    Gsampling[:] = 0.0
    noisemap[:] = 0.0
    UVpixsize = 2.0 / (imsize * np.pi / 180.0 / 3600.0)
    baseline_phases = {}

    for nb in bas2change:
        pixU = np.rint(u[nb] / UVpixsize).flatten().astype(np.int32)
        pixV = np.rint(v[nb] / UVpixsize).flatten().astype(np.int32)
        goodpix = np.where(np.logical_and(np.abs(pixU) < Nphf, np.abs(pixV) < Nphf))[0]

        # Add atmospheric phase errors
        phase_nb = np.angle(Gains[nb, goodpix])
        baseline_phases[nb] = phase_nb
        phase_rms = np.std(baseline_phases[nb])
        random_phase_error = np.random.normal(scale=phase_rms)
        Gains[nb] *= np.exp(1j * random_phase_error)

        pU = pixU[goodpix] + Nphf
        pV = pixV[goodpix] + Nphf
        mU = -pixU[goodpix] + Nphf
        mV = -pixV[goodpix] + Nphf
        pixpos[nb] = [np.copy(pU), np.copy(pV), np.copy(mU), np.copy(mV)]

        for pi, gp in enumerate(goodpix):
            gabs = np.abs(Gains[nb, gp])
            pVi = pV[pi]
            mUi = mU[pi]
            mVi = mV[pi]
            pUi = pU[pi]
            totsampling[pVi, mUi] += 1.0
            totsampling[mVi, pUi] += 1.0
            Gsampling[pVi, mUi] += Gains[nb, gp]
            Gsampling[mVi, pUi] += np.conjugate(Gains[nb, gp])
            noisemap[pVi, mUi] += Noise[nb, gp] * gabs
            noisemap[mVi, pUi] += np.conjugate(Noise[nb, gp]) * gabs

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
    robfac: float,
    totsampling: np.ndarray,
    robustsamp: np.ndarray,
    Gsampling: np.ndarray,
    GrobustNoise: np.ndarray,
    Grobustsamp: np.ndarray,
    noisemap: np.ndarray,
    beam: np.ndarray,
    Nphf: int,
) -> tuple:
    """Calculate dirty beam with robust weighting."""
    denom = 1.0 + robfac * totsampling
    robustsamp[:] = totsampling / denom
    Grobustsamp[:] = Gsampling / denom
    GrobustNoise[:] = noisemap / denom
    beam[:] = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(robustsamp))).real
    beamScale = np.max(beam[Nphf : Nphf + 1, Nphf : Nphf + 1])
    beam[:] /= beamScale
    return beam, beamScale, robustsamp, GrobustNoise, Grobustsamp


def check_lfac(Xmax: float, wavelength: list[float], lfac: float) -> tuple:
    """Check and adjust length factor for UV coordinates."""
    mw = 2.0 * Xmax / wavelength[2] / lfac
    if mw < 0.1 and lfac == 1.0e6:
        lfac = 1.0e3
    elif mw >= 100.0 and lfac == 1.0e3:
        lfac = 1.0e6
    ulab = r"U (k$\lambda$)"
    vlab = r"V (k$\lambda$)"
    return lfac, ulab, vlab


def _prepare_model(
    Npix: int, img: np.ndarray, Nphf: int, Np4: int, zooming: int
) -> tuple:
    """Prepare model image for processing."""
    modelim = [np.zeros((Npix, Npix), dtype=np.float32) for i in [0, 1]]
    modelimTrue = np.zeros((Npix, Npix), dtype=np.float32)
    dims = np.shape(img)
    d1 = img.shape[0]
    if d1 == Nphf:
        sh0 = (Nphf - dims[0]) // 2
        sh1 = (Nphf - dims[1]) // 2
        zoomimg = img
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


def add_thermal_noise(img: np.ndarray, noise: float) -> np.ndarray:
    """Add thermal noise to image."""
    return img + np.random.normal(loc=0.0, scale=noise, size=np.shape(img))


def set_primary_beam(
    header: fits.Header,
    distmat: np.ndarray,
    wavelength: list[float],
    Diameters: list[float],
    modelim: list,
    modelimTrue: np.ndarray,
) -> tuple:
    """Set primary beam and calculate model FFT."""
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
    beamImg = np.exp(distmat / PB)
    modelim[0][:] = modelimTrue * beamImg
    modelfft = np.fft.fft2(np.fft.fftshift(modelim[0]))
    return modelfft, modelim


def observe(
    dirtymap: np.ndarray,
    GrobustNoise: np.ndarray,
    modelfft: np.ndarray,
    Grobustsamp: np.ndarray,
    beamScale: float,
) -> tuple:
    """Observe model through interferometer and create dirty map."""
    dirtymap[:] = (
        np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(GrobustNoise)
                + modelfft * np.fft.ifftshift(Grobustsamp)
            )
        )
    ).real
    dirtymap /= beamScale
    modelvis = np.fft.fftshift(modelfft)
    dirtyvis = np.fft.fftshift(
        np.fft.ifftshift(GrobustNoise) + modelfft * np.fft.ifftshift(Grobustsamp)
    )
    return dirtymap, modelvis, dirtyvis


@delayed
def image_channel(
    img: np.ndarray,
    wavelength: list[float],
    Npix: int,
    Nant: int,
    Hcov: list[float],
    nH: int,
    noise: float,
    antPos: list,
    robfac: float,
    trlat: list[float],
    trdec: list[float],
    Diameters: list[float],
    imsize: float,
    Xmax: float,
    lfac: float,
    distmat: np.ndarray,
    Nphf: int,
    Np4: int,
    zooming: int,
    header: fits.Header,
    robust: float,
) -> tuple:
    """Process a single channel through interferometric simulation."""
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


