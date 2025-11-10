"""Baseline preparation and management for interferometry."""
import numpy as np


def prepare_baselines(
    Nant: int,
    nH: int,
    Hcov: list[float],
) -> tuple:
    """
    Prepare baselines for an interferometer array.

    Parameters
    ----------
    Nant : int
        Number of antennas
    nH : int
        Number of hour angle samples
    Hcov : list[float]
        Hour angle coverage [start, end]

    Returns
    -------
    tuple
        Baseline parameters including Nbas, B, basnum, basidx, antnum,
        Gains, Noise, H, u, v, ravelDims
    """
    Nbas = Nant * (Nant - 1) // 2
    NBmax = Nbas

    B = np.zeros((NBmax, nH), dtype=np.float32)
    basnum = np.zeros((Nant, Nant - 1), dtype=np.int8)
    basidx = np.zeros((Nant, Nant), dtype=np.int8)
    antnum = np.zeros((NBmax, 2), dtype=np.int8)
    Gains = np.ones((Nbas, nH), dtype=np.complex64)
    Noise = np.zeros((Nbas, nH), dtype=np.complex64)

    Horig = np.linspace(Hcov[0], Hcov[1], nH)
    H = Horig[np.newaxis, :]
    del Horig
    H = [np.sin(H), np.cos(H)]

    bi = 0
    nii = [0 for n in range(Nant)]

    for n1 in range(Nant - 1):
        for n2 in range(n1 + 1, Nant):
            basnum[n1, nii[n1]] = np.int8(bi)
            basnum[n2, nii[n2]] = np.int8(bi)
            basidx[n1, n2] = np.int8(bi)
            antnum[bi] = [n1, n2]
            nii[n1] += 1
            nii[n2] += 1
            bi += np.int16(1)

    u = np.zeros((NBmax, nH))
    v = np.zeros((NBmax, nH))
    ravelDims = (NBmax, nH)
    return Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims


def set_noise(noise: float, Noise: np.ndarray) -> np.ndarray:
    """Set noise values for baselines."""
    Noise[:] = np.random.normal(
        loc=0.0, scale=noise, size=np.shape(Noise)
    ) + 1.0j * np.random.normal(loc=0.0, scale=noise, size=np.shape(Noise))
    return Noise


def set_baselines(
    Nbas: int,
    antnum: np.ndarray,
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    antPos: list,
    trlat: list[float],
    trdec: list[float],
    H: list,
    wavelength: list[float],
) -> tuple:
    """Set baseline vectors and u-v coordinates."""
    bas2change = range(Nbas)
    for currBas in bas2change:
        n1, n2 = antnum[currBas]
        B[currBas, 0] = -(antPos[n2][1] - antPos[n1][1]) * trlat[0] / wavelength[2]
        B[currBas, 1] = (antPos[n2][0] - antPos[n1][0]) / wavelength[2]
        B[currBas, 2] = (antPos[n2][1] - antPos[n1][1]) * trlat[1] / wavelength[2]
        u[currBas, :] = -(B[currBas, 0] * H[0] + B[currBas, 1] * H[1])
        v[currBas, :] = (
            -B[currBas, 0] * trdec[0] * H[1]
            + B[currBas, 1] * trdec[0] * H[0]
            + trdec[1] * B[currBas, 2]
        )
    return B, u, v


