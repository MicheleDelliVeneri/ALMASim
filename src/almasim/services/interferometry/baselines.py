"""Baseline preparation and management for interferometry."""
from typing import Tuple
import numpy as np


def prepare_baselines(
    Nant: int,
    nH: int,
    Hcov: list[float],
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, Tuple[int, int]]:
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
        (Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims)
        Baseline parameters
    """
    if Nant < 2:
        raise ValueError("Number of antennas must be at least 2")
    if nH < 1:
        raise ValueError("Number of hour angle samples must be at least 1")
    if len(Hcov) < 2:
        raise ValueError("Hour angle coverage must have [start, end]")
    
    Nbas = Nant * (Nant - 1) // 2
    NBmax = Nbas

    # B stores 3D baseline vector components (x, y, z) for each baseline
    B = np.zeros((NBmax, 3), dtype=np.float32)
    # Use int16 to support more antennas (int8 max is 127 baselines)
    basnum = np.zeros((Nant, Nant - 1), dtype=np.int16)
    basidx = np.zeros((Nant, Nant), dtype=np.int16)
    antnum = np.zeros((NBmax, 2), dtype=np.int16)
    Gains = np.ones((Nbas, nH), dtype=np.complex64)
    Noise = np.zeros((Nbas, nH), dtype=np.complex64)

    # Calculate hour angles
    H_angles = np.linspace(Hcov[0], Hcov[1], nH)
    H = [np.sin(H_angles), np.cos(H_angles)]

    # Generate baseline pairs
    bi = 0
    nii = np.zeros(Nant, dtype=np.int16)

    for n1 in range(Nant - 1):
        for n2 in range(n1 + 1, Nant):
            basnum[n1, nii[n1]] = bi
            basnum[n2, nii[n2]] = bi
            basidx[n1, n2] = bi
            antnum[bi] = [n1, n2]
            nii[n1] += 1
            nii[n2] += 1
            bi += 1

    u = np.zeros((NBmax, nH))
    v = np.zeros((NBmax, nH))
    ravelDims = (NBmax, nH)
    return Nbas, B, basnum, basidx, antnum, Gains, Noise, H, u, v, ravelDims


def set_noise(noise: float, Noise: np.ndarray) -> np.ndarray:
    """
    Set noise values for baselines.
    
    Parameters
    ----------
    noise : float
        Noise standard deviation
    Noise : np.ndarray
        Array to fill with noise values (complex)
        
    Returns
    -------
    np.ndarray
        Noise array with random complex noise
    """
    if noise < 0:
        raise ValueError("Noise must be non-negative")
    shape = Noise.shape
    Noise[:] = (
        np.random.normal(loc=0.0, scale=noise, size=shape)
        + 1.0j * np.random.normal(loc=0.0, scale=noise, size=shape)
    )
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set baseline vectors and u-v coordinates.
    
    Parameters
    ----------
    Nbas : int
        Number of baselines
    antnum : np.ndarray
        Antenna number pairs for each baseline
    B : np.ndarray
        Baseline vector array (output)
    u, v : np.ndarray
        UV coordinates (output)
    antPos : list
        Antenna positions [[x0, y0], [x1, y1], ...]
    trlat : list[float]
        Transformation latitude [sin, cos]
    trdec : list[float]
        Transformation declination [sin, cos]
    H : list
        Hour angle [sin(H), cos(H)]
    wavelength : list[float]
        Wavelengths [min, max, mean]
        
    Returns
    -------
    tuple
        (B, u, v) baseline vectors and UV coordinates
    """
    if len(wavelength) < 3:
        raise ValueError("wavelength must have at least 3 elements")
    if len(trlat) < 2 or len(trdec) < 2:
        raise ValueError("trlat and trdec must have at least 2 elements")
    
    mean_wavelength = wavelength[2]
    
    for currBas in range(Nbas):
        n1, n2 = antnum[currBas]
        
        # Calculate baseline vector components
        dx = antPos[n2][0] - antPos[n1][0]
        dy = antPos[n2][1] - antPos[n1][1]
        
        B[currBas, 0] = -dy * trlat[0] / mean_wavelength
        B[currBas, 1] = dx / mean_wavelength
        B[currBas, 2] = dy * trlat[1] / mean_wavelength
        
        # Calculate UV coordinates
        u[currBas, :] = -(B[currBas, 0] * H[0] + B[currBas, 1] * H[1])
        v[currBas, :] = (
            -B[currBas, 0] * trdec[0] * H[1]
            + B[currBas, 1] * trdec[0] * H[0]
            + trdec[1] * B[currBas, 2]
        )
    return B, u, v


