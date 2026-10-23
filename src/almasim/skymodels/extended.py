"""Extended (TNG) sky model implementation."""

import os
from typing import Any

import astropy.cosmology.units as cu
import astropy.units as U
import numba
import numpy as np
from astropy.cosmology import WMAP9
from martini import DataCube, L_coords, Martini
from martini.sources import SPHSource, TNGSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import WendlandC2Kernel

from almasim.services.interferometry.predict import ALMASIM_NTHREADS

from .datasets.tng import TNG_SIMULATION


def inserted_hi_mass(M: Martini) -> U.Quantity:
    """The HI mass in ``M``'s cube, as martini reports it after insertion."""
    cube = M.datacube
    flux = (cube._array[cube.pad_mask] * cube.px_size**2).sum((0, 1)).squeeze().to_value(U.Jy)
    widths = np.abs(np.diff(cube.velocity_channel_edges.to_value(U.km / U.s)))
    return 2.36e5 * U.Msun * M.source.distance.to_value(U.Mpc) ** 2 * np.sum(flux * widths)


def insert_source(
    terminal: Any | None,
    source: SPHSource,
    n_px: int,
    n_channels: int,
    freq_sup: U.Quantity,
    px_size: U.Quantity,
) -> Martini:
    """Insert ``source`` into a cube of ``n_px`` pixels of ``px_size``.

    martini threads the insertion with numba over ``ALMASIM_NTHREADS`` CPUs, at most
    the CPUs this process may use (numba refuses more, e.g. a node's in a Slurm job).
    """
    datacube = DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_channels,
        px_size=px_size,
        channel_width=freq_sup,
        spectral_centre=source.vsys,
        ra=source.ra,
        dec=source.dec,
    )
    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=WendlandC2Kernel(),
        spectral_model=GaussianSpectrum(sigma="thermal"),
        quiet=True,
    )
    # skip_validation: warn rather than fail where the kernel approximation is coarse.
    M.insert_source_in_cube(
        skip_validation=True, ncpu=min(ALMASIM_NTHREADS, numba.get_num_threads())
    )
    if terminal is not None:
        flux = np.sum(M.datacube._array * M.datacube.px_size**2).to(U.Jy)
        terminal.add_log(
            f"Source inserted at {source.distance:.2f}.\n"
            f"  Flux density in cube: {flux:.2e}\n"
            f"  Mass in cube: {inserted_hi_mass(M):.2e}"
            f" of {source.input_mass:.2e}"
        )
    return M


def insert_tng(
    terminal: Any | None,
    n_px: int,
    n_channels: int,
    freq_sup: U.Quantity,
    snapshot: int,
    subhalo_id: int,
    distance: float,
    x_rot: U.Quantity,
    y_rot: U.Quantity,
    tngpath: str,
    ra: U.Quantity,
    dec: U.Quantity,
    api_key: str | None,
    px_size: U.Quantity,
) -> Martini:
    """Insert a TNG subhalo, inclined by ``x_rot`` and turned by ``y_rot`` about its spin."""
    source = TNGSource(
        simulation=TNG_SIMULATION,
        snapNum=snapshot,
        subID=subhalo_id,
        cutout_dir=tngpath,
        distance=distance * U.Mpc,
        L_coords=L_coords(incl=x_rot, az_rot=y_rot),
        ra=ra,
        dec=dec,
        api_key=api_key,
    )
    return insert_source(terminal, source, n_px, n_channels, freq_sup, px_size)


_MIN_TNG_DISTANCE_MPC = 1.0


class ExtendedSkyModel:
    """Extended (TNG) sky model wrapper."""

    def __init__(
        self,
        datacube: Any,
        tngpath: str,
        snapshot: int,
        subhalo_id: int,
        redshift: float,
        ra: U.Quantity,
        dec: U.Quantity,
        api_key: str | None,
        terminal: Any | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize Extended sky model.

        Parameters
        ----------
        datacube : Any
            DataCube object
        tngpath : str
            Path to TNG data directory
        snapshot : int
            TNG snapshot number
        subhalo_id : int
            TNG subhalo ID
        redshift : float
            Source redshift
        ra : U.Quantity
            Right ascension
        dec : U.Quantity
            Declination
        api_key : Optional[str]
            TNG API key
        terminal : Optional[Any]
            Terminal logger
        """
        self.datacube = datacube
        self.tngpath = tngpath
        self.snapshot = snapshot
        self.subhalo_id = subhalo_id
        self.redshift = redshift
        self.ra = ra
        self.dec = dec
        self.api_key = api_key
        self.terminal = terminal
        self.rng = rng if rng is not None else np.random.default_rng()

    def insert(self) -> Any:
        """Insert extended source into datacube."""
        x_rot = int(self.rng.integers(0, 360)) * U.deg
        y_rot = int(self.rng.integers(0, 360)) * U.deg
        tngpath = os.path.join(self.tngpath, TNG_SIMULATION, "output")
        redshift = self.redshift * cu.redshift
        computed = redshift.to(U.Mpc, cu.redshift_distance(WMAP9, kind="comoving"))
        # Start at the source's comoving distance (at least 1 Mpc, so a zero
        # redshift cannot stall the loop below that moves it further out).
        distance = max(float(computed.to_value(U.Mpc)), _MIN_TNG_DISTANCE_MPC)
        if self.terminal is not None:
            self.terminal.add_log(f"Computed a distance of {computed} for redshift {redshift}")
        M = insert_tng(
            self.terminal,
            self.datacube.n_px_x,
            self.datacube.n_channels,
            self.datacube.channel_width,
            self.snapshot,
            self.subhalo_id,
            distance,
            x_rot,
            y_rot,
            tngpath,
            self.ra,
            self.dec,
            self.api_key,
            self.datacube.px_size,
        )
        initial_mass_ratio = inserted_hi_mass(M) / M.source.input_mass * 100
        if self.terminal is not None:
            self.terminal.add_log(f"Mass ratio: {initial_mass_ratio}%")
        mass_ratio = initial_mass_ratio
        while mass_ratio < 50:
            if mass_ratio < 10:
                distance = distance * 8
            elif mass_ratio < 20:
                distance = distance * 5
            elif mass_ratio < 30:
                distance = distance * 2
            else:
                distance = distance * 1.5
            if self.terminal is not None:
                self.terminal.add_log(f"Injecting source at distance {distance}")
            M = insert_tng(
                self.terminal,
                self.datacube.n_px_x,
                self.datacube.n_channels,
                self.datacube.channel_width,
                self.snapshot,
                self.subhalo_id,
                distance,
                x_rot,
                y_rot,
                tngpath,
                self.ra,
                self.dec,
                self.api_key,
                self.datacube.px_size,
            )
            mass_ratio = inserted_hi_mass(M) / M.source.input_mass * 100
            if self.terminal is not None:
                self.terminal.add_log(f"Mass ratio: {mass_ratio}%")
        if self.terminal is not None:
            self.terminal.add_log("Datacube generated, inserting source")
        return M.datacube
