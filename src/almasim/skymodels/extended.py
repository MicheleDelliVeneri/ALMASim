"""Extended (TNG) sky model implementation."""

import os
import numpy as np
import astropy.units as U
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9
from typing import Optional, Any
from itertools import product
from dask import delayed
from dask.distributed import Client
from martini.sources import TNGSource
from martini import DataCube, Martini
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import WendlandC2Kernel

from .utils import track_progress


@delayed
def insert_pixel(datacube_array, insertion_slice, insertion_data):
    """Insert the spectrum for a single pixel into the datacube array."""
    datacube_array[insertion_slice] = insertion_data
    return


@delayed
def evaluate_pixel_spectrum(
    ranks_and_ij_pxs,
    pixcoords,
    kernel_sm_ranges,
    kernel_px_weights,
    datacube_stokes_axis,
    spectral_model_spectra,
):
    """Add up contributions of particles to the spectrum in a pixel."""
    result = list()
    rank, ij_pxs = ranks_and_ij_pxs
    for i, ij_px in enumerate(ij_pxs):
        ij = np.array(ij_px)[..., np.newaxis] * U.pix
        mask = (np.abs(ij - pixcoords[:2]) <= kernel_sm_ranges).all(axis=0)
        weights = kernel_px_weights(pixcoords[:2, mask] - ij, mask=mask)
        insertion_slice = (
            np.s_[ij_px[0], ij_px[1], :, 0]
            if datacube_stokes_axis
            else np.s_[ij_px[0], ij_px[1], :]
        )
        result.append(
            (
                insertion_slice,
                (spectral_model_spectra[mask] * weights[..., np.newaxis]).sum(axis=-2),
            )
        )
    return result


class MartiniMod(Martini):
    """Modified Martini class with parallel processing support."""

    def _insert_source_in_cube(
        self,
        client: Client,
        update_progress: Optional[Any] = None,
        terminal: Optional[Any] = None,
    ):
        """Insert source into cube using parallel processing."""
        assert self.spectral_model.spectra is not None
        self.sph_kernel._confirm_validation(noraise=True, quiet=True)

        # Scatter the datacube array across the workers
        scattered_array = client.scatter(self._datacube._array, broadcast=True)

        ij_pxs = list(
            product(
                np.arange(self._datacube._array.shape[0]),
                np.arange(self._datacube._array.shape[1]),
            )
        )

        # Parallel execution with Dask (let Dask decide how to distribute the tasks)
        delayed_results = []
        # Split the pixel grid among workers
        for icpu in range(len(ij_pxs)):
            # Directly call the delayed method (no need for explicit dask.delayed)
            delayed_result = evaluate_pixel_spectrum(
                (icpu, [ij_pxs[icpu]]),
                self.source.pixcoords,
                self.sph_kernel.sm_ranges,
                self.sph_kernel._px_weight,
                self._datacube.stokes_axis,
                self.spectral_model.spectra,
            )
            delayed_results.append(delayed_result)

        # Compute all the delayed tasks in parallel
        futures = client.compute(delayed_results)
        track_progress(update_progress, futures)
        # Process the results and insert into the scattered datacube array
        for result in futures:
            for insertion_slice, insertion_data in result:
                insert_pixel(scattered_array, insertion_slice, insertion_data)

        # Gather the results back to the local datacube array
        self._datacube._array = client.gather(scattered_array)

        # Final operations on the datacube
        self._datacube._array = self._datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self._datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self._datacube.padx : -self._datacube.padx,
                self._datacube.pady : -self._datacube.pady,
                ...,
            ]
            if self._datacube.padx > 0 and self._datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux_density = np.sum(
            self._datacube._array[pad_mask] * self._datacube.px_size**2
        ).to(U.Jy)
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * np.sum(
                (self._datacube._array[pad_mask] * self._datacube.px_size**2)
                .sum((0, 1))
                .squeeze()
                .to_value(U.Jy)
                * np.abs(np.diff(self._datacube.velocity_channel_edges)).to_value(
                    U.km / U.s
                )
            )
        )
        self.inserted_mass = inserted_mass
        if terminal is not None:
            terminal.add_log(
                "Source inserted.\n"
                f"  Flux density in cube: {inserted_flux_density:.2e}\n"
                f"  Mass in cube (assuming distance {self.source.distance:.2f} and a"
                f" spatially resolved source):"
                f" {inserted_mass:.2e}"
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]\n"
                f"  Maximum pixel: {self._datacube._array.max():.2e}\n"
                "  Median non-zero pixel:"
                f" {np.median(self._datacube._array[self._datacube._array > 0]):.2e}"
            )
        return


def insert_tng(
    client: Client,
    update_progress: Optional[Any],
    terminal: Optional[Any],
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
    api_key: Optional[str],
) -> MartiniMod:
    """Insert TNG source into datacube."""
    source = TNGSource(
        simulation="TNG100-1",
        snapNum=snapshot,
        subID=subhalo_id,
        cutout_dir=tngpath,
        distance=distance * U.Mpc,
        rotation={"L_coords": (x_rot, y_rot)},
        ra=ra,
        dec=dec,
        api_key=api_key,
    )

    datacube = DataCube(
        n_px_x=n_px,
        n_px_y=n_px,
        n_channels=n_channels,
        px_size=10 * U.arcsec,
        channel_width=freq_sup,
        spectral_centre=source.vsys,
        ra=source.ra,
        dec=source.dec,
    )
    spectral_model = GaussianSpectrum(sigma="thermal")
    sph_kernel = WendlandC2Kernel()
    M = MartiniMod(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
        quiet=False,
    )
    M._insert_source_in_cube(
        client,
        update_progress,
        terminal,
    )
    return M


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
        api_key: Optional[str],
        client: Client,
        update_progress: Optional[Any] = None,
        terminal: Optional[Any] = None,
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
        client : Client
            Dask client for parallel processing
        update_progress : Optional[Any]
            Progress emitter callback
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
        self.client = client
        self.update_progress = update_progress
        self.terminal = terminal

    def insert(self) -> Any:
        """Insert extended source into datacube."""
        x_rot = np.random.randint(0, 360) * U.deg
        y_rot = np.random.randint(0, 360) * U.deg
        tngpath = os.path.join(self.tngpath, "TNG100-1", "output")
        redshift = self.redshift * cu.redshift
        distance = redshift.to(U.Mpc, cu.redshift_distance(WMAP9, kind="comoving"))
        if self.terminal is not None:
            self.terminal.add_log(
                "Computed a distance of {} for redshift {}".format(distance, redshift)
            )
        distance = 50
        M = insert_tng(
            self.client,
            self.update_progress,
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
        )
        initial_mass_ratio = M.inserted_mass / M.source.input_mass * 100
        if self.terminal is not None:
            self.terminal.add_log("Mass ratio: {}%".format(initial_mass_ratio))
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
                self.terminal.add_log(
                    "Injecting source at distance {}".format(distance)
                )
            M = insert_tng(
                self.client,
                self.update_progress,
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
            )
            mass_ratio = M.inserted_mass / M.source.input_mass * 100
            if self.terminal is not None:
                self.terminal.add_log("Mass ratio: {}%".format(mass_ratio))
        if self.terminal is not None:
            self.terminal.add_log("Datacube generated, inserting source")
        return M.datacube
