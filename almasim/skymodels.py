import numpy as np
import math
import astropy.units as U
from Hdecompose.atomic_frac import atomic_frac
from .astro import loadSubset
from martini.sources.sph_source import SPHSource
from martini.spectral_models import GaussianSpectrum
from martini.sph_kernels import (
    CubicSplineKernel,
    find_fwhm,
    WendlandC2Kernel,
)
import astropy.cosmology.units as cu
from astropy.cosmology import WMAP9
from astropy import wcs
import os
from astropy.io import fits
import almasim.astro as uas
import astropy.constants as C
from itertools import product
from tqdm import tqdm
from astropy.time import Time
import matplotlib.image as plimg
from scipy.ndimage import zoom
from scipy.signal import fftconvolve
import nifty8 as ift
import random


# ----------------- Martini Modified Functions ---------------------- #


class myTNGSource(SPHSource):
    def __init__(
        self,
        snapNum,
        subID,
        basePath=None,
        distance=3.0 * U.Mpc,
        vpeculiar=0 * U.km / U.s,
        rotation={"rotmat": np.eye(3)},
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        api_key=None,
    ):
        X_H = 0.76
        full_fields_g = (
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "Density",
            "CenterOfMass",
            "GFM_Metals",
        )
        mdi_full = [None, None, None, None, None, None, 0]
        data_header = uas.loadHeader(basePath, snapNum)
        data_sub = uas.loadSingle(basePath, snapNum, subhaloID=subID)
        haloID = data_sub["SubhaloGrNr"]
        subset_g = uas.getSnapOffsets(basePath, snapNum, haloID, "Group", api_key)
        try:
            data_g = uas.loadSubset(
                basePath,
                snapNum,
                "gas",
                fields=full_fields_g,
                subset=subset_g,
                mdi=mdi_full,
                api_key=api_key,
            )

            minisnap = False
        except Exception as exc:
            print(exc.args)
            if ("Particle type" in exc.args[0]) and (
                "does not have field" in exc.args[0]
            ):
                data_g.update(
                    loadSubset(
                        basePath,
                        snapNum,
                        "gas",
                        fields=("CenterOfMass",),
                        subset=subset_g,
                        sq=False,
                        api_key=api_key,
                    )
                )
                minisnap = True
                X_H_g = X_H
            else:
                raise
        X_H_g = (
            X_H if minisnap else data_g["GFM_Metals"]
        )  # only loaded column 0: Hydrogen
        a = data_header["Time"]
        z = data_header["Redshift"]
        h = data_header["HubbleParam"]
        xe_g = data_g["ElectronAbundance"]
        rho_g = data_g["Density"] * 1e10 / h * U.Msun * np.power(a / h * U.kpc, -3)
        u_g = data_g["InternalEnergy"]  # unit conversion handled in T_g
        mu_g = 4 * C.m_p.to(U.g).value / (1 + 3 * X_H_g + 4 * X_H_g * xe_g)
        gamma = 5.0 / 3.0  # see http://www.tng-project.org/data/docs/faq/#gen4
        T_g = (gamma - 1) * u_g / C.k_B.to(U.erg / U.K).value * 1e10 * mu_g * U.K
        m_g = data_g["Masses"] * 1e10 / h * U.Msun
        # cast to float64 to avoid underflow error
        nH_g = U.Quantity(rho_g * X_H_g / mu_g, dtype=np.float64) / C.m_p
        # In TNG_corrections I set f_neutral = 1 for particles with density
        # > .1cm^-3. Might be possible to do a bit better here, but HI & H2
        # tables for TNG will be available soon anyway.
        fatomic_g = atomic_frac(
            z, nH_g, T_g, rho_g, X_H_g, onlyA1=True, TNG_corrections=True
        )
        mHI_g = m_g * X_H_g * fatomic_g
        try:
            xyz_g = data_g["CenterOfMass"] * a / h * U.kpc
        except KeyError:
            xyz_g = data_g["Coordinates"] * a / h * U.kpc
        vxyz_g = data_g["Velocities"] * np.sqrt(a) * U.km / U.s
        V_cell = (
            data_g["Masses"] / data_g["Density"] * np.power(a / h * U.kpc, 3)
        )  # Voronoi cell volume
        r_cell = np.power(3.0 * V_cell / 4.0 / np.pi, 1.0 / 3.0).to(U.kpc)
        # hsm_g has in mind a cubic spline that =0 at r=h, I think
        hsm_g = 2.5 * r_cell * find_fwhm(CubicSplineKernel().kernel)
        xyz_centre = data_sub["SubhaloPos"] * a / h * U.kpc
        xyz_g -= xyz_centre
        vxyz_centre = data_sub["SubhaloVel"] * np.sqrt(a) * U.km / U.s
        vxyz_g -= vxyz_centre
        super().__init__(
            distance=distance,
            vpeculiar=vpeculiar,
            rotation=rotation,
            ra=ra,
            dec=dec,
            h=h,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )
        return


class DataCube(object):
    """
    Handles creation and management of the data cube itself.

    Basic usage simply involves initializing with the parameters listed below.
    More advanced usage might arise if designing custom classes for other sub-
    modules, especially beams. To initialize a DataCube from a saved state, see
    DataCube.load_state.

    Parameters
    ----------
    n_px_x : int, optional
        Pixel count along the x (RA) axis. Even integers strongly preferred.
        (Default: 256.)

    n_px_y : int, optional
        Pixel count along the y (Dec) axis. Even integers strongly preferred.
        (Default: 256.)

    n_channels : int, optional
        Number of channels along the spectral axis. (Default: 64.)

    px_size : Quantity, with dimensions of angle, optional
        Angular scale of one pixel. (Default: 15 arcsec.)

    channel_width : Quantity, with dimensions of velocity or frequency, optional
        Step size along the spectral axis. Can be provided as a velocity or a
        frequency. (Default: 4 km/s.)

    velocity_centre : Quantity, with dimensions of velocity or frequency, optional
        Velocity (or frequency) of the centre along the spectral axis.
        (Default: 0 km/s.)

    ra : Quantity, with dimensions of angle, optional
        Right ascension of the cube centroid. (Default: 0 deg.)

    dec : Quantity, with dimensions of angle, optional
        Declination of the cube centroid. (Default: 0 deg.)

    stokes_axis : bool, optional
        Whether the datacube should be initialized with a Stokes' axis. (Default: False.)

    See Also
    --------
    load_state
    """

    def __init__(
        self,
        n_px_x=256,
        n_px_y=256,
        n_channels=64,
        px_size=15.0 * U.arcsec,
        channel_width=4.0 * U.km * U.s**-1,
        velocity_centre=0.0 * U.km * U.s**-1,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
        stokes_axis=False,
    ):
        self.HIfreq = 1.420405751e9 * U.Hz
        self.stokes_axis = stokes_axis
        datacube_unit = U.Jy * U.pix**-2
        self._array = np.zeros((n_px_x, n_px_y, n_channels)) * datacube_unit
        if self.stokes_axis:
            self._array = self._array[..., np.newaxis]
        self.n_px_x, self.n_px_y, self.n_channels = n_px_x, n_px_y, n_channels
        self.px_size = px_size
        self.arcsec2_to_pix = (
            U.Jy * U.pix**-2,
            U.Jy * U.arcsec**-2,
            lambda x: x / self.px_size**2,
            lambda x: x * self.px_size**2,
        )
        self.velocity_centre = velocity_centre.to(
            U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.channel_width = np.abs(
            (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(self.HIfreq)
                )
                + 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                velocity_centre.to(
                    channel_width.unit, equivalencies=U.doppler_radio(self.HIfreq)
                )
                - 0.5 * channel_width
            ).to(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.ra = ra
        self.dec = dec
        self.padx = 0
        self.pady = 0
        self._freq_channel_mode = False
        self._init_wcs()
        self._channel_mids()
        self._channel_edges()

        return

    def _init_wcs(self):
        self.wcs = wcs.WCS(naxis=3)
        self.wcs.wcs.crpix = [
            self.n_px_x / 2.0 + 0.5,
            self.n_px_y / 2.0 + 0.5,
            self.n_channels / 2.0 + 0.5,
        ]
        self.units = [U.deg, U.deg, U.m / U.s]
        self.wcs.wcs.cunit = [unit.to_string("fits") for unit in self.units]
        self.wcs.wcs.cdelt = [
            -self.px_size.to_value(self.units[0]),
            self.px_size.to_value(self.units[1]),
            self.channel_width.to_value(
                self.units[2], equivalencies=U.doppler_radio(self.HIfreq)
            ),
        ]
        self.wcs.wcs.crval = [
            self.ra.to_value(self.units[0]),
            self.dec.to_value(self.units[1]),
            self.velocity_centre.to_value(
                self.units[2], equivalencies=U.doppler_radio(self.HIfreq)
            ),
        ]
        self.wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
        self.wcs.wcs.specsys = "GALACTOC"
        if self.stokes_axis:
            self.wcs = wcs.utils.add_stokes_axis_to_wcs(self.wcs, self.wcs.wcs.naxis)
        return

    def _channel_mids(self):
        """
        Calculate the centres of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels),
            np.zeros(self.n_channels),
            np.arange(self.n_channels) - 0.5,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels),)
        self.channel_mids = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def _channel_edges(self):
        """
        Calculate the edges of the channels from the coordinate system.
        """
        pixels = (
            np.zeros(self.n_channels + 1),
            np.zeros(self.n_channels + 1),
            np.arange(self.n_channels + 1) - 1,
        )
        if self.stokes_axis:
            pixels = pixels + (np.zeros(self.n_channels + 1),)
        self.channel_edges = (
            self.wcs.wcs_pix2world(
                *pixels,
                0,
            )[2]
            * self.units[2]
        )
        return

    def spatial_slices(self):
        """
        Return an iterator over the spatial 'slices' of the cube.

        Returns
        -------
        out : iterator
            Iterator over the spatial 'slices' of the cube.
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].transpose((2, 0, 1)))

    def spectra(self):
        """
        Return an iterator over the spectra (one in each spatial pixel).

        Returns
        -------
        out : iterator
            Iterator over the spectra (one in each spatial pixel).
        """
        s = np.s_[..., 0] if self.stokes_axis else np.s_[...]
        return iter(self._array[s].reshape(self.n_px_x * self.n_px_y, self.n_channels))

    def freq_channels(self):
        """
        Convert spectral axis to frequency units.
        """
        if self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = -np.abs(
            (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.Hz, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.Hz, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.wcs.wcs.ctype[2] = "FREQ"
        self.units[2] = U.Hz
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = True
        self._channel_mids()
        self._channel_edges()
        return

    def velocity_channels(self):
        """
        Convert spectral axis to velocity units.
        """
        if not self._freq_channel_mode:
            return

        self.wcs.wcs.cdelt[2] = np.abs(
            (
                (self.wcs.wcs.crval[2] - 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
            - (
                (self.wcs.wcs.crval[2] + 0.5 * self.wcs.wcs.cdelt[2]) * self.units[2]
            ).to_value(U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq))
        )
        self.wcs.wcs.crval[2] = (self.wcs.wcs.crval[2] * self.units[2]).to_value(
            U.m / U.s, equivalencies=U.doppler_radio(self.HIfreq)
        )
        self.wcs.wcs.ctype[2] = "VRAD"
        self.units[2] = U.m * U.s**-1
        self.wcs.wcs.cunit[2] = self.units[2].to_string("fits")
        self._freq_channel_mode = False
        self._channel_mids()
        self._channel_edges()
        return

    def add_pad(self, pad):
        """
        Resize the cube to add a padding region in the spatial direction.

        Accurate convolution with a beam requires a cube padded according to
        the size of the beam kernel (its representation sampled on a grid with
        the same spacing). The beam class is required to handle defining the
        size of pad required.

        Parameters
        ----------
        pad : 2-tuple (or other sequence)
            Number of pixels to add in the x (RA) and y (Dec) directions.

        See Also
        ----------
        drop_pad
        """

        if self.padx > 0 or self.pady > 0:
            raise RuntimeError("Tried to add padding to already padded datacube array.")
        tmp = self._array
        shape = (self.n_px_x + pad[0] * 2, self.n_px_y + pad[1] * 2, self.n_channels)
        if self.stokes_axis:
            shape = shape + (1,)
        self._array = np.zeros(shape)
        self._array = self._array * tmp.unit
        xregion = np.s_[pad[0] : -pad[0]] if pad[0] > 0 else np.s_[:]
        yregion = np.s_[pad[1] : -pad[1]] if pad[1] > 0 else np.s_[:]
        self._array[xregion, yregion, ...] = tmp
        extend_crpix = [pad[0], pad[1], 0]
        if self.stokes_axis:
            extend_crpix.append(0)
        self.wcs.wcs.crpix += np.array(extend_crpix)
        self.padx, self.pady = pad
        return

    def drop_pad(self):
        """
        Remove the padding added using add_pad.

        After convolution, the pad region contains meaningless information and
        can be discarded.

        See Also
        --------
        add_pad
        """

        if (self.padx == 0) and (self.pady == 0):
            return
        self._array = self._array[self.padx : -self.padx, self.pady : -self.pady, ...]
        retract_crpix = [self.padx, self.pady, 0]
        if self.stokes_axis:
            retract_crpix.append(0)
        self.wcs.wcs.crpix -= np.array(retract_crpix)
        self.padx, self.pady = 0, 0
        return

    def copy(self):
        """
        Produce a copy of the DataCube.

        May be especially useful to create multiple datacubes with differing
        intermediate steps.

        Returns
        -------
        out : DataCube
            Copy of the DataCube object.
        """
        in_freq_channel_mode = self._freq_channel_mode
        if in_freq_channel_mode:
            self.velocity_channels()
        copy = DataCube(
            self.n_px_x,
            self.n_px_y,
            self.n_channels,
            self.px_size,
            self.channel_width,
            self.velocity_centre,
            self.ra,
            self.dec,
        )
        copy.padx, copy.pady = self.padx, self.pady
        copy.wcs = self.wcs
        copy._freq_channel_mode = self._freq_channel_mode
        copy.channel_edges = self.channel_edges
        copy.channel_mids = self.channel_mids
        copy._array = self._array.copy()
        return copy

    def save_state(self, filename, overwrite=False):
        """
        Write a file from which the current DataCube state can be
        re-initialized (see DataCube.load_state). Note that h5py must be
        installed for use. NOT for outputting mock observations, for this
        see Martini.write_fits and Martini.write_hdf5.

        Parameters
        ----------
        filename : str
            File to write.

        overwrite : bool
            Whether to allow overwriting existing files (default: False).

        See Also
        --------
        load_state
        """
        import h5py

        mode = "w" if overwrite else "w-"
        with h5py.File(filename, mode=mode) as f:
            array_unit = self._array.unit
            f["_array"] = self._array.to_value(array_unit)
            f["_array"].attrs["datacube_unit"] = str(array_unit)
            f["_array"].attrs["n_px_x"] = self.n_px_x
            f["_array"].attrs["n_px_y"] = self.n_px_y
            f["_array"].attrs["n_channels"] = self.n_channels
            px_size_unit = self.px_size.unit
            f["_array"].attrs["px_size"] = self.px_size.to_value(px_size_unit)
            f["_array"].attrs["px_size_unit"] = str(px_size_unit)
            channel_width_unit = self.channel_width.unit
            f["_array"].attrs["channel_width"] = self.channel_width.to_value(
                channel_width_unit
            )
            f["_array"].attrs["channel_width_unit"] = str(channel_width_unit)
            velocity_centre_unit = self.velocity_centre.unit
            f["_array"].attrs["velocity_centre"] = self.velocity_centre.to_value(
                velocity_centre_unit
            )
            f["_array"].attrs["velocity_centre_unit"] = str(velocity_centre_unit)
            ra_unit = self.ra.unit
            f["_array"].attrs["ra"] = self.ra.to_value(ra_unit)
            f["_array"].attrs["ra_unit"] = str(ra_unit)
            dec_unit = self.dec.unit
            f["_array"].attrs["dec"] = self.dec.to_value(dec_unit)
            f["_array"].attrs["dec_unit"] = str(self.dec.unit)
            f["_array"].attrs["padx"] = self.padx
            f["_array"].attrs["pady"] = self.pady
            f["_array"].attrs["_freq_channel_mode"] = int(self._freq_channel_mode)
            f["_array"].attrs["stokes_axis"] = self.stokes_axis
        return

    @classmethod
    def load_state(cls, filename):
        """
        Initialize a DataCube from a state saved using DataCube.save_state.
        Note that h5py must be installed for use. Note that ONLY the DataCube
        state is restored, other modules and their configurations are not
        affected.

        Parameters
        ----------
        filename : str
            File to open.

        Returns
        -------
        out : martini.DataCube
            A suitably initialized DataCube object.

        See Also
        --------
        save_state
        """
        import h5py

        with h5py.File(filename, mode="r") as f:
            n_px_x = f["_array"].attrs["n_px_x"]
            n_px_y = f["_array"].attrs["n_px_y"]
            n_channels = f["_array"].attrs["n_channels"]
            px_size = f["_array"].attrs["px_size"] * U.Unit(
                f["_array"].attrs["px_size_unit"]
            )
            channel_width = f["_array"].attrs["channel_width"] * U.Unit(
                f["_array"].attrs["channel_width_unit"]
            )
            velocity_centre = f["_array"].attrs["velocity_centre"] * U.Unit(
                f["_array"].attrs["velocity_centre_unit"]
            )
            ra = f["_array"].attrs["ra"] * U.Unit(f["_array"].attrs["ra_unit"])
            dec = f["_array"].attrs["dec"] * U.Unit(f["_array"].attrs["dec_unit"])
            stokes_axis = bool(f["_array"].attrs["stokes_axis"])
            D = cls(
                n_px_x=n_px_x,
                n_px_y=n_px_y,
                n_channels=n_channels,
                px_size=px_size,
                channel_width=channel_width,
                velocity_centre=velocity_centre,
                ra=ra,
                dec=dec,
                stokes_axis=stokes_axis,
            )
            D._init_wcs()
            D.add_pad((f["_array"].attrs["padx"], f["_array"].attrs["pady"]))
            if bool(f["_array"].attrs["_freq_channel_mode"]):
                D.freq_channels()
            D._array = f["_array"] * U.Unit(f["_array"].attrs["datacube_unit"])
        return D

    def __repr__(self):
        """
        Print the contents of the data cube array itself.

        Returns
        -------
        out : str
            Text representation of the DataCube._array contents.
        """
        return self._array.__repr__()


class Martini:
    """
    Creates synthetic HI data cubes from simulation data.

    Usual use of martini involves first creating instances of classes from each
    of the required and optional sub-modules, then creating a Martini with
    these instances as arguments. The object can then be used to create
    synthetic observations, usually by calling `insert_source_in_cube`,
    (optionally) `add_noise`, (optionally) `convolve_beam` and `write_fits` in
    order.

    Parameters
    ----------
    source : an instance of a class derived from martini.source._BaseSource
        A description of the HI emitting object, including position, geometry
        and an interface to the simulation data (SPH particle masses,
        positions, etc.). Sources leveraging the simobj package for reading
        simulation data (github.com/kyleaoman/simobj) and a few test sources
        (e.g. single particle) are provided, creation of customized sources,
        for instance to leverage other interfaces to simulation data, is
        straightforward. See sub-module documentation.

    datacube : martini.DataCube instance
        A description of the datacube to create, including pixels, channels,
        sky position. See sub-module documentation.

    beam : an instance of a class derived from beams._BaseBeam, optional
        A description of the beam for the simulated telescope. Given a
        description, either mathematical or as an image, the creation of a
        custom beam is straightforward. See sub-module documentation.

    noise : an instance of a class derived from noise._BaseNoise, optional
        A description of the simulated noise. A simple Gaussian noise model is
        provided; implementation of other noise models is straightforward. See
        sub-module documentation.

    sph_kernel : an instance of a class derived from sph_kernels._BaseSPHKernel
        A description of the SPH smoothing kernel. Check simulation
        documentation for the kernel used in a particular simulation, and
        SPH kernel submodule documentation for guidance.

    spectral_model : an instance of a class derived from \
    spectral_models._BaseSpectrum
        A description of the HI line produced by a particle of given
        properties. A Dirac-delta spectrum, and both fixed-width and
        temperature-dependent Gaussian line models are provided; implementing
        other models is straightforward. See sub-module documentation.

    quiet : bool
        If True, suppress output to stdout. (Default: False)

    See Also
    --------
    martini.sources
    martini.DataCube
    martini.beams
    martini.noise
    martini.sph_kernels
    martini.spectral_models

    Examples
    --------
    More detailed examples can be found in the examples directory in the github
    distribution of the package.

    The following example illustrates basic use of martini, using a (very!)
    crude model of a gas disk. This example can be run by doing
    'from martini import demo; demo()'::

        # ------make a toy galaxy----------
        N = 500
        phi = np.random.rand(N) * 2 * np.pi
        r = []
        for L in np.random.rand(N):

            def f(r):
                return L - 0.5 * (2 - np.exp(-r) * (np.power(r, 2) + 2 * r + 2))

            r.append(fsolve(f, 1.0)[0])
        r = np.array(r)
        # exponential disk
        r *= 3 / np.sort(r)[N // 2]
        z = -np.log(np.random.rand(N))
        # exponential scale height
        z *= 0.5 / np.sort(z)[N // 2] * np.sign(np.random.rand(N) - 0.5)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        xyz_g = np.vstack((x, y, z)) * U.kpc
        # linear rotation curve
        vphi = 100 * r / 6.0
        vx = -vphi * np.sin(phi)
        vy = vphi * np.cos(phi)
        # small pure random z velocities
        vz = (np.random.rand(N) * 2.0 - 1.0) * 5
        vxyz_g = np.vstack((vx, vy, vz)) * U.km * U.s**-1
        T_g = np.ones(N) * 8e3 * U.K
        mHI_g = np.ones(N) / N * 5.0e9 * U.Msun
        # ~mean interparticle spacing smoothing
        hsm_g = np.ones(N) * 4 / np.sqrt(N) * U.kpc
        # ---------------------------------

        source = SPHSource(
            distance=3.0 * U.Mpc,
            rotation={"L_coords": (60.0 * U.deg, 0.0 * U.deg)},
            ra=0.0 * U.deg,
            dec=0.0 * U.deg,
            h=0.7,
            T_g=T_g,
            mHI_g=mHI_g,
            xyz_g=xyz_g,
            vxyz_g=vxyz_g,
            hsm_g=hsm_g,
        )

        datacube = DataCube(
            n_px_x=128,
            n_px_y=128,
            n_channels=32,
            px_size=10.0 * U.arcsec,
            channel_width=10.0 * U.km * U.s**-1,
            velocity_centre=source.vsys,
        )

        beam = GaussianBeam(
            bmaj=30.0 * U.arcsec, bmin=30.0 * U.arcsec, bpa=0.0 * U.deg, truncate=4.0
        )

        noise = GaussianNoise(rms=3.0e-5 * U.Jy * U.beam**-1)

        spectral_model = GaussianSpectrum(sigma=7 * U.km * U.s**-1)

        sph_kernel = CubicSplineKernel()

        M = Martini(
            source=source,
            datacube=datacube,
            beam=beam,
            noise=noise,
            spectral_model=spectral_model,
            sph_kernel=sph_kernel,
        )

        M.insert_source_in_cube()
        M.add_noise()
        M.convolve_beam()
        M.write_beam_fits(beamfile, channels="velocity")
        M.write_fits(cubefile, channels="velocity")
        print(f"Wrote demo fits output to {cubefile}, and beam image to {beamfile}.")
        try:
            M.write_hdf5(hdf5file, channels="velocity")
        except ModuleNotFoundError:
            print("h5py package not present, skipping hdf5 output demo.")
        else:
            print(f"Wrote demo hdf5 output to {hdf5file}.")
    """

    def __init__(
        self,
        source=None,
        datacube=None,
        beam=None,
        noise=None,
        sph_kernel=None,
        spectral_model=None,
        quiet=False,
        find_distance=False,
    ):
        self.quiet = quiet
        self.find_distance = find_distance
        if source is not None:
            self.source = source
        else:
            raise ValueError("A source instance is required.")
        if datacube is not None:
            self.datacube = datacube
        else:
            raise ValueError("A datacube instance is required.")
        self.beam = beam
        self.noise = noise
        if sph_kernel is not None:
            self.sph_kernel = sph_kernel
        else:
            raise ValueError("An SPH kernel instance is required.")
        if spectral_model is not None:
            self.spectral_model = spectral_model
        else:
            raise ValueError("A spectral model instance is required.")

        if self.beam is not None:
            self.beam.init_kernel(self.datacube)
            self.datacube.add_pad(self.beam.needs_pad())

        self.source._init_skycoords()
        self.source._init_pixcoords(self.datacube)  # after datacube is padded

        self.sph_kernel._init_sm_lengths(source=self.source, datacube=self.datacube)
        self.sph_kernel._init_sm_ranges()
        if self.find_distance is False:
            self._prune_particles()  # prunes both source, and kernel if applicable
            self.spectral_model.init_spectra(self.source, self.datacube)
            self.inserted_mass = 0

        return

    def convolve_beam(self):
        """
        Convolve the beam and DataCube.
        """

        if self.beam is None:
            print("Skipping beam convolution, no beam object provided to " "Martini.")
            return

        unit = self.datacube._array.unit
        for spatial_slice in self.datacube.spatial_slices():
            # use a view [...] to force in-place modification
            spatial_slice[...] = (
                fftconvolve(spatial_slice, self.beam.kernel, mode="same") * unit
            )
        self.datacube.drop_pad()
        self.datacube._array = self.datacube._array.to(
            U.Jy * U.beam**-1,
            equivalencies=U.beam_angular_area(self.beam.area),
        )
        if not self.quiet:
            print(
                "Beam convolved.",
                "  Data cube RMS after beam convolution:"
                f" {np.std(self.datacube._array):.2e}",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def add_noise(self):
        """
        Insert noise into the DataCube.
        """

        if self.noise is None:
            print("Skipping noise, no noise object provided to Martini.")
            return

        # this unit conversion means noise can be added before or after source insertion:
        noise_cube = (
            self.noise.generate(self.datacube, self.beam)
            .to(
                U.Jy * U.arcsec**-2,
                equivalencies=U.beam_angular_area(self.beam.area),
            )
            .to(self.datacube._array.unit, equivalencies=[self.datacube.arcsec2_to_pix])
        )
        self.datacube._array = self.datacube._array + noise_cube
        if not self.quiet:
            print(
                "Noise added.",
                f"  Noise cube RMS: {np.std(noise_cube):.2e} (before beam convolution).",
                "  Data cube RMS after noise addition (before beam convolution): "
                f"{np.std(self.datacube._array):.2e}",
                sep="\n",
            )
        return

    def _prune_particles(self):
        """
        Determines which particles cannot contribute to the DataCube and
        removes them to speed up calculation. Assumes the kernel is 0 at
        distances greater than the kernel size (which may differ from the
        SPH smoothing length).
        """

        if not self.quiet:
            print(
                f"Source module contained {self.source.npart} particles with total HI"
                f" mass of {self.source.mHI_g.sum():.2e}."
            )
        spectrum_half_width = (
            self.spectral_model.half_width(self.source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                self.source.pixcoords[:2] + self.sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            self.source.pixcoords[0] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            self.source.pixcoords[1] - self.sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            self.source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            self.source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(self.source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        self.source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        self.sph_kernel._apply_mask(np.logical_not(reject_mask))
        if not self.quiet:
            print(
                f"Pruned particles that will not contribute to data cube, "
                f"{self.source.npart} particles remaining with total HI mass of "
                f"{self.source.mHI_g.sum():.2e}."
            )
        return

    def _compute_particles_num(self):
        new_source = self.source
        new_sph_kernel = self.sph_kernel
        initial_npart = self.source.npart
        spectrum_half_width = (
            self.spectral_model.half_width(new_source) / self.datacube.channel_width
        )
        reject_conditions = (
            (
                new_source.pixcoords[:2] + new_sph_kernel.sm_ranges[np.newaxis]
                < 0 * U.pix
            ).any(axis=0),
            new_source.pixcoords[0] - new_sph_kernel.sm_ranges
            > (self.datacube.n_px_x + self.datacube.padx * 2) * U.pix,
            new_source.pixcoords[1] - new_sph_kernel.sm_ranges
            > (self.datacube.n_px_y + self.datacube.pady * 2) * U.pix,
            new_source.pixcoords[2] + 4 * spectrum_half_width * U.pix < 0 * U.pix,
            new_source.pixcoords[2] - 4 * spectrum_half_width * U.pix
            > self.datacube.n_channels * U.pix,
        )
        reject_mask = np.zeros(new_source.pixcoords[0].shape)
        for condition in reject_conditions:
            reject_mask = np.logical_or(reject_mask, condition)
        new_source.apply_mask(np.logical_not(reject_mask))
        # most kernels ignore this line, but required by AdaptiveKernel
        new_sph_kernel._apply_mask(np.logical_not(reject_mask))
        final_npart = new_source.npart
        del new_source
        del new_sph_kernel
        return final_npart / initial_npart * 100

    def _evaluate_pixel_spectrum(self, ranks_and_ij_pxs, progressbar=True):
        """
        Add up contributions of particles to the spectrum in a pixel.

        This is the core loop of MARTINI. It is embarrassingly parallel. To support
        parallel excecution we accept storing up to a copy of the entire (future) datacube
        in one-pixel pieces. This avoids the need for concurrent access to the datacube
        by parallel processes, which would in the simplest case duplicate a copy of the
        datacube array per parallel process! In realistic use cases the memory overhead
        from a the equivalent of a second datacube array should be minimal - memory-
        limited applications should be limited by the memory consumed by particle data,
        which is not duplicated in parallel execution.

        The arguments that differ between parallel ranks must be bundled into one for
        compatibility with `multiprocess`.

        Parameters
        ----------
        rank_and_ij_pxs : tuple
            A 2-tuple containing an integer (cpu "rank" in the case of parallel execution)
            and a list of 2-tuples specifying the indices (i, j) of pixels in the grid.

        Returns
        -------
        out : list
            A list containing 2-tuples. Each 2-tuple contains and "insertion slice" that
            is an index into the datacube._array instance held by this martini instance
            where the pixel spectrum is to be placed, and a 1D array containing the
            spectrum, whose length must match the length of the spectral axis of the
            datacube.
        """
        result = list()
        rank, ij_pxs = ranks_and_ij_pxs
        if progressbar:
            ij_pxs = tqdm(ij_pxs, position=rank)
        for ij_px in ij_pxs:
            ij = np.array(ij_px)[..., np.newaxis] * U.pix
            mask = (
                np.abs(ij - self.source.pixcoords[:2]) <= self.sph_kernel.sm_ranges
            ).all(axis=0)
            weights = self.sph_kernel._px_weight(
                self.source.pixcoords[:2, mask] - ij, mask=mask
            )
            insertion_slice = (
                np.s_[ij_px[0], ij_px[1], :, 0]
                if self.datacube.stokes_axis
                else np.s_[ij_px[0], ij_px[1], :]
            )
            result.append(
                (
                    insertion_slice,
                    (self.spectral_model.spectra[mask] * weights[..., np.newaxis]).sum(
                        axis=-2
                    ),
                )
            )
        return result

    def _insert_pixel(self, insertion_slice, insertion_data):
        """
        Insert the spectrum for a single pixel into the datacube array.

        Parameters
        ----------
        insertion_slice : integer, tuple or slice
            Index into the datacube's _array specifying the insertion location.
        insertion data : array-like
            1D array containing the spectrum at the location specified by insertion_slice.
        """
        self.datacube._array[insertion_slice] = insertion_data
        return

    def insert_source_in_cube(self, skip_validation=False, progressbar=None, ncpu=1):
        """
        Populates the DataCube with flux from the particles in the source.

        Parameters
        ----------
        skip_validation : bool, optional
            SPH kernel interpolation onto the DataCube is approximated for
            increased speed. For some combinations of pixel size, distance
            and SPH smoothing length, the approximation may break down. The
            kernel class will check whether this will occur and raise a
            RuntimeError if so. This validation can be skipped (at the cost
            of accuracy!) by setting this parameter True. (Default: False.)

        progressbar : bool, optional
            A progress bar is shown by default. Progress bars work, with perhaps
            some visual glitches, in parallel. If martini was initialised with
            `quiet` set to `True`, progress bars are switched off unless explicitly
            turned on. (Default: None.)

        ncpu : int
            Number of processes to use in main source insertion loop. Using more than
            one cpu requires the `multiprocess` module (n.b. not the same as
            `multiprocessing`). (Default: 1)

        """

        assert self.spectral_model.spectra is not None

        if progressbar is None:
            progressbar = not self.quiet

        self.sph_kernel._confirm_validation(noraise=skip_validation, quiet=self.quiet)

        ij_pxs = list(
            product(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
            )
        )

        if ncpu == 1:
            for insertion_slice, insertion_data in self._evaluate_pixel_spectrum(
                (0, ij_pxs), progressbar=progressbar
            ):
                self._insert_pixel(insertion_slice, insertion_data)
        else:
            # not multiprocessing, need serialization from dill not pickle
            from multiprocess import Pool

            with Pool(processes=ncpu) as pool:
                for result in pool.imap_unordered(
                    lambda x: self._evaluate_pixel_spectrum(x, progressbar=progressbar),
                    [(icpu, ij_pxs[icpu::ncpu]) for icpu in range(ncpu)],
                ):
                    for insertion_slice, insertion_data in result:
                        self._insert_pixel(insertion_slice, insertion_data)

        self.datacube._array = self.datacube._array.to(
            U.Jy / U.arcsec**2, equivalencies=[self.datacube.arcsec2_to_pix]
        )
        pad_mask = (
            np.s_[
                self.datacube.padx : -self.datacube.padx,
                self.datacube.pady : -self.datacube.pady,
                ...,
            ]
            if self.datacube.padx > 0 and self.datacube.pady > 0
            else np.s_[...]
        )
        inserted_flux = self.datacube._array[pad_mask].sum() * self.datacube.px_size**2
        inserted_mass = (
            2.36e5
            * U.Msun
            * self.source.distance.to_value(U.Mpc) ** 2
            * inserted_flux.to_value(U.Jy)
            * self.datacube.channel_width.to_value(U.km / U.s)
        )
        self.inserted_mass = inserted_mass
        if not self.quiet:
            print(
                "Source inserted.",
                f"  Flux in cube: {inserted_flux:.2e}",
                f"  Mass in cube (assuming distance {self.source.distance:.2f}):"
                f" {inserted_mass:.2e}",
                f"    [{inserted_mass / self.source.input_mass * 100:.0f}%"
                f" of initial source mass]",
                f"  Maximum pixel: {self.datacube._array.max():.2e}",
                "  Median non-zero pixel:"
                f" {np.median(self.datacube._array[self.datacube._array > 0]):.2e}",
                sep="\n",
            )
        return

    def write_fits(
        self,
        filename,
        channels="frequency",
        overwrite=True,
    ):
        """
        Output the DataCube to a FITS-format file.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)
        """

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()
        wcs_header.rename_keyword("WCSAXES", "NAXIS")

        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", self.datacube.n_px_x))
        header.append(("NAXIS2", self.datacube.n_px_y))
        header.append(("NAXIS3", self.datacube.n_channels))
        if self.datacube.stokes_axis:
            header.append(("NAXIS4", 1))
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
        if self.datacube.stokes_axis:
            header.append(("CDELT4", wcs_header["CDELT4"]))
            header.append(("CRPIX4", wcs_header["CRPIX4"]))
            header.append(("CRVAL4", wcs_header["CRVAL4"]))
            header.append(("CTYPE4", wcs_header["CTYPE4"]))
            header.append(("CUNIT4", "PAR"))
        header.append(("EPOCH", 2000))
        # header.append(('BLANK', -32768)) #only for integer data
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        datacube_array_units = self.datacube._array.unit
        header.append(
            ("DATAMAX", np.max(self.datacube._array.to_value(datacube_array_units)))
        )
        header.append(
            ("DATAMIN", np.min(self.datacube._array.to_value(datacube_array_units)))
        )
        # long names break fits format, don't let the user set this
        header.append(("OBJECT", "MOCK"))
        if self.beam is not None:
            header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("OBSERVER", "K. Oman"))
        # header.append(('NITERS', ???))
        # header.append(('RMS', ???))
        # header.append(('LWIDTH', ???))
        # header.append(('LSTEP', ???))
        header.append(("BUNIT", datacube_array_units.to_string("fits")))
        # header.append(('PCDEC', ???))
        # header.append(('LSTART', ???))
        header.append(("DATE-OBS", Time.now().to_value("fits")))
        # header.append(('LTYPE', ???))
        # header.append(('PCRA', ???))
        # header.append(('CELLSCAL', ???))
        if self.beam is not None:
            header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
            header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BTYPE", "Intensity"))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header, data=self.datacube._array.to_value(datacube_array_units).T
        )
        hdu.writeto(filename, overwrite=overwrite)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_beam_fits(self, filename, channels="frequency", overwrite=True):
        """
        Output the beam to a FITS-format file.

        The beam is written to file, with pixel sizes, coordinate system, etc.
        similar to those used for the DataCube.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.fits' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        Raises
        ------
        ValueError
            If Martini was initialized without a beam.
        """

        if self.beam is None:
            raise ValueError(
                "Martini.write_beam_fits: Called with beam set " "to 'None'."
            )
        assert self.beam.kernel is not None
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            self.datacube.velocity_channels()
        else:
            raise ValueError(
                "Martini.write_beam_fits: Unknown 'channels' "
                "value (use 'frequency' or 'velocity'."
            )

        filename = filename if filename[-5:] == ".fits" else filename + ".fits"

        wcs_header = self.datacube.wcs.to_header()

        beam_kernel_units = self.beam.kernel.unit
        header = fits.Header()
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        # header.append(('NAXIS', self.beam.kernel.ndim))
        header.append(("NAXIS", 3))
        header.append(("NAXIS1", self.beam.kernel.shape[0]))
        header.append(("NAXIS2", self.beam.kernel.shape[1]))
        header.append(("NAXIS3", 1))
        header.append(("EXTEND", "T"))
        header.append(("BSCALE", 1.0))
        header.append(("BZERO", 0.0))
        # this is 1/arcsec^2, is this right?
        header.append(("BUNIT", beam_kernel_units.to_string("fits")))
        header.append(("CRPIX1", self.beam.kernel.shape[0] // 2 + 1))
        header.append(("CDELT1", wcs_header["CDELT1"]))
        header.append(("CRVAL1", wcs_header["CRVAL1"]))
        header.append(("CTYPE1", wcs_header["CTYPE1"]))
        header.append(("CUNIT1", wcs_header["CUNIT1"]))
        header.append(("CRPIX2", self.beam.kernel.shape[1] // 2 + 1))
        header.append(("CDELT2", wcs_header["CDELT2"]))
        header.append(("CRVAL2", wcs_header["CRVAL2"]))
        header.append(("CTYPE2", wcs_header["CTYPE2"]))
        header.append(("CUNIT2", wcs_header["CUNIT2"]))
        header.append(("CRPIX3", 1))
        header.append(("CDELT3", wcs_header["CDELT3"]))
        header.append(("CRVAL3", wcs_header["CRVAL3"]))
        header.append(("CTYPE3", wcs_header["CTYPE3"]))
        header.append(("CUNIT3", wcs_header["CUNIT3"]))
        header.append(("SPECSYS", wcs_header["SPECSYS"]))
        header.append(("BMAJ", self.beam.bmaj.to_value(U.deg)))
        header.append(("BMIN", self.beam.bmin.to_value(U.deg)))
        header.append(("BPA", self.beam.bpa.to_value(U.deg)))
        header.append(("BTYPE", "beam    "))
        header.append(("EPOCH", 2000))
        header.append(("OBSERVER", "K. Oman"))
        # long names break fits format
        header.append(("OBJECT", "MOCKBEAM"))
        header.append(("DATAMAX", np.max(self.beam.kernel.to_value(beam_kernel_units))))
        header.append(("DATAMIN", np.min(self.beam.kernel.to_value(beam_kernel_units))))

        # flip axes to write
        hdu = fits.PrimaryHDU(
            header=header,
            data=self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis].T,
        )
        hdu.writeto(filename, overwrite=True)

        if channels == "frequency":
            self.datacube.velocity_channels()
        return

    def write_hdf5(
        self,
        filename,
        channels="frequency",
        overwrite=True,
        memmap=False,
        compact=False,
    ):
        """
        Output the DataCube and Beam to a HDF5-format file. Requires the h5py
        package.

        Parameters
        ----------
        filename : string
            Name of the file to write. '.hdf5' will be appended if not already
            present.

        channels : {'frequency', 'velocity'}, optional
            Type of units used along the spectral axis in output file.
            (Default: 'frequency'.)

        overwrite: bool, optional
            Whether to allow overwriting existing files. (Default: True.)

        memmap: bool, optional
            If True, create a file-like object in memory and return it instead
            of writing file to disk. (Default: False.)

        compact: bool, optional
            If True, omit pixel coordinate arrays to save disk space. In this
            case pixel coordinates can still be reconstructed from FITS-style
            keywords stored in the FluxCube attributes. (Default: False.)
        """

        import h5py

        self.datacube.drop_pad()
        if channels == "frequency":
            self.datacube.freq_channels()
        elif channels == "velocity":
            pass
        else:
            raise ValueError(
                "Martini.write_fits: Unknown 'channels' value "
                "(use 'frequency' or 'velocity')."
            )

        filename = filename if filename[-5:] == ".hdf5" else filename + ".hdf5"

        wcs_header = self.datacube.wcs.to_header()

        mode = "w" if overwrite else "x"
        driver = "core" if memmap else None
        h5_kwargs = {"backing_store": False} if memmap else dict()
        f = h5py.File(filename, mode, driver=driver, **h5_kwargs)
        datacube_array_units = self.datacube._array.unit
        s = np.s_[..., 0] if self.datacube.stokes_axis else np.s_[...]
        f["FluxCube"] = self.datacube._array.to_value(datacube_array_units)[s]
        c = f["FluxCube"]
        origin = 0  # index from 0 like numpy, not from 1
        if not compact:
            xgrid, ygrid, vgrid = np.meshgrid(
                np.arange(self.datacube._array.shape[0]),
                np.arange(self.datacube._array.shape[1]),
                np.arange(self.datacube._array.shape[2]),
            )
            cgrid = (
                np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                        np.zeros(vgrid.shape).flatten(),
                    )
                ).T
                if self.datacube.stokes_axis
                else np.vstack(
                    (
                        xgrid.flatten(),
                        ygrid.flatten(),
                        vgrid.flatten(),
                    )
                ).T
            )
            wgrid = self.datacube.wcs.all_pix2world(cgrid, origin)
            ragrid = wgrid[:, 0].reshape(self.datacube._array.shape)[s]
            decgrid = wgrid[:, 1].reshape(self.datacube._array.shape)[s]
            chgrid = wgrid[:, 2].reshape(self.datacube._array.shape)[s]
            f["RA"] = ragrid
            f["RA"].attrs["Unit"] = wcs_header["CUNIT1"]
            f["Dec"] = decgrid
            f["Dec"].attrs["Unit"] = wcs_header["CUNIT2"]
            f["channel_mids"] = chgrid
            f["channel_mids"].attrs["Unit"] = wcs_header["CUNIT3"]
        c.attrs["AxisOrder"] = "(RA,Dec,Channels)"
        c.attrs["FluxCubeUnit"] = str(self.datacube._array.unit)
        c.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
        c.attrs["RA0_in_px"] = wcs_header["CRPIX1"] - 1
        c.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
        c.attrs["RAUnit"] = wcs_header["CUNIT1"]
        c.attrs["RAProjType"] = wcs_header["CTYPE1"]
        c.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
        c.attrs["Dec0_in_px"] = wcs_header["CRPIX2"] - 1
        c.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
        c.attrs["DecUnit"] = wcs_header["CUNIT2"]
        c.attrs["DecProjType"] = wcs_header["CTYPE2"]
        c.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
        c.attrs["V0_in_px"] = wcs_header["CRPIX3"] - 1
        c.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
        c.attrs["VUnit"] = wcs_header["CUNIT3"]
        c.attrs["VProjType"] = wcs_header["CTYPE3"]
        if self.beam is not None:
            c.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            c.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            c.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
        c.attrs["DateCreated"] = str(Time.now())
        if self.beam is not None:
            if self.beam.kernel is None:
                raise ValueError(
                    "Martini.write_hdf5: Called with beam present but beam kernel"
                    " uninitialized."
                )
            beam_kernel_units = self.beam.kernel.unit
            f["Beam"] = self.beam.kernel.to_value(beam_kernel_units)[..., np.newaxis]
            b = f["Beam"]
            b.attrs["BeamUnit"] = self.beam.kernel.unit.to_string("fits")
            b.attrs["deltaRA_in_RAUnit"] = wcs_header["CDELT1"]
            b.attrs["RA0_in_px"] = self.beam.kernel.shape[0] // 2
            b.attrs["RA0_in_RAUnit"] = wcs_header["CRVAL1"]
            b.attrs["RAUnit"] = wcs_header["CUNIT1"]
            b.attrs["RAProjType"] = wcs_header["CTYPE1"]
            b.attrs["deltaDec_in_DecUnit"] = wcs_header["CDELT2"]
            b.attrs["Dec0_in_px"] = self.beam.kernel.shape[1] // 2
            b.attrs["Dec0_in_DecUnit"] = wcs_header["CRVAL2"]
            b.attrs["DecUnit"] = wcs_header["CUNIT2"]
            b.attrs["DecProjType"] = wcs_header["CTYPE2"]
            b.attrs["deltaV_in_VUnit"] = wcs_header["CDELT3"]
            b.attrs["V0_in_px"] = 0
            b.attrs["V0_in_VUnit"] = wcs_header["CRVAL3"]
            b.attrs["VUnit"] = wcs_header["CUNIT3"]
            b.attrs["VProjType"] = wcs_header["CTYPE3"]
            b.attrs["BeamPA"] = self.beam.bpa.to_value(U.deg)
            b.attrs["BeamMajor_in_deg"] = self.beam.bmaj.to_value(U.deg)
            b.attrs["BeamMinor_in_deg"] = self.beam.bmin.to_value(U.deg)
            b.attrs["DateCreated"] = str(Time.now())

        if channels == "frequency":
            self.datacube.velocity_channels()
        if memmap:
            return f
        else:
            f.close()
            return

    def reset(self):
        """
        Re-initializes the DataCube with zero-values.
        """
        init_kwargs = dict(
            n_px_x=self.datacube.n_px_x,
            n_px_y=self.datacube.n_px_y,
            n_channels=self.datacube.n_channels,
            px_size=self.datacube.px_size,
            channel_width=self.datacube.channel_width,
            velocity_centre=self.datacube.velocity_centre,
            ra=self.datacube.ra,
            dec=self.datacube.dec,
            stokes_axis=self.datacube.stokes_axis,
        )
        self.datacube = DataCube(**init_kwargs)
        if self.beam is not None:
            self.datacube.add_pad(self.beam.needs_pad())
        return


# ------------------ SkyModels ------------------------------------- #


def gaussian(x, amp, cen, fwhm):
    """
    Generates a 1D Gaussian given the following input parameters:
    x: position
    amp: amplitude
    fwhm: fwhm
    """
    # def integrand(x, amp, cen, fwhm):
    #    return np.exp(-(x-cen)**2/(2*(fwhm/2.35482)**2))
    # integral, _ = quad(integrand, -np.inf, np.inf, args=(1, cen, fwhm))
    gaussian = np.exp(-((x - cen) ** 2) / (2 * (fwhm / 2.35482) ** 2))
    if np.sum(gaussian) != 0:
        norm = amp / np.sum(gaussian)
    else:
        norm = amp
    result = norm * gaussian
    # norm = 1 / integral
    return result


def gaussian2d(x, y, amp, cen_x, cen_y, fwhm_x, fwhm_y, angle):
    """
    Generates a 2D Gaussian given the following input parameters:
    x, y: positions
    amp: amplitude
    cen_x, cen_y: centers
    fwhm_x, fwhm_y: FWHMs (full width at half maximum) along x and y axes
    angle: angle of rotation (in degrees)
    """
    angle_rad = math.radians(angle)

    # Rotate coordinates
    xp = (x - cen_x) * np.cos(angle_rad) - (y - cen_y) * np.sin(angle_rad) + cen_x
    yp = (x - cen_x) * np.sin(angle_rad) + (y - cen_y) * np.cos(angle_rad) + cen_y

    gaussian = np.exp(
        -(
            (xp - cen_x) ** 2 / (2 * (fwhm_x / 2.35482) ** 2)
            + (yp - cen_y) ** 2 / (2 * (fwhm_y / 2.35482) ** 2)
        )
    )
    norm = amp / np.sum(gaussian)

    result = norm * gaussian

    return result


def insert_pointlike(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_x,
    pos_y,
    pos_z,
    fwhm_z,
    n_chan,
):
    """
    Inserts a point source into the datacube at the specified position and amplitude.
    datacube: datacube object
    amplitude: amplitude of the point source
    pos_x: x position
    pos_y: y position
    pos_z: z position
    fwhm_z: fwhm in z
    n_px: number of pixels in the cube
    n_chan: number of channels in the cube
    """
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
        update_progress.emit(i / len(line_fluxes) * 100)
    datacube._array[
        pos_x,
        pos_y,
    ] = (
        (continum + gs) * U.Jy * U.pix**-2
    )
    return datacube


def insert_gaussian(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_x,
    pos_y,
    pos_z,
    fwhm_x,
    fwhm_y,
    fwhm_z,
    angle,
    n_px,
    n_chan,
):
    """
    Inserts a 3D Gaussian into the datacube at the specified position and amplitude.
    datacube: datacube object
    amplitude: amplitude of the source
    pos_x: x position
    pos_y: y position
    pos_z: z position
    fwhm_x: fwhm in x
    fwhm_y: fwhm in y
    fwhm_z: fwhm in z
    angle: angle of rotation
    n_px: number of pixels in the cube
    n_chan: number of channels in the cube
    """
    X, Y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cont = gaussian2d(X, Y, continum[z], pos_x, pos_y, fwhm_x, fwhm_y, angle)
        line = gaussian2d(X, Y, gs[z], pos_x, pos_y, fwhm_x, fwhm_y, angle)
        slice_ = cont + line
        datacube._array[:, :, z] += slice_ * U.Jy * U.pix**-2
        update_progress.emit(z / n_chan * 100)
    return datacube


def interpolate_array(arr, n_px):
    """Interpolates a 2D array to have n_px pixels while preserving aspect ratio."""
    zoom_factor = n_px / arr.shape[0]
    return zoom(arr, zoom_factor)


def insert_galaxy_zoo(
    update_progress,
    datacube,
    continum,
    line_fluxes,
    pos_z,
    fwhm_z,
    n_px,
    n_chan,
    data_path,
):
    files = np.array(os.listdir(data_path))
    imfile = os.path.join(data_path, np.random.choice(files))
    img = plimg.imread(imfile).astype(np.float32)
    dims = np.shape(img)
    d3 = min(2, dims[2])
    avimg = np.average(img[:, :, :d3], axis=2)
    avimg -= np.min(avimg)
    avimg *= 1 / np.max(avimg)
    avimg = interpolate_array(avimg, n_px)
    avimg /= np.sum(avimg)
    z_idxs = np.arange(0, n_chan)
    gs = np.zeros(n_chan)
    cube = np.zeros((n_px, n_px, n_chan))
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cube[:, :, z] += avimg * (continum[z] + gs[z])
        update_progress.emit(z / n_chan * 100)
    datacube._array[:, :, :] = cube * U.Jy / U.pix**2
    return datacube


def insert_tng(
    n_px,
    n_channels,
    freq_sup,
    snapshot,
    subhalo_id,
    distance,
    x_rot,
    y_rot,
    tngpath,
    ra,
    dec,
    api_key,
    ncpu,
):
    source = myTNGSource(
        snapNum=snapshot,
        subID=subhalo_id,
        distance=distance * U.Mpc,
        rotation={"L_coords": (x_rot, y_rot)},
        basePath=tngpath,
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
        velocity_centre=source.vsys,
        ra=source.ra,
        dec=source.dec,
    )
    spectral_model = GaussianSpectrum(sigma="thermal")
    sph_kernel = WendlandC2Kernel()
    M = Martini(
        source=source,
        datacube=datacube,
        sph_kernel=sph_kernel,
        spectral_model=spectral_model,
        quiet=False,
        find_distance=False,
    )
    M.insert_source_in_cube(skip_validation=True, progressbar=True, ncpu=ncpu)
    return M


def insert_extended(
    terminal, datacube, tngpath, snapshot, subhalo_id, redshift, ra, dec, api_key, ncpu
):
    x_rot = np.random.randint(0, 360) * U.deg
    y_rot = np.random.randint(0, 360) * U.deg
    tngpath = os.path.join(tngpath, "TNG100-1", "output")
    redshift = redshift * cu.redshift
    distance = redshift.to(U.Mpc, cu.redshift_distance(WMAP9, kind="comoving"))
    terminal.add_log(
        "Computed a distance of {} for redshift {}".format(distance, redshift)
    )
    distance = 50
    M = insert_tng(
        datacube.n_px_x,
        datacube.n_channels,
        datacube.channel_width,
        snapshot,
        subhalo_id,
        distance,
        x_rot,
        y_rot,
        tngpath,
        ra,
        dec,
        api_key,
        ncpu,
    )
    initial_mass_ratio = M.inserted_mass / M.source.input_mass * 100
    terminal.add_log("Mass ratio: {}%".format(initial_mass_ratio))
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
        terminal.add_log("Injecting source at distance {}".format(distance))
        M = insert_tng(
            datacube.n_px_x,
            datacube.n_channels,
            datacube.channel_width,
            snapshot,
            subhalo_id,
            distance,
            x_rot,
            y_rot,
            tngpath,
            ra,
            dec,
            api_key,
            ncpu,
        )
        mass_ratio = M.inserted_mass / M.source.input_mass * 100
        terminal.add_log("Mass ratio: {}%".format(mass_ratio))
    terminal.add_log("Datacube generated, inserting source")
    return M.datacube


def diffuse_signal(n_px):
    ift.random.push_sseq(random.randint(1, 1000))
    space = ift.RGSpace((2 * n_px, 2 * n_px))
    args = {
        "offset_mean": 24,
        "offset_std": (1, 0.1),
        "fluctuations": (5.0, 1.0),
        "loglogavgslope": (-3.5, 0.5),
        "flexibility": (1.2, 0.4),
        "asperity": (0.2, 0.2),
    }

    cf = ift.SimpleCorrelatedField(space, **args)
    exp_cf = ift.exp(cf)
    random_pos = ift.from_random(exp_cf.domain)
    sample = np.log(exp_cf(random_pos))
    data = sample.val[0:n_px, 0:n_px]
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data


def insert_diffuse(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_px, n_chan
):
    z_idxs = np.arange(0, n_chan)
    ts = diffuse_signal(n_px)
    ts = np.nan_to_num(ts)
    ts = ts / np.sum(ts)
    cube = np.zeros((n_px, n_px, n_chan)) * ts
    gs = np.zeros(n_chan)
    for i in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[i], pos_z[i], fwhm_z[i])
    for z in range(0, n_chan):
        cube[:, :, z] *= continum[z] + gs[z]
    datacube._array[:, :, :] += cube * U.Jy * U.pix**-2
    update_progress.emit(z / n_chan * 100)
    return datacube


def distance_1d(p1, p2):
    return math.sqrt((p1 - p2) ** 2)


def distance_2d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_1d(bb1, bb2):
    assert bb1["z1"] < bb1["z2"]
    assert bb2["z1"] < bb2["z2"]
    z_left = max(bb1["z1"], bb2["z1"])
    z_right = min(bb1["z2"], bb2["z2"])
    if z_right < z_left:
        return 0.0
    intersection = z_right - z_left
    bb1_area = bb1["z2"] - bb1["z1"]
    bb2_area = bb2["z2"] - bb2["z1"]
    union = bb1_area + bb2_area - intersection
    return intersection / union


def get_pos(x_radius, y_radius, z_radius):
    x = np.random.randint(-x_radius, x_radius)
    y = np.random.randint(-y_radius, y_radius)
    z = np.random.randint(-z_radius, z_radius)
    return (x, y, z)


def sample_positions(
    terminal,
    pos_x,
    pos_y,
    pos_z,
    fwhm_x,
    fwhm_y,
    fwhm_z,
    n_components,
    fwhm_xs,
    fwhm_ys,
    fwhm_zs,
    xy_radius,
    z_radius,
    sep_xy,
    sep_z,
):
    sample = []
    i = 0
    n = 0
    while (len(sample) < n_components) and (n < 1000):
        new_p = get_pos(xy_radius, xy_radius, z_radius)
        new_p = int(new_p[0] + pos_x), int(new_p[1] + pos_y), int(new_p[2] + pos_z)
        if len(sample) == 0:
            spatial_dist = distance_2d((new_p[0], new_p[1]), (pos_x, pos_y))
            freq_dist = distance_1d(new_p[2], pos_z)
            if spatial_dist < sep_xy or freq_dist < sep_z:
                n += 1
                continue
            else:
                spatial_iou = get_iou(
                    {
                        "x1": new_p[0] - fwhm_xs[i],
                        "x2": new_p[0] + fwhm_xs[i],
                        "y1": new_p[1] - fwhm_ys[i],
                        "y2": new_p[1] + fwhm_ys[i],
                    },
                    {
                        "x1": pos_x - fwhm_x,
                        "x2": pos_x + fwhm_x,
                        "y1": pos_y - fwhm_y,
                        "y2": pos_y + fwhm_y,
                    },
                )
                freq_iou = get_iou_1d(
                    {"z1": new_p[2] - fwhm_zs[i], "z2": new_p[2] + fwhm_zs[i]},
                    {"z1": pos_z - fwhm_z, "z2": pos_z + fwhm_z},
                )
                if spatial_iou > 0.1 or freq_iou > 0.1:
                    n += 1
                    continue
                else:
                    sample.append(new_p)
                    i += 1
                    n = 0
                    terminal.add_log("Found {}st component".format(len(sample)))
        else:
            spatial_distances = [
                distance_2d((new_p[0], new_p[1]), (p[0], p[1])) for p in sample
            ]
            freq_distances = [distance_1d(new_p[2], p[2]) for p in sample]
            checks = [
                spatial_dist < sep_xy or freq_dist < sep_z
                for spatial_dist, freq_dist in zip(spatial_distances, freq_distances)
            ]
            if any(checks) is True:
                n += 1
                continue
            else:
                spatial_iou = [
                    get_iou(
                        {
                            "x1": new_p[0] - fwhm_xs[i],
                            "x2": new_p[0] + fwhm_xs[i],
                            "y1": new_p[1] - fwhm_ys[i],
                            "y2": new_p[1] + fwhm_ys[i],
                        },
                        {
                            "x1": p[0] - fwhm_xs[j],
                            "x2": p[0] + fwhm_xs[j],
                            "y1": p[1] - fwhm_ys[j],
                            "y2": p[1] + fwhm_ys[j],
                        },
                    )
                    for j, p in enumerate(sample)
                ]
                freq_iou = [
                    get_iou_1d(
                        {"z1": new_p[2] - fwhm_zs[i], "z2": new_p[2] + fwhm_zs[i]},
                        {"z1": p[2] - fwhm_zs[j], "z2": p[2] + fwhm_zs[j]},
                    )
                    for j, p in enumerate(sample)
                ]
                checks = [
                    spatial_iou > 0.1 or freq_iou > 0.1
                    for spatial_iou, freq_iou in zip(spatial_iou, freq_iou)
                ]
                if any(checks) is True:
                    n += 1
                    continue
                else:
                    i += 1
                    n = 0
                    sample.append(new_p)
                    terminal.add_log("Found {}st component".format(len(sample)))

    return sample


def insert_serendipitous(
    terminal,
    update_progress,
    datacube,
    continum,
    cont_sens,
    line_fluxes,
    line_names,
    line_frequencies,
    freq_sup,
    pos_zs,
    fwhm_x,
    fwhm_y,
    fwhm_zs,
    n_px,
    n_chan,
    sim_params_path,
):
    wcs = datacube.wcs
    xy_radius = n_px / 4
    z_radius = n_chan / 2
    n_sources = np.random.randint(1, 5)
    # Generate fwhm for x and y
    fwhm_xs = np.random.randint(1, fwhm_x, n_sources)
    fwhm_ys = np.random.randint(1, fwhm_y, n_sources)
    # generate a random number of lines for each serendipitous source
    if len(line_fluxes) == 1:
        n_lines = np.array([1] * n_sources)
    else:
        n_lines = np.random.randint(1, 3, n_sources)
    # generate the width of the first line based on the first line of the central source
    s_fwhm_zs = np.random.randint(2, fwhm_zs[0], n_sources)
    # get posx and poy of the centtral source
    pos_x, pos_y, _ = datacube.wcs.sub(3).wcs_world2pix(
        datacube.ra, datacube.dec, datacube.velocity_centre, 0
    )
    # get a mininum separation based on spatial dimensions
    sep_x, sep_z = np.random.randint(0, xy_radius), np.random.randint(0, z_radius)
    # get the position of the first line of the central source
    pos_z = pos_zs[0]
    # get maximum continum value
    cont_peak = np.max(continum)
    # get serendipitous continum maximum
    serendipitous_norms = np.random.uniform(cont_sens, cont_peak, n_sources)
    # normalize continum to each serendipitous continum maximum
    serendipitous_conts = np.array(
        [
            continum * serendipitous_norm / cont_peak
            for serendipitous_norm in serendipitous_norms
        ]
    )
    # sample coordinates of the first line
    sample_coords = sample_positions(
        terminal,
        pos_x,
        pos_y,
        pos_z,
        fwhm_x,
        fwhm_y,
        fwhm_zs[0],
        n_sources,
        fwhm_xs,
        fwhm_ys,
        s_fwhm_zs,
        xy_radius,
        z_radius,
        sep_x,
        sep_z,
    )
    # get the rotation angles
    pas = np.random.randint(0, 360, n_sources)
    with open(sim_params_path, "w") as f:
        f.write("\n Injected {} serendipitous sources\n".format(n_sources))
        f.close()
    for c_id, choords in enumerate(sample_coords):
        with open(sim_params_path, "w") as f:
            n_line = n_lines[c_id]
            terminal.add_log(
                "Simulating serendipitous source {} with {} lines".format(
                    c_id + 1, n_line
                )
            )
            s_line_fluxes = np.random.uniform(cont_sens, np.max(line_fluxes), n_line)
            s_line_names = line_names[:n_line]
            for s_name, s_flux in zip(s_line_names, s_line_fluxes):
                terminal.add_log("Line {} Flux: {}".format(s_name, s_flux))
            pos_x, pos_y, pos_z = choords
            delta = pos_z - pos_zs[0]
            pos_z = np.array([pos + delta for pos in pos_zs])[:n_line]
            s_ra, s_dec, _ = wcs.sub(3).wcs_pix2world(pos_x, pos_y, 0, 0)
            s_freq = np.array(
                [line_freq + delta * freq_sup for line_freq in line_frequencies]
            )[:n_line]
            fwhmsz = [s_fwhm_zs[0]]
            for _ in range(n_line - 1):
                fwhmsz.append(np.random.randint(2, np.random.choice(fwhm_zs, 1))[0])
            s_continum = serendipitous_conts[c_id]
            f.write("RA: {}\n".format(s_ra))
            f.write("DEC: {}\n".format(s_dec))
            f.write("FWHM_x (pixels): {}\n".format(fwhm_xs[c_id]))
            f.write("FWHM_y (pixels): {}\n".format(fwhm_ys[c_id]))
            f.write("Projection Angle: {}\n".format(pas[c_id]))
            for i in range(len(s_freq)):
                f.write(
                    f"Line: {s_line_names[i]} - Frequency: {s_freq[i]} GHz "
                    f"- Flux: {line_fluxes[i]} Jy - Width (Channels): {fwhmsz[i]}\n"
                )
            datacube = insert_gaussian(
                update_progress,
                datacube,
                s_continum,
                s_line_fluxes,
                pos_x,
                pos_y,
                pos_z,
                fwhm_xs[c_id],
                fwhm_ys[c_id],
                fwhmsz,
                pas[c_id],
                n_px,
                n_chan,
            )
            f.close()
    return datacube


def get_datacube_header(datacube, obs_date):
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
    header.append(("BUNIT", datacube._array.unit.to_string("fits")))
    header.append(("MJD-OBS", obs_date))
    header.append(("BTYPE", "Intensity"))
    header.append(("SPECSYS", wcs_header["SPECSYS"]))
    return header


def write_datacube_to_fits(
    datacube,
    filename,
    obs_date,
    channels="frequency",
    overwrite=True,
):
    """
    Output the DataCube to a FITS-format file.

    Parameters
    ----------
    filename : string
        Name of the file to write. '.fits' will be appended if not already
        present.

    channels : {'frequency', 'velocity'}, optional
        Type of units used along the spectral axis in output file.
        (Default: 'frequency'.)

    overwrite: bool, optional
        Whether to allow overwriting existing files. (Default: True.)
    """

    datacube.drop_pad()
    if channels == "frequency":
        datacube.freq_channels()
    elif channels == "velocity":
        datacube.velocity_channels()
    else:
        raise ValueError("Unknown 'channels' value " "(use 'frequency' or 'velocity'.")

    filename = filename if filename[-5:] == ".fits" else filename + ".fits"

    wcs_header = datacube.wcs.to_header()
    wcs_header.rename_keyword("WCSAXES", "NAXIS")
    header = fits.Header()
    if len(datacube._array.shape) == 3:
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
    else:
        header.append(("SIMPLE", "T"))
        header.append(("BITPIX", 16))
        header.append(("NAXIS", wcs_header["NAXIS"]))
        header.append(("NAXIS1", datacube.n_px_x))
        header.append(("NAXIS2", datacube.n_px_y))
        header.append(("NAXIS3", datacube.n_channels))
        header.append(("NAXIS4", 1))
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
        header.append(("CDELT4", wcs_header["CDELT4"]))
        header.append(("CRPIX4", wcs_header["CRPIX4"]))
        header.append(("CRVAL4", wcs_header["CRVAL4"]))
        header.append(("CTYPE4", wcs_header["CTYPE4"]))
        header.append(("CUNIT4", "PAR"))
    header.append(("EPOCH", 2000.0))
    # header.append(('BLANK', -32768)) #only for integer data
    header.append(("BSCALE", 1.0))
    header.append(("BZERO", 0.0))
    datacube_array_units = datacube._array.unit
    header.append(("DATAMAX", np.max(datacube._array.to_value(datacube_array_units))))
    header.append(("DATAMIN", np.min(datacube._array.to_value(datacube_array_units))))

    # long names break fits format, don't let the user set this
    header.append(("OBJECT", "MOCK"))
    header.append(("BUNIT", datacube_array_units.to_string("fits")))
    header.append(("MJD-OBS", obs_date))
    header.append(("BTYPE", "Intensity"))
    header.append(("SPECSYS", wcs_header["SPECSYS"]))

    # flip axes to write
    hdu = fits.PrimaryHDU(
        header=header, data=datacube._array.to_value(datacube_array_units).T
    )
    hdu.writeto(filename, overwrite=overwrite)

    if channels == "frequency":
        datacube.velocity_channels()
    return
