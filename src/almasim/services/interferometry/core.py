"""Core interferometry classes and functionality."""
from __future__ import annotations

from typing import Callable, Optional
import os
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as U
from astropy.constants import c
from dask.distributed import Client

from .imaging import image_channel

LogFn = Optional[Callable[[str], None]]


class ProgressSignal:
    """Simple progress callback system (not PyQt-specific)."""

    def __init__(self):
        self._subscribers: list[Callable[[int], None]] = []

    def connect(self, callback: Optional[Callable[[int], None]]) -> None:
        """Connect a callback function to receive progress updates."""
        if callback is not None:
            self._subscribers.append(callback)

    def emit(self, value: int) -> None:
        """Emit progress value to all connected callbacks."""
        for callback in self._subscribers:
            callback(value)


class Interferometer:
    """ALMA interferometric observation simulator."""

    def __init__(
        self,
        idx: int,
        skymodel: np.ndarray,
        client: Client,
        main_dir: str,
        output_dir: str,
        ra: U.Quantity,
        dec: U.Quantity,
        central_freq: U.Quantity,
        bandwidth: U.Quantity,
        fov: float,
        antenna_array: str,
        noise: float,
        snr: float,
        integration_time: float,
        observation_date: str,
        header: fits.Header,
        save_mode: str,
        robust: float,
        logger: LogFn = None,
    ):
        """
        Initialize interferometer.

        Parameters
        ----------
        idx : int
            Simulation index
        skymodel : np.ndarray
            Sky model cube
        client : Client
            Dask client for parallel processing
        main_dir : str
            Main data directory
        output_dir : str
            Output directory
        ra : U.Quantity
            Right ascension
        dec : U.Quantity
            Declination
        central_freq : U.Quantity
            Central frequency
        bandwidth : U.Quantity
            Bandwidth
        fov : float
            Field of view in arcsec
        antenna_array : str
            Antenna array configuration
        noise : float
            Noise level
        snr : float
            Signal-to-noise ratio
        integration_time : float
            Integration time in hours
        observation_date : str
            Observation date
        header : fits.Header
            FITS header
        save_mode : str
            Save mode ('npz', 'h5', or 'fits')
        robust : float
            Robustness parameter
        logger : LogFn, optional
            Logger callback function
        """
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
        self.logger = logger
        self.progress_signal = ProgressSignal()

        # Initialize variables
        self._init_variables()
        # Get the observing location
        self._get_observing_location()
        # Get coverage and antennas
        self._get_Hcov()
        self._read_antennas()
        # Get the observing wavelengths for each channel
        self._get_wavelengths()

        msg = f"Performing {self.nH} scans with a scan time of {self.scan_time} seconds"
        self._log(msg)

    def _log(self, message: str) -> None:
        """Log a message using the logger callback or print."""
        if self.logger is not None:
            self.logger(message)
        else:
            print(message)

    def _hz_to_m(self, freq: float) -> float:
        """Convert frequency in Hz to wavelength in meters."""
        return self.c_ms / freq

    def _init_variables(self) -> None:
        """Initialize calculation variables."""
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
        self._log(f"Number of Epochs: {self.nH}")

    def _get_observing_location(self) -> None:
        """Get ALMA observing location."""
        self.observing_location = EarthLocation.of_site("ALMA")

    def _get_Hcov(self) -> None:
        """Calculate hour angle coverage."""
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

    def _get_hour_angle(self, time: Time) -> float:
        """Calculate hour angle for given time."""
        lst = time.sidereal_time("apparent", longitude=self.observing_location.lon)
        ha = lst.deg - self.ra
        if ha < 0:
            ha += 360
        return ha

    def _get_az_el(self) -> None:
        """Calculate azimuth and elevation."""
        self._get_observing_location()
        self._get_middle_time()
        sky_coords = SkyCoord(
            ra=self.ra * self.rad2deg, dec=self.dec * self.rad2deg, unit="deg"
        )
        aa = AltAz(location=self.observing_location, obstime=self.middle_time)
        sky_coords.transform_to(aa)
        self.az = sky_coords.az
        self.el = sky_coords.alt

    def _get_nH(self) -> None:
        """Calculate number of hour angle samples."""
        self.scan_time = 6
        self.nH = int(self.integration_time / (self.scan_time * self.second2hour))
        if self.nH > 200:
            self.scan_time = 8.064
            self.nH = int(self.integration_time / (self.scan_time * self.second2hour))
            if self.nH > 200:
                self.scan_time = 18.144
                self.nH = int(self.integration_time / (self.scan_time * self.second2hour))
                if self.nH > 200:
                    self.scan_time = 30.24
                    self.nH = int(self.integration_time / (self.scan_time * self.second2hour))
        self.header.append(("EPOCH", self.nH))

    def _read_antennas(self) -> None:
        """Read antenna positions from configuration."""
        antenna_coordinates = pd.read_csv(
            os.path.join(self.main_dir, "antenna_config", "antenna_coordinates.csv")
        )
        obs_antennas = self.antenna_array.split(" ")
        obs_antennas = [antenna.split(":")[0] for antenna in obs_antennas]
        obs_coordinates = antenna_coordinates[
            antenna_coordinates["name"].isin(obs_antennas)
        ]
        antenna_coordinates = obs_coordinates[["x", "y"]].values
        antPos = []
        Xmax = 0.0
        for line in antenna_coordinates:
            antPos.append([line[0] * 1e-3, line[1] * 1e-3])
            Xmax = np.max(np.abs(antPos[-1] + [Xmax]))
        self.Xmax = Xmax
        self.antPos = antPos

        cosW = -np.tan(self.lat) * np.tan(self.dec)
        if np.abs(cosW) < 1.0:
            Hhor = np.arccos(cosW)
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

    def _get_wavelengths(self) -> None:
        """Calculate observing wavelengths for each channel."""
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

    def get_channel_wavelength(self, channel: int) -> tuple:
        """Get wavelength and format string for a channel."""
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

    def run_interferometric_sim(self) -> dict:
        """Run interferometric simulation."""
        # Scatter input data to workers
        scattered_channels = [
            self.client.scatter(self.skymodel[i]) for i in range(self.skymodel.shape[0])
        ]

        delayed_results = []
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
                self.robust,
            )
            delayed_results.append(delayed_result)

        # Compute per-channel futures
        futures_per_channel = [self.client.compute(dr) for dr in delayed_results]

        # Start tracking progress of the per-channel computations
        self.track_progress(futures_per_channel)

        # Gather the results after completion
        results_per_channel = self.client.gather(futures_per_channel)

        # Extract and stack the outputs
        modelcube = [res[0] for res in results_per_channel]
        dirtycube = [res[1] for res in results_per_channel]
        modelvis = [res[2] for res in results_per_channel]
        dirtyvis = [res[3] for res in results_per_channel]
        u = [res[4] for res in results_per_channel]
        v = [res[5] for res in results_per_channel]
        beam = [res[6] for res in results_per_channel]
        totsampling = [res[7] for res in results_per_channel]

        # Stack the results into 3D arrays
        modelCube = np.stack(modelcube, axis=0)
        modelVis = np.stack(modelvis, axis=0)
        dirtyCube = np.stack(dirtycube, axis=0)
        dirtyVis = np.stack(dirtyvis, axis=0)
        u = np.stack(u, axis=0)
        v = np.stack(v, axis=0)
        beam = np.stack(beam, axis=0)
        totsampling = np.stack(totsampling, axis=0)

        # Save the results
        self._savez_compressed_cubes(modelCube, modelVis, dirtyCube, dirtyVis)

        # Get wavelength and format for middle channel
        self.s_wavelength, self.s_fmtB = self.get_channel_wavelength(self.Nchan // 2)

        # Prepare the simulation results dictionary
        simulation_results = {
            "model_cube": modelCube,
            "model_vis": modelVis,
            "dirty_cube": dirtyCube,
            "dirty_vis": dirtyVis,
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

        # Clean up variables to free memory
        del (
            modelCube,
            modelVis,
            dirtyCube,
            dirtyVis,
            u,
            v,
            beam,
            totsampling,
            modelcube,
            modelvis,
            dirtycube,
            dirtyvis,
        )

        return simulation_results

    def track_progress(self, futures: list) -> None:
        """Track progress of Dask futures."""
        total_tasks = len(futures)
        completed_tasks = 0
        while completed_tasks < total_tasks:
            completed_tasks = sum(f.done() for f in futures)
            progress_value = int((completed_tasks / total_tasks) * 100)
            self.progress_signal.emit(progress_value)
            time.sleep(1)

    def _savez_compressed_cubes(
        self,
        modelCube: np.ndarray,
        visCube: np.ndarray,
        dirtyCube: np.ndarray,
        dirtyvisCube: np.ndarray,
    ) -> None:
        """Save simulation results in various formats."""
        if self.save_mode == "npz":
            np.savez_compressed(
                os.path.join(self.output_dir, f"clean-cube_{self.idx}.npz"),
                modelCube,
            )
            np.savez_compressed(
                os.path.join(self.output_dir, f"dirty-cube_{self.idx}.npz"),
                dirtyCube,
            )
            np.savez_compressed(
                os.path.join(self.output_dir, f"dirty-vis-cube_{self.idx}.npz"),
                dirtyvisCube,
            )
            np.savez_compressed(
                os.path.join(self.output_dir, f"clean-vis-cube_{self.idx}.npz"),
                visCube,
            )
        elif self.save_mode == "h5":
            with h5py.File(
                os.path.join(self.output_dir, f"clean-cube_{self.idx}.h5"), "w"
            ) as f:
                f.create_dataset("clean_cube", data=modelCube)
            with h5py.File(
                os.path.join(self.output_dir, f"dirty-cube_{self.idx}.h5"), "w"
            ) as f:
                f.create_dataset("dirty_cube", data=dirtyCube)
            with h5py.File(
                os.path.join(self.output_dir, f"dirty-vis-cube_{self.idx}.h5"), "w"
            ) as f:
                f.create_dataset("dirty_vis_cube", data=dirtyvisCube)
            with h5py.File(
                os.path.join(self.output_dir, f"clean-vis-cube_{self.idx}.h5"), "w"
            ) as f:
                f.create_dataset("clean_vis_cube", data=visCube)
        elif self.save_mode == "fits":
            self.clean_header = self.header.copy()
            self.clean_header.append(("DATAMAX", np.max(modelCube)))
            self.clean_header.append(("DATAMIN", np.min(modelCube)))
            hdu = fits.PrimaryHDU(header=self.clean_header, data=modelCube)
            hdu.writeto(
                os.path.join(self.output_dir, f"clean-cube_{self.idx}.fits"),
                overwrite=True,
            )
            self.dirty_header = self.header.copy()
            self.dirty_header.append(("DATAMAX", np.max(dirtyCube)))
            self.dirty_header.append(("DATAMIN", np.min(dirtyCube)))
            hdu = fits.PrimaryHDU(header=self.dirty_header, data=dirtyCube)
            hdu.writeto(
                os.path.join(self.output_dir, f"dirty-cube_{self.idx}.fits"),
                overwrite=True,
            )
            real_part = np.real(dirtyvisCube)
            imag_part = np.imag(dirtyvisCube)
            hdu_real = fits.PrimaryHDU(real_part)
            hdu_imag = fits.PrimaryHDU(imag_part)
            hdu_real.writeto(
                os.path.join(self.output_dir, f"dirty-vis-cube_real{self.idx}.fits"),
                overwrite=True,
            )
            hdu_imag.writeto(
                os.path.join(self.output_dir, f"dirty-vis-cube_imag{self.idx}.fits"),
                overwrite=True,
            )
            real_part = np.real(visCube)
            imag_part = np.imag(visCube)
            hdu_real = fits.PrimaryHDU(real_part)
            hdu_imag = fits.PrimaryHDU(imag_part)
            hdu_real.writeto(
                os.path.join(self.output_dir, f"clean-vis-cube_real{self.idx}.fits"),
                overwrite=True,
            )
            hdu_imag.writeto(
                os.path.join(self.output_dir, f"clean-vis-cube_imag{self.idx}.fits"),
                overwrite=True,
            )
            del real_part
            del imag_part

        self._log(f"Total Flux detected in model cube: {round(np.sum(modelCube), 2)} Jy")
        self._log(f"Total Flux detected in dirty cube: {round(np.sum(dirtyCube), 2)} Jy")


