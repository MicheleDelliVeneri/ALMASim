from PyQt6.QtCore import QObject, pyqtSignal
from astropy.io import fits
from astropy.time import Time
from scipy.integrate import odeint
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import astropy.units as U
from astropy.constants import c
import scipy.ndimage.interpolation as spndint
import matplotlib.cm as cm
import numpy as np
import os
import matplotlib
import h5py

matplotlib.use("Agg")


def showError(message):
    raise Exception(message)


class Interferometer(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(
        self,
        idx,
        skymodel,
        main_dir,
        output_dir,
        ra,
        dec,
        central_freq,
        band_range,
        fov,
        antenna_array,
        noise,
        int_time,
        obs_date,
        header,
        save_mode,
        terminal,
        robust=0.5,
    ):
        super().__init__()
        self.idx = idx
        self.terminal = terminal
        self.skymodel = skymodel
        self.antenna_array = antenna_array
        self.noise = noise
        self.int_time = int_time
        self.obs_date = obs_date
        # Constants
        self.c_ms = c.to(U.m / U.s).value
        # Directories
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, "plots")
        # Parameters
        self.Hfac = np.pi / 180.0 * 15.0
        self.deg2rad = np.pi / 180.0
        self.rad2deg = 180.0 / np.pi
        self.deg2arcsec = 3600.0
        self.arcsec2deg = 1.0 / 3600.0
        self.second2hour = 1.0 / 3600.0
        self.curzoom = [0, 0, 0, 0]
        self.robust = robust
        self.deltaAng = 1.0 * self.deg2rad
        self.gamma = 0.5
        self.lfac = 1.0e6
        self.header = header
        self._get_nH()
        self.terminal.add_log(
            f"Performing {self.nH} scans with a scan time of {self.scan_time} seconds"
        )
        self.Hmax = np.pi
        self.lat = -23.028 * self.deg2rad
        self.trlat = [np.sin(self.lat), np.cos(self.lat)]
        self.Diameters = [12.0, 0]
        self.ra = ra.value * self.deg2rad
        self.dec = dec.value * self.deg2rad
        self.trdec = [np.sin(self.dec), np.cos(self.dec)]
        self.central_freq = central_freq.to(U.Hz).value
        self.band_range = band_range.to(U.Hz).value
        self.imsize = 2 * 1.5 * fov.value
        self.Xaxmax = self.imsize / 2.0
        self.Npix = skymodel.shape[1]
        self.Np4 = self.Npix // 4
        self.Nchan = skymodel.shape[0]
        self.Nphf = self.Npix // 2
        self.pixsize = float(self.imsize) / self.Npix
        self.xx = np.linspace(-self.imsize / 2.0, self.imsize / 2.0, self.Npix)
        self.yy = np.ones(self.Npix, dtype=np.float32)
        self.distmat = (
            -np.outer(self.xx**2.0, self.yy) - np.outer(self.yy, self.xx**2.0)
        ) * self.pixsize**2.0
        self.robfac = 0.0
        self.W2W1 = 1
        self.currcmap = cm.jet
        self.zooming = 0
        self.save_mode = save_mode
        # Get the antenna coordinates, and the hour angle coverage
        self._get_observing_location()
        # This function must be checked
        self._get_Hcov()
        self._read_antennas()
        # Get the observing wavelengths for each channel
        self._get_wavelengths()
        self._prepare_cubes()
        self.terminal.add_log(f"Hour Angle Coverage {self.Hcov[0]} - {self.Hcov[1]}")

    def run_interferometric_sim(self):
        for channel in range(self.Nchan):
            self._image_channel(channel, self.skymodel)
            self.progress_signal.emit((channel + 1) * 100 // self.Nchan)
        self._savez_compressed_cubes()
        simulation_results = {
            "modelCube": self.modelCube,
            "dirtyCube": self.dirtyCube,
            "visCube": self.visCube,
            "dirtyvisCube": self.dirtyvisCube,
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
            "beam": self.s_beam,
            "fmtB": self.s_fmtB,
            "totsampling": self.s_totsampling,
            "wavelength": self.s_wavelength,
            "curzoom": self.curzoom,
            "Nphf": self.Nphf,
            "Xmax": self.Xmax,
            "u": self.s_u,
            "v": self.s_v,
            "antPos": self.antPos,
            "Nant": self.Nant,
        }
        self._free_space()
        return simulation_results

    def _get_hour_angle(self, time):
        lst = time.sidereal_time("apparent", longitude=self.observing_location.lon)
        ha = lst.deg - self.ra
        if ha < 0:
            ha += 360
        return ha

    # ------- Utility Functions --------------------------
    def _get_observing_location(self):
        self.observing_location = EarthLocation.of_site("ALMA")

    def _get_Hcov(self):
        self.int_time = self.int_time * U.s
        start_time = Time(self.obs_date + "T00:00:00", format="isot", scale="utc")
        middle_time = start_time + self.int_time / 2
        end_time = start_time + self.int_time
        ha_start = self._get_hour_angle(start_time)
        ha_middle = self._get_hour_angle(middle_time)
        ha_end = self._get_hour_angle(end_time)
        start = ha_start - ha_middle
        end = ha_end - ha_middle
        self.Hcov = [start, end]

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

    def _get_initial_and_final_H(self):

        def differential_equations(y, t, phi, A, E, forward=True):
            """
            Defines the system of differential equations.
            Args:
                y: Current state vector (A, E).
                t: Independent variable (time).
                tau: Time constant.
                phi: Angle (radians).
                A0: Initial value of A.
                E0: Initial value of E.

            Returns:
                The derivatives of A and E (dA/dt, dE/dt).
            """

            def dtau_dt(t):
                """
                PLACEHOLDER TO BE SUBSTITUTED WITH REAL
                """
                return 1

            def dtau_dt_reversed(t):
                return -dtau_dt(t)

            A, E = y
            if forward is True:
                dA_dt = dtau_dt(t) * (
                    (np.sin(phi) * np.cos(E) - np.cos(phi) * np.sin(E) * np.cos(A))
                    / np.cos(E)
                )
                dE_dt = dtau_dt(t) * (np.cos(phi) * np.sin(A))
            else:
                dA_dt = dtau_dt_reversed(t) * (
                    (np.sin(phi) * np.cos(E) - np.cos(phi) * np.sin(E) * np.cos(A))
                    / np.cos(E)
                )
                dE_dt = dtau_dt_reversed(t) * (np.cos(phi) * np.sin(A))
            return [dA_dt, dE_dt]

        self._get_az_el()
        y0 = [self.az, self.el]
        t_final_solve = np.linspace(self.middle_time, self.end_time, self.nH)
        sol_final = odeint(
            differential_equations, y0, t_final_solve, args=(self.lat, self.az, self.el)
        )
        az_final = sol_final[-1, 0]
        el_final = sol_final[-1, 1]
        t_initial_solve = np.linspace(self.middle_time, self.start_time, self.nH)[::-1]
        sol_initial = odeint(
            differential_equations,
            y0,
            t_initial_solve,
            args=(self.lat, self.az, self.el, False),
        )
        az_initial = sol_initial[-1, 0]
        el_initial = sol_initial[-1, 1]
        aa_initial = AltAz(
            az=az_initial,
            alt=el_initial,
            location=self.observing_location,
            obstime=self.start_time,
        )
        aa_final = AltAz(
            az=az_final,
            alt=el_final,
            location=self.observing_location,
            obstime=self.end_time,
        )
        self.initial_H = -aa_initial.alt.to(U.hourangle).value
        self.final_H = -aa_final.alt.to(U.hourangle).value
        print(self.initial_H, self.final_H)

    def _get_nH(self):
        self.scan_time = 6
        self.nH = int(self.int_time / (self.scan_time * self.second2hour))
        if self.nH > 200:
            # Try increasing the divisor to 8.064 to lower nH
            self.scan_time = 8.064
            self.nH = int(self.int_time / (self.scan_time * self.second2hour))
            if self.nH > 200:
                # Further increase the divisor to 18.144
                self.scan_time = 18.144
                self.nH = int(self.int_time / (self.scan_time * self.second2hour))
                if self.nH > 200:
                    self.scan_time = 30.24
                    # Final attempt with the largest divisor (30.24)
                    self.nH = int(self.int_time / (self.scan_time * self.second2hour))
        self.header.append(("EPOCH", self.nH))

    def _read_antennas(self):
        antenna_coordinates = pd.read_csv(
            os.path.join(
                self.main_dir, "almasim", "antenna_config", "antenna_coordinates.csv"
            )
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
                self.central_freq - self.band_range / 2,
                self.central_freq + self.band_range / 2,
            ]
        ]
        waves = np.linspace(self.w_min, self.w_max, self.Nchan + 1)
        obs_wavelengths = np.array(
            [[waves[i], waves[i + 1]] for i in range(len(waves) - 1)]
        )
        self.obs_wavelengths = obs_wavelengths

    def _prepare_cubes(self):
        self.modelCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.float32)
        self.dirtyCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.float32)
        self.visCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.complex64)
        self.dirtyvisCube = np.zeros(
            (self.Nchan, self.Npix, self.Npix), dtype=np.complex64
        )

    def _hz_to_m(self, freq):
        return self.c_ms / freq

    # ------- Imaging and Interferometric Functions ------
    def _image_channel(self, channel, skymodel):
        self.channel = channel
        self._get_channel_wavelength()
        self._prepare_2d_arrays()
        self._prepare_baselines()  # get the baseline numbers
        self.set_noise()  # set noise level
        self._set_baselines()  # compute baseline vectors and the u-v components
        self._grid_uv()
        self._set_beam()
        self._check_lfac()
        if self.channel == self.Nchan // 2:
            self.s_wavelength = self.wavelength
            self.s_fmtB = self.fmtB
            self.s_totsampling = self.totsampling
            self.s_beam = self.beam
            self.s_u = self.u
            self.s_v = self.v
            # self._plot_beam()
            # self._plot_antennas()
            # self._plot_uv_coverage()
        self.img = skymodel[channel]
        self._prepare_model()
        self._set_primary_beam()
        self._observe()
        self._update_cubes()

    def _get_channel_wavelength(self):
        """
        This method calculates the range of wavelengths for a given channel and formats
         the output for display.

        The method first retrieves the observed wavelengths for the current channel and
         converts them to a list.
        It then calculates the average of the  two wavelengths and appends it to the list.
        The list of wavelengths is then stored in the instance variable `self.wavelength`.

        The method also prepares two formatted strings for display.
        `self.fmtB1` is a string that displays the average wavelength in millimeters.
        `self.fmtB` is a string that includes `self.fmtB1` and additional placeholders
        for flux density (in Jy/beam) and
        changes in right ascension and declination (in degrees).

        Attributes Set:
            self.wavelength (list): A list of wavelengths for the current channel,
            including the average of the first two.
            self.fmtB1 (str): A formatted string displaying the average wavelength.
            self.fmtB (str): A formatted string including `self.fmtB1` and placeholders
            for additional data.

        """
        wavelength = list(self.obs_wavelengths[self.channel] * 1e-3)
        wavelength.append((wavelength[0] + wavelength[1]) / 2.0)
        self.wavelength = wavelength
        self.fmtB1 = r"$\lambda = $ %4.1fmm  " % (self.wavelength[2] * 1.0e6)
        self.fmtB = (
            self.fmtB1
            + "\n"
            + r"% 4.2f Jy/beam"
            + "\n"
            + r"$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f "
        )

    def _prepare_2d_arrays(self):
        """
        This method initializes several 2D arrays with zeros, each of size (Npix x Npix).
        The arrays are used to store various types of data related to the interferometer's
        operation.
        The method is typically called at the start of a new observation or simulation
        to ensure that the arrays are in a clean state.

        Attributes Set:
            self.beam (np.array): A 2D array to store the beam data.
            self.totsampling (np.array): A 2D array to store the total sampling data.
            self.dirtymap (np.array): A 2D array to store the dirty map data.
            self.noisemap (np.array): A 2D array to store the noise map data.
            self.robustsamp (np.array): A 2D array to store the robust sampling data.
            self.Gsampling (np.array): A 2D array to store the G sampling data.
            self.Grobustsamp (np.array): A 2D array to store the G robust sampling data.
            self.GrobustNoise (np.array): A 2D array to store the G robust noise data.
        """
        self.beam = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.totsampling = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.dirtymap = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.noisemap = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.robustsamp = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.Gsampling = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.Grobustsamp = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.GrobustNoise = np.zeros((self.Npix, self.Npix), dtype=np.complex64)

    def _prepare_baselines(self):
        """
        This method prepares the baselines for an interferometer array.
        The method first calculates the number of unique
        baselines in an array of N antennas.
        It then initializes several arrays to store baseline parameters and
        visibility data, including baseline vectors, baseline indices,
        antenna pair indices, complex gains, complex noise values, and hour angle values.
        The method also calculates the sine and cosine of the hour angles
        and stores them in `self.H`.
        It then iterates over all unique antenna pairs,
        assigning a baseline index to each pair
        and storing the antenna pair indices for each baseline.
        Finally, it initializes arrays to store u and v coordinates
        (in wavelengths) for each
        baseline at each hour angle, and sets `self.ravelDims`
        to the shape of these arrays.

        Attributes Set:
            self.Nbas (int): The number of unique baselines in the array.
            self.B (np.array): Baseline vectors for each baseline at each hour angle.
            self.basnum (np.array): A matrix storing the baseline index for each pair
                                    of antennas.
            self.basidx (np.array): A square matrix storing the baseline index for each
                                    pair of antennas.
            self.antnum (np.array): Stores the antenna pair indices for each baseline.
            self.Gains (np.array): Complex gains for each baseline at each hour angle.
            self.Noise (np.array): Complex noise values for each baseline at each
                                   hour angle.
            self.Horig (np.array): Original hour angle values, evenly spaced over the
                                   observation time.
            self.H (list): Trigonometric values (sine and cosine) of the hour angles.
            self.u (np.array): u coordinates (in wavelengths) for each baseline at
                               each hour angle.
            self.v (np.array): v coordinates (in wavelengths) for each baseline at
                               each hour angle.
            self.ravelDims (tuple): The shape of the u and v arrays.
        """
        # Calculate the number of unique baselines in an array of N antennas
        self.Nbas = self.Nant * (self.Nant - 1) // 2
        # Create a redundant variable for the number of baselines
        NBmax = self.Nbas
        # Initialize arrays to store baseline parameters and visibility data:
        # B: Baseline vectors (x, y, z components) for each baseline at each hour angle.
        self.B = np.zeros((NBmax, self.nH), dtype=np.float32)
        # basnum: A matrix storing the baseline index for each pair of antennas.
        self.basnum = np.zeros((self.Nant, self.Nant - 1), dtype=np.int8)
        # basidx: A square matrix storing the baseline index for each pair of
        # antennas (redundant storage).
        self.basidx = np.zeros((self.Nant, self.Nant), dtype=np.int8)
        # antnum: Stores the antenna pair indices for each baseline.
        self.antnum = np.zeros((NBmax, 2), dtype=np.int8)
        # Gains: Complex gains for each baseline at each hour angle (initialized to 1).
        self.Gains = np.ones((self.Nbas, self.nH), dtype=np.complex64)
        # Noise:  Complex noise values for each baseline at each hour angle
        # (initialized to 0).
        self.Noise = np.zeros((self.Nbas, self.nH), dtype=np.complex64)
        # Horig:  Original hour angle values, evenly spaced over the observation time.
        self.Horig = np.linspace(self.Hcov[0], self.Hcov[1], self.nH)
        H = self.Horig[np.newaxis, :]  # Add a new axis for broadcasting
        # Trigonometric values (sine and cosine) of the hour angles.
        self.H = [np.sin(H), np.cos(H)]
        bi = 0  # bi: Baseline index counter (starts at 0).
        # nii: List to keep track of the next available index in basnum for each antenna.
        nii = [0 for n in range(self.Nant)]
        # Iterate over all unique antenna pairs
        for n1 in range(self.Nant - 1):
            for n2 in range(n1 + 1, self.Nant):
                # Assign a baseline index to each antenna pair in both basnum and basidx.
                self.basnum[n1, nii[n1]] = np.int8(bi)
                self.basnum[n2, nii[n2]] = np.int8(bi)
                self.basidx[n1, n2] = np.int8(bi)
                # Store the antenna pair indices for the current baseline.
                self.antnum[bi] = [n1, n2]
                # Increment the next available index for each antenna in basnum.
                nii[n1] += 1
                nii[n2] += 1
                # Increment the baseline index counter
                bi += np.int16(1)
        # Initialize arrays to store u and v coordinates (in wavelengths)
        # for each baseline at each hour angle.
        self.u = np.zeros((NBmax, self.nH))
        self.v = np.zeros((NBmax, self.nH))
        self.ravelDims = (NBmax, self.nH)

    def set_noise(self):
        """
        This method sets the noise level for the interferometer array.

        If the noise level (`self.noise`) is set to 0.0, the method sets all elements
         of the `self.Noise` array to 0.0.

        If the noise level is not 0.0, the method generates a complex random noise
         distribution with a mean of 0 and a
        standard deviation equal to the noise level. This distribution is then assigned
         to the `self.Noise` array.

        The noise distribution is generated using the numpy `random.normal` function,
         which draws random samples from a
        normal (Gaussian) distribution. The real and imaginary parts of the noise are
        generated separately, each with a
        mean of 0 and a standard deviation equal to the noise level.

        Attributes Set:
            self.Noise (np.array): The noise array for the interferometer array.
            Each element represents the noise level
            at a specific baseline and hour angle.

        """
        if self.noise == 0.0:
            self.Noise[:] = 0.0
        else:
            # Inject in the noise array a random distribution
            # with mean 0 and standard deviation equal to the noise
            # The distribution in the complex domain is scaled to the
            # imaginary unit
            self.Noise[:] = np.random.normal(
                loc=0.0, scale=self.noise, size=np.shape(self.Noise)
            ) + 1.0j * np.random.normal(
                loc=0.0, scale=self.noise, size=np.shape(self.Noise)
            )

    def _set_baselines(self, antidx=-1):
        """

        This method is responsible for calculating the baseline vectors
         (in wavelengths) and the
        corresponding u and v coordinates (spatial frequencies) in the
         UV plane for each baseline
        in the interferometer array.
        The method first determines which baselines to update based on
         the provided antenna index (`antidx`).
        If `antidx` is -1, all baselines are updated. If `antidx` is a
         valid antenna index, only baselines
        involving that antenna are updated. If `antidx` is invalid,
         no baselines are updated.

        The method then iterates over the baselines that need updating. For each baseline,
         it calculates the
        baseline vector components (B_x, B_y, B_z) in wavelengths, and the u and v
         coordinates in the UV plane.

        Parameters:
            antidx (int, optional): The index of the antenna for which to update
             baselines. If -1, all baselines are updated. Defaults to -1.

        Attributes Set:
            self.B (np.array): The baseline vectors for each baseline in
                               the interferometer array.
            self.u (np.array): The u coordinates (spatial frequencies)
                               for each baseline in the UV plane.
            self.v (np.array): The v coordinates (spatial frequencies)
                               for each baseline in the UV plane.

        """
        # Determine which baselines to update:
        if antidx == -1:
            # Update all baselines if no specific antenna index is provided.
            bas2change = range(self.Nbas)
        elif antidx < self.Nant:
            # Update only baselines involving the specified antenna.
            bas2change = self.basnum[
                antidx
            ].flatten()  # Get baselines associated with antidx
        else:
            # If the provided antidx is invalid, update no baselines.
            bas2change = []
        # Iterate over the baselines that need updating
        for currBas in bas2change:
            # Get the antenna indices that form the current baseline
            n1, n2 = self.antnum[currBas]
            # Calculate the baseline vector components (B_x, B_y, B_z) in wavelengths:
            # B_x: Projection of baseline onto the plane perpendicular to Earth's
            # rotation axis.
            self.B[currBas, 0] = (
                -(self.antPos[n2][1] - self.antPos[n1][1])
                * self.trlat[0]
                / self.wavelength[2]
            )
            # B_y: Projection of baseline onto the East-West direction.
            self.B[currBas, 1] = (
                self.antPos[n2][0] - self.antPos[n1][0]
            ) / self.wavelength[2]
            # B_z: Projection of baseline onto the North-South direction.
            self.B[currBas, 2] = (
                (self.antPos[n2][1] - self.antPos[n1][1])
                * self.trlat[1]
                / self.wavelength[2]
            )
            # Calculate u and v coordinates (spatial frequencies) in wavelengths:
            # u: Projection of the baseline vector onto the UV plane (East-West
            # component).
            self.u[currBas, :] = -(
                self.B[currBas, 0] * self.H[0] + self.B[currBas, 1] * self.H[1]
            )
            # v: Projection of the baseline vector onto the UV plane (North-South
            # component).
            self.v[currBas, :] = (
                -self.B[currBas, 0] * self.trdec[0] * self.H[1]
                + self.B[currBas, 1] * self.trdec[0] * self.H[0]
                + self.trdec[1] * self.B[currBas, 2]
            )

    def _grid_uv(self, antidx=-1):
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
        - self.pixpos: A list of lists storing the pixel positions for each baseline.
        - self.totsampling: An array storing the total sampling for each pixel
                             in the UV grid.
        - self.Gsampling: An array storing the total gains for each pixel in the UV grid.
        - self.noisemap: An array storing the total noise for each pixel in the UV grid.
        - self.UVpixsize: The pixel size in the UV plane.
        - self.baseline_phases: A dictionary storing the phase of the gains
                                for each baseline.
        - self.bas2change: A list of the baselines to change.
        - self.robfac: A robustness factor based on the total
                         sampling and a user-defined parameter.
        """
        # Determine which baselines to grid:
        if antidx == -1:
            # Grid all baselines if no specific antenna is provided
            bas2change = range(self.Nbas)
            # Initialize lists to store pixel positions for each baseline.
            self.pixpos = [[] for nb in bas2change]
            # Reset sampling, gain, and noise arrays for a clean grid.
            self.totsampling[:] = 0.0
            self.Gsampling[:] = 0.0
            self.noisemap[:] = 0.0
        elif antidx < self.Nant:
            # Grid only the baselines associated with the specified antenna.
            bas2change = list(map(int, list(self.basnum[antidx].flatten())))
        else:
            # Don't grid any baselines if the provided antenna index is invalid.
            bas2change = []
        # set the pixsize in the UV plane
        self.UVpixsize = 2.0 / (self.imsize * np.pi / 180.0 / 3600.0)
        self.baseline_phases = {}
        self.bas2change = bas2change
        for nb in self.bas2change:
            # Calculate the pixel coordinates (pixU, pixV) in the UV plane
            # for each visibility sample of the current baseline.
            # Rounding to the nearest integer determines the UV pixel location
            pixU = np.rint(self.u[nb] / self.UVpixsize).flatten().astype(np.int32)
            pixV = np.rint(self.v[nb] / self.UVpixsize).flatten().astype(np.int32)
            # Filter out visibility samples that fall outside the field of view
            # (half-plane).
            goodpix = np.where(
                np.logical_and(np.abs(pixU) < self.Nphf, np.abs(pixV) < self.Nphf)
            )[0]
            # added to introduce Atmospheric Errors
            phase_nb = np.angle(self.Gains[nb, goodpix])
            self.baseline_phases[nb] = phase_nb
            # Calculate positive and negative pixel indices
            # (accounting for the shift in the FFT).
            # Isolates positives and negative pixels
            pU = pixU[goodpix] + self.Nphf
            pV = pixV[goodpix] + self.Nphf
            mU = -pixU[goodpix] + self.Nphf
            mV = -pixV[goodpix] + self.Nphf
            if antidx != -1:
                # If we are only updating baselines for a single antenna, subtract the old
                # contributions of the baseline from the total sampling and gains.
                # subtracting previous gains and sampling contributions based on
                # stored positions
                self.totsampling[self.pixpos[nb][1], self.pixpos[nb][2]] -= 1.0
                self.totsampling[self.pixpos[nb][3], self.pixpos[nb][0]] -= 1.0
                self.Gsampling[self.pixpos[nb][1], self.pixpos[nb][2]] -= self.Gains[
                    nb, goodpix
                ]
                self.Gsampling[self.pixpos[nb][3], self.pixpos[nb][0]] -= np.conjugate(
                    self.Gains[nb, goodpix]
                )
                self.noisemap[self.pixpos[nb][1], self.pixpos[nb][2]] -= self.Noise[
                    nb, goodpix
                ] * np.abs(self.Gains[nb, goodpix])
                self.noisemap[self.pixpos[nb][3], self.pixpos[nb][0]] -= np.conjugate(
                    self.Noise[nb, goodpix]
                ) * np.abs(self.Gains[nb, goodpix])
            # updated pixel positions for current baseline
            self.pixpos[nb] = [np.copy(pU), np.copy(pV), np.copy(mU), np.copy(mV)]
            # Iterate over the good pixels for the current baseline and update:
            for pi, gp in enumerate(goodpix):
                # computes the absolute gains for the current baseline
                gabs = np.abs(self.Gains[nb, gp])
                pVi = pV[pi]
                mUi = mU[pi]
                mVi = mV[pi]
                pUi = pU[pi]
                # Update the sampling counts, gains, and noise at the corresponding pixel
                # locations in the UV grid.
                self.totsampling[pVi, mUi] += 1.0
                self.totsampling[mVi, pUi] += 1.0
                self.Gsampling[pVi, mUi] += self.Gains[nb, gp]
                self.Gsampling[mVi, pUi] += np.conjugate(self.Gains[nb, gp])
                self.noisemap[pVi, mUi] += self.Noise[nb, gp] * gabs
                self.noisemap[mVi, pUi] += np.conjugate(self.Noise[nb, gp]) * gabs
        # Calculate a robustness factor based on the total sampling and a
        # user-defined parameter.
        self.robfac = (
            (5.0 * 10.0 ** (-self.robust)) ** 2.0
            * (2.0 * self.Nbas * self.nH)
            / np.sum(self.totsampling**2.0)
        )

    def _set_beam(self):
        """
        This method calculates the "dirty beam" of the interferometer, which is the
         Fourier transform of the weighted sampling distribution in the UV plane.
         It first calculates a denominator for robust weighting,
        which is used to balance data points with varying noise and sampling density.
        The denominator is calculated as 1 plus the product of a robustness factor and
         the total sampling.
        Then the method applies robust weighting to the total sampling, gains,
        and noise by dividing each of
        these quantities by the calculated denominator.
        The results are stored in the instance variables `robustsamp`,
         `Grobustsamp`, and `GrobustNoise`, respectively.
        The method then calculates the dirty beam by performing a
         2D inverse Fourier transform
        on the weighted sampling distribution. The zero-frequency component of the
         weighted sampling is shifted to the center before the Fourier transform and
          shifted back to the original corner after the Fourier transform.
           The real part of the result is extracted and normalized by a factor
            determined by `W2W1`. The result is stored in the instance variable `beam`.
        Finally, the method scales and normalizes the beam by dividing it by
         its maximum value within a central region.
        The maximum value is found and stored in the instance variable `beamScale`,
        and the beam is then divided by this value.
        The function modifies the following instance variables:
        - self.robustsamp: The weighted sampling distribution in the UV plane.
        - self.Grobustsamp: The weighted gains in the UV plane.
        - self.GrobustNoise: The weighted noise in the UV plane.
        - self.beam: The dirty beam of the interferometer.
        - self.beamScale: The maximum value of the beam within a central region.
        """
        # 1. Robust Weighting Calculation:
        #   - denom: Denominator used for robust weighting to balance data
        #      points with varying noise and sampling density.
        denom = 1.0 + self.robfac * self.totsampling
        # 2. Apply Robust Weighting to Sampling, Gains, and Noise:
        #   - robustsamp: Weighted sampling distribution in the UV plane.
        self.robustsamp[:] = self.totsampling / denom
        self.Grobustsamp[:] = self.Gsampling / denom
        self.GrobustNoise[:] = self.noisemap / denom
        # 3. Dirty Beam Calculation:
        #   - np.fft.fftshift(self.robustsamp): Shift the zero-frequency component
        #       of the weighted sampling to the center for FFT.
        #   - np.fft.ifft2(...): Perform the 2D inverse Fourier Transform to
        #       get the dirty beam in the image domain.
        #   - np.fft.ifftshift(...): Shift the zero-frequency component back
        #       to the original corner.
        #   - .real: Extract the real part of the complex result, as the beam
        #       is a real-valued function.
        #   -  / (1. + self.W2W1): Normalize the beam by a factor determined by `W2W1`.
        self.beam[:] = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(self.robustsamp))
        ).real / (1.0 + self.W2W1)
        # 4. Beam Scaling and Normalization:
        #   - Find the maximum value of the beam within a central region
        #       (likely to avoid edge effects).
        self.beamScale = np.max(
            self.beam[self.Nphf : self.Nphf + 1, self.Nphf : self.Nphf + 1]
        )
        self.beam[:] /= self.beamScale

    def _check_lfac(self):
        """
        This method checks and adjusts the scale factor (`lfac`)
         used for the UV plane coordinates
        and updates the labels (`ulab` and `vlab`) accordingly.
        The function first calculates a metric `mw`
        based on the maximum X coordinate (`Xmax`),
        the third element of the wavelength array, and the current scale factor (`lfac`).
        If `mw` is less than 0.1 and the current scale factor is 1.e6 (indicating
         that the UV plane coordinates
        are currently in Mλ), the function changes the scale factor to 1.e3
         (to switch the UV plane coordinates to kλ)
        and updates the labels accordingly.

        If `mw` is greater than or equal to 100 and the current scale factor is 1.e3
         (indicating that the UV
        plane coordinates are currently in kλ), the function changes the scale factor
         to 1.e6 (to switch the UV
        plane coordinates to Mλ) and updates the labels accordingly.

        The function modifies the following instance variables:
        - self.lfac: The scale factor used for the UV plane coordinates.
        - self.ulab: The label for the U coordinate in the UV plane.
        - self.vlab: The label for the V coordinate in the UV plane.
        """
        mw = 2.0 * self.Xmax / self.wavelength[2] / self.lfac
        if mw < 0.1 and self.lfac == 1.0e6:
            self.lfac = 1.0e3
            self.ulab = r"U (k$\lambda$)"
            self.vlab = r"V (k$\lambda$)"
        elif mw >= 100.0 and self.lfac == 1.0e3:
            self.lfac = 1.0e6
            self.ulab = r"U (M$\lambda$)"
            self.vlab = r"V (M$\lambda$)"

    def _prepare_model(self):
        self.modelim = [
            np.zeros((self.Npix, self.Npix), dtype=np.float32) for i in [0, 1]
        ]
        self.modelimTrue = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        dims = np.shape(self.img)
        d1 = self.img.shape[0]
        if d1 == self.Nphf:
            sh0 = (self.Nphf - dims[0]) // 2
            sh1 = (self.Nphf - dims[1]) // 2
            self.modelimTrue[
                sh0 + self.Np4 : sh0 + self.Np4 + dims[0],
                sh1 + self.Np4 : sh1 + self.Np4 + dims[1],
            ] += self.zoomimg
        else:
            zoomimg = spndint.zoom(self.img, float(self.Nphf) / d1)
            zdims = np.shape(zoomimg)
            zd0 = min(zdims[0], self.Nphf)
            zd1 = min(zdims[1], self.Nphf)
            sh0 = (self.Nphf - zdims[0]) // 2
            sh1 = (self.Nphf - zdims[1]) // 2
            self.modelimTrue[
                sh0 + self.Np4 : sh0 + self.Np4 + zd0,
                sh1 + self.Np4 : sh1 + self.Np4 + zd1,
            ] += zoomimg[:zd0, :zd1]

        self.modelimTrue[self.modelimTrue < 0.0] = 0.0

    # def _set_elliptical_beam(self):
    #    cov_matrix = np.cov(self.u, self.v)
    #    # Eigen decomposition of the covariance matrix
    #    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    #    # Eigenvector corresponding to the largest eigenvalue gives the major axis
    #    # direction
    #    major_axis_vector = eigvecs[:, np.argmax(eigvals)]
    #    BPA_rad = self.rad2deg(np.arctan2(major_axis_vector[1], major_axis_vector[0]))
    #    scale_factor = (
    #        1220 * self.deg2arcsec * self.wavelength[2] / self.Diameters[0] / 2.3548
    #    )
    #    # Rotate the coordinates
    #    x_rot = x * np.cos(BPA_rad) + y * np.sin(BPA_rad)
    #    y_rot = -x * np.sin(BPA_rad) + y * np.cos(BPA_rad)
    #    # Eigenvalues correspond to the variances along the major and minor axes
    #    sigma_major = np.sqrt(np.max(eigvals)) * scale_factor
    #    sigma_minor = np.sqrt(np.min(eigvals)) * scale_factor
    #    PB = (x_rot / sigma_major) ** 2 + (y_rot / sigma_minor) ** 2
    #    self.beamImg = np.exp(self.distmat / PB)

    def _set_primary_beam(self):
        """
        Calculates and applies the primary beam of the telescope to the model image.

        The primary beam models the sensitivity pattern of the antenna, which
        decreases as you move away from the center of the field of view.
        """

        # 1. Primary Beam Calculation:
        #   - Calculates the primary beam width (PB) in units of pixels squared.
        #   - The formula is derived from the Airy disk pattern
        #   for a circular aperture (telescope dish).
        #   - Key factors:
        #       - 1220:  Scaling factor related to the Airy disk.
        #       - 180/np.pi * 3600: Conversion from radians to arcseconds.
        #       - self.wavelength[2]: Observing wavelength at the center of the channel.
        #       - self.Diameters[0]: Diameter of the primary reflector (antenna dish).
        #       - 2.3548: Factor related to the full-width-half-maximum
        #           (FWHM) of the Airy disk.
        PB = (
            2.0
            * (
                1220.0
                * 180.0
                / np.pi
                * 3600.0
                * self.wavelength[2]
                / self.Diameters[0]
                / 2.3548
            )
            ** 2.0
        )

        # BMAJ  # Beam FWHM along major axis [deg]
        # BMIN # Beam FWHM along minor axis [deg]
        # BPA # Beam position angle [deg]
        self.header.append(
            ("BMAJ", 180.0 * np.sqrt(PB) / np.pi, "Beam FWHM along major axis [deg]")
        )
        self.header.append(
            ("BMIN", 180.0 * np.sqrt(PB) / np.pi, "Beam FWHM along minor axis [deg]")
        )
        self.header.append(("BPA", 0.0, "Beam position angle [deg]"))
        # 2. Create Primary Beam Image:
        #   - Creates a 2D Gaussian image (`beamImg`) representing the primary beam.
        #   - self.distmat: Pre-calculated matrix of squared
        #           distances from the image center.
        #   - np.exp(self.distmat / PB): Calculates
        #       the Gaussian profile of the beam based on
        #            the distances and primary beam width.
        beamImg = np.exp(self.distmat / PB)

        # 3. Apply Primary Beam to Model:
        #   - Multiplies the original model image (`self.modelimTrue`)
        #       by the primary beam image (`beamImg`).
        #   - This effectively attenuates the model image towards the edges,
        #       simulating the telescope's sensitivity pattern.
        #   - The result is stored in the first element of the `self.modelim` list
        self.modelim[0][:] = self.modelimTrue * beamImg

        # 4. Calculate Model FFT:
        #   - Computes the 2D Fast Fourier Transform (FFT) of the primary
        #       beam-corrected model image.
        #   - np.fft.fftshift(...): Shifts the zero-frequency component to the
        #  center of the array before the FFT, as required for correct FFT interpretation.
        self.modelfft = np.fft.fft2(np.fft.fftshift(self.modelim[0]))

    def _observe(self):
        """
        Simulates the observation process of the interferometer,
            generating a 'dirty map' and 'dirty visibilities.'

        The dirty map is the image obtained directly from the observed
             visibilities without any deconvolution,
        and the dirty visibilities are the Fourier transform of the dirty map.
        """

        # 1. Calculate Dirty Map:
        #   - np.fft.ifftshift(self.GrobustNoise), np.fft.ifftshift(self.Grobustsamp):
        #  Shift the zero-frequency components to the corners before inverse FFT.
        #   - self.modelfft * np.fft.ifftshift(self.Grobustsamp): Element-wise
        #       multiplication of the model FFT and the shifted weighted sampling to
        #       incorporate the effect of the instrument.
        #   - ... + self.modelfft * ... : Add the complex noise to the model's
        #            visibility after scaling by the robust sampling
        #           to obtain the observed visibilities.
        #   - np.fft.ifft2(...): Perform the 2D inverse Fast Fourier Transform (IFFT)
        #           on the combined visibilities (shifted noise + weighted model).
        #   - np.fft.fftshift(...): Shift the zero-frequency component back to the center.
        #   - .real: Extract the real part of the IFFT result to get the dirty map, which
        #       is a real-valued image.
        #   - / (1. + self.W2W1): Normalize the dirty map by a factor related to the
        #  weighting scheme (`W2W1`).
        self.dirtymap[:] = (
            np.fft.fftshift(
                np.fft.ifft2(
                    np.fft.ifftshift(self.GrobustNoise)
                    + self.modelfft * np.fft.ifftshift(self.Grobustsamp)
                )
            )
        ).real / (1.0 + self.W2W1)

        # 2. Normalize Dirty Map:
        #   - Divide the dirty map by the beam scale factor (`self.beamScale`)
        #       calculated earlier in `_set_beam`.
        #   - This normalization ensures that the peak brightness in the dirty map
        #        is consistent with the beam's peak intensity.
        self.dirtymap /= self.beamScale

        # 3. Correct Negative Values in Dirty Map (Optional):
        #   - Find the minimum value in the dirty map.
        min_dirty = np.min(self.dirtymap)
        #   - If there are negative values, shift the whole dirty map
        #        upwards to make all values non-negative.
        #   - This step might be necessary to avoid issues with certain
        #            image display or processing algorithms.
        if min_dirty < 0.0:
            self.dirtymap += np.abs(min_dirty)
        else:
            self.dirtymap -= min_dirty

        # 4. Calculate Model and Dirty Visibilities:
        #   - modelvis: Shift the zero-frequency component of the
        #        model's Fourier transform to the center.
        self.modelvis = np.fft.fftshift(
            self.modelfft
        )  # Already calculated in _set_primary_beam
        #   - dirtyvis: Shift the zero-frequency component of the dirty
        #         visibilities (shifted noise + weighted model) to the center.
        self.dirtyvis = np.fft.fftshift(
            np.fft.ifftshift(self.GrobustNoise)
            + self.modelfft * np.fft.ifftshift(self.Grobustsamp)
        )

    def _update_cubes(self):
        self.modelCube[self.channel] = self.modelim[0]
        self.dirtyCube[self.channel] = self.dirtymap
        self.visCube[self.channel] = self.modelvis
        self.dirtyvisCube[self.channel] = self.dirtyvis

    # ------------ Noise Functions ------------------------

    def _add_atmospheric_noise(self):
        for nb in self.bas2change:
            phase_rms = np.std(
                self.baseline_phases[nb]
            )  # Standard deviation is RMS for phases
            random_phase_error = np.random.normal(scale=phase_rms)
            self.Gains[nb] *= np.exp(1j * random_phase_error)

    def _add_thermal_noise(self):
        mean_val = np.mean(self.img)
        # self.img += np.random.normal(scale=mean_val / self.snr)
        return mean_val

    # ------------------- IO Functions
    def _savez_compressed_cubes(self):
        min_dirty = np.min(self.dirtyCube)
        if min_dirty < 0:
            self.dirtyCube += min_dirty
        max_dirty = np.sum(self.dirtyCube)
        max_clean = np.sum(self.modelCube)
        self.dirtyCube = self.dirtyCube / max_dirty
        self.dirtyCube = self.dirtyCube * max_clean
        if self.save_mode == "npz":
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "clean-cube_{}.npz".format(str(self.idx))
                ),
                self.modelCube,
            )
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "dirty-cube_{}.npz".format(str(self.idx))
                ),
                self.dirtyCube,
            )
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "dirty-vis-cube_{}.npz".format(str(self.idx))
                ),
                self.dirtyvisCube,
            )
            np.savez_compressed(
                os.path.join(
                    self.output_dir, "clean-vis-cube_{}.npz".format(str(self.idx))
                ),
                self.visCube,
            )
        elif self.save_mode == "h5":
            with h5py.File(
                os.path.join(self.output_dir, "clean-cube_{}.h5".format(str(self.idx))),
                "w",
            ) as f:
                f.create_dataset("clean_cube", data=self.modelCube)
            with h5py.File(
                os.path.join(self.output_dir, "dirty-cube_{}.h5".format(str(self.idx))),
                "w",
            ) as f:
                f.create_dataset("dirty_cube", data=self.dirtyCube)
            with h5py.File(
                os.path.join(
                    self.output_dir, "dirty-vis-cube_{}.h5".format(str(self.idx))
                ),
                "w",
            ) as f:
                f.create_dataset("dirty_vis_cube", data=self.dirtyvisCube)
            with h5py.File(
                os.path.join(
                    self.output_dir, "clean-vis-cube_{}.h5".format(str(self.idx))
                ),
                "w",
            ) as f:
                f.create_dataset("clean_vis_cube", data=self.visCube)
        elif self.save_mode == "fits":
            self.clean_header = self.header
            self.clean_header.append(("DATAMAX", np.max(self.modelCube)))
            self.clean_header.append(("DATAMIN", np.min(self.modelCube)))
            hdu = fits.PrimaryHDU(header=self.clean_header, data=self.modelCube)
            hdu.writeto(
                os.path.join(
                    self.output_dir, "clean-cube_{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            self.dirty_header = self.header
            self.dirty_header.append(("DATAMAX", np.max(self.dirtyCube)))
            self.dirty_header.append(("DATAMIN", np.min(self.dirtyCube)))
            hdu = fits.PrimaryHDU(header=self.dirty_header, data=self.dirtyCube)
            hdu.writeto(
                os.path.join(
                    self.output_dir, "dirty-cube_{}.fits".format(str(self.idx))
                ),
                overwrite=True,
            )
            real_part = np.real(self.dirtyvisCube)
            imag_part = np.imag(self.dirtyvisCube)
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
            real_part = np.real(self.visCube)
            imag_part = np.imag(self.visCube)
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
        self.terminal.add_log(
            f"Total Flux detected in model cube: {round(np.sum(self.modelCube), 2)} Jy"
        )
        self.terminal.add_log(
            f"Total Flux detected in dirty cube: {round(np.sum(self.dirtyCube), 2)} Jy"
        )

    def _free_space(self):
        del self.modelCube
        del self.dirtyCube
        del self.dirtyvisCube
        del self.visCube
