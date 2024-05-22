import os
import time
import sys
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.cm as cm
import matplotlib.image as plimg
import scipy.ndimage.interpolation as spndint
from astropy.constants import c
import astropy.units as U
from tqdm import tqdm
import astropy.time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.constants import M_earth, R_earth, G
import pandas as pd
from scipy.integrate import odeint
from astropy.time import Time

def showError(message):
        raise Exception(message)

class Interferometer():

    def __init__(self, idx, skymodel, main_dir, 
                output_dir, ra, dec, central_freq, band_range, 
                fov, antenna_array, noise, int_time, obs_date, 
                robust=0.5):
        self.idx = idx
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
        self.plot_dir = os.path.join(output_dir, 'plots')
        # Parameters
        self.Hfac = np.pi / 180. * 15.
        self.deg2rad = np.pi / 180.
        self.rad2deg = 180. / np.pi
        self.second2hour = 1. / 3600.
        self.curzoom = [0, 0, 0, 0]
        self.robust = robust
        self.deltaAng = 1. * self.deg2rad
        self.gamma = 0.5
        self.lfac = 1.e6
        # PLACEHOLDER MUST BE SUBSTITUTED WITH REAL NUMBER OF SCANS 
        self.nH = int(self.int_time / (6 * self.second2hour))
        #if self.nH > 200: 
        #    self.nH = int(self.int_time / (8.064 * self.second2hour))
        #if self.nH > 200:
        #    self.nH = int(self.int_time / 18.144 * self.second2hour)
        #if self.nH > 200:
        #    self.nH = int(self.int_time / 30.24 * self.second2hour)
        print(f'Number of scans: {self.nH}')
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
        self.Xaxmax = self.imsize / 2.
        self.Npix = skymodel.shape[1]
        self.Np4 = self.Npix // 4
        self.Nchan = skymodel.shape[0]
        self.Nphf = self.Npix // 2
        self.pixsize = float(self.imsize) / self.Npix
        self.xx = np.linspace(-self.imsize / 2., self.imsize / 2., self.Npix)
        self.yy = np.ones(self.Npix, dtype=np.float32)
        self.distmat = (-np.outer(self.xx**2., self.yy) - np.outer(self.yy, self.xx**2.)) * self.pixsize**2.
        self.robfac = 0.0
        self.W2W1 = 1     
        self.currcmap = cm.jet
        self.zooming = 0
        
        # Get the antenna coordinates, and the hour angle coverage
        self._get_observing_location()
        # This function must be checked
        self._get_Hcov()
        self._read_antennas()
        # Get the observing wavelengths for each channel 
        self._get_wavelengths()
        self._prepare_cubes()
        
        for channel in tqdm(range(self.Nchan)):
            self.channel = channel
            self._get_channel_range()
            self._prepare_2d_arrays()
            self._prepare_baselines()
            self.set_noise()
            self._set_baselines()
            self._grid_uv()
            self._set_beam()
            self._check_lfac()
            if self.channel == self.Nchan // 2:
                self._plot_beam()
                self._plot_antennas()
                self._plot_uv_coverage()
            self.img = skymodel[channel]
            self._prepare_model()
            self._set_primary_beam()
            self._observe()
            self._update_cubes()
        self._savez_compressed_cubes()
        self._plot_sim()
        self._free_space()

   
    def _get_observing_location(self):
        self.observing_location = EarthLocation.of_site('ALMA')

    def _get_Hcov(self):
        self.int_time = self.int_time * U.s
        start_time = Time(self.obs_date + 'T00:00:00', format='isot', scale='utc')
        midle_time = start_time + self.int_time / 2
        end_time = start_time + self.int_time
        sidereal_start = start_time.sidereal_time('mean', longitude=self.lat)
        sidereal_middle = midle_time.sidereal_time('mean', longitude=self.lat)
        sidereal_end = end_time.sidereal_time('mean', longitude=self.lat)
        #self.start_time = sidereal_time
        #self.middle_time = sidereal_time + self.int_time.to(U.hourangle) / 2
        #self.end_time = sidereal_time.value + self.int_time.to(U.hournangle)
        start = (sidereal_middle - sidereal_start)
        end = (sidereal_end - sidereal_middle)
        self.Hcov = [-start.value, end.value]
        
    def _get_az_el(self):
        self._get_observing_location()
        self._get_middle_time()
        sky_coords = SkyCoord(ra=self.ra * self.rad2deg, dec=self.dec * self.rad2deg, unit='deg')
        aa = AltAz(location=self.observing_location, obstime=self.middle_time)
        sky_coords.transform_to(aa)
        self.az = sky_coords.az
        self.el = sky_coords.alt
    
    def _get_initial_and_final_H(self):

        def differential_equations(y, t,phi, A, E, forward=True):
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
                return - dtau_dt(t)

            A, E = y
            if forward == True: 
                dA_dt = dtau_dt(t) * ((np.sin(phi) * np.cos(E) - np.cos(phi) * np.sin(E) * np.cos(A)) / np.cos(E))
                dE_dt = dtau_dt(t) * (np.cos(phi) * np.sin(A))
            else: 
                dA_dt = dtau_dt_reversed(t) * ((np.sin(phi) * np.cos(E) - np.cos(phi) * np.sin(E) * np.cos(A)) / np.cos(E))
                dE_dt = dtau_dt_reversed(t) * (np.cos(phi) * np.sin(A))
            return [dA_dt, dE_dt]

        self._get_az_el()
        y0 = [self.az, self.el]
        middle_H = self.el.to(U.hourangle).value
        t_final_solve = np.linspace(self.middle_time, self.end_time, self.nH)
        sol_final = odeint(differential_equations, y0, t_final_solve, args=(self.lat, self.az, self.el))
        az_final = sol_final[-1, 0]
        el_final = sol_final[-1, 1]
        t_initial_solve = np.linspace(self.middle_time, self.start_time , self.nH)[::-1]
        sol_initial = odeint(differential_equations, y0, t_initial_solve, args=(self.lat, self.az, self.el, False))
        az_initial = sol_initial[-1, 0]
        el_initial = sol_initial[-1, 1]
        aa_initial = AltAz(az=az_initial, alt=el_initial, location=self.observing_location, obstime=self.start_time)
        aa_final = AltAz(az=az_final, alt=el_final, location=self.observing_location, obstime=self.end_time)
        self.initial_H =  - aa_initial.alt.to(U.hourangle).value 
        self.final_H = - aa_final.alt.to(U.hourangle).value
        print(self.initial_H, self.final_H)
  
    def _hz_to_m(self, freq):
        return self.c_ms / freq

    def _read_antennas(self):
        antenna_coordinates = pd.read_csv(os.path.join(self.main_dir, 'antenna_config', 'antenna_coordinates.csv'))
        obs_antennas = self.antenna_array.split(' ')
        obs_antennas = [antenna.split(':')[0] for antenna in obs_antennas]
        obs_coordinates = antenna_coordinates[antenna_coordinates['name'].isin(obs_antennas)]
        antenna_coordinates = obs_coordinates[['x', 'y']].values
        antPos = []
        Xmax = 0.0
        # PLACEHOLDER MUSTE BE SUBSTITUTED WITH REAL TIME RANGE
        #Hcov = [-3.0 * self.Hfac, 3.0 * self.Hfac]
        for line in antenna_coordinates:
            antPos.append([line[0] * 1e-3, line[1] * 1e-3])
            Xmax = np.max(np.abs(antPos[-1] + [Xmax]))
        self.Xmax = Xmax
        self.antPos = antPos
        #self.Hcov = Hcov
        cosW = -np.tan(self.lat) * np.tan(self.dec)
        if np.abs(cosW) < 1.0:
            Hhor = np.arccos(cosW)
        elif np.abs(lat - dec) > np.pi / 2.:
            Hhor = 0
        else:
            Hhor = np.pi
        
        if Hhor > 0.0:
            if self.Hcov[0] < -Hhor:
                self.Hcov[0] = -Hhor
            if self.Hcov[1] > Hhor:
                self.Hcov[1] = Hhor

        self.Hmax = Hhor
        H = np.linspace(self.Hcov[0], self.Hcov[1],
                        self.nH)[np.newaxis, :]
        
        self.Xmax = self.Xmax * 1.5 
        self.Nant = len(self.antPos)
    
    def _get_wavelengths(self):
        self.w_max, self.w_min = [self._hz_to_m(freq) for freq in [self.central_freq - self.band_range / 2, self.central_freq + self.band_range / 2]]
        waves = np.linspace(self.w_min, self.w_max, self.Nchan + 1)
        obs_wavelengths= np.array([[waves[i], waves[i + 1] ] for i in range(len(waves) - 1)])
        self.obs_wavelengths = obs_wavelengths
    
    def _get_channel_range(self):
        wavelength = list(self.obs_wavelengths[self.channel] * 1e-3)
        wavelength.append((wavelength[0] + wavelength[1]) / 2.)
        self.wavelength = wavelength
        self.fmtB1 = r'$\lambda = $ %4.1fmm  ' % (self.wavelength[2] * 1.e6)
        self.fmtB = self.fmtB1 + "\n" + r'% 4.2f Jy/beam' + "\n" + r'$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f '   

    def _prepare_2d_arrays(self):
        self.beam = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.totsampling = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.dirtymap = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.noisemap = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.robustsamp = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.Gsampling = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.Grobustsamp = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.GrobustNoise = np.zeros((self.Npix, self.Npix),dtype=np.complex64)
    
    def _prepare_cubes(self):
        self.modelCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.float32)
        self.dirtyCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.float32)
        self.visCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.complex64)
        self.dirtyvisCube = np.zeros((self.Nchan, self.Npix, self.Npix), dtype=np.complex64)

    def _update_cubes(self):
        self.modelCube[self.channel] = self.modelim[0]
        self.dirtyCube[self.channel] = self.dirtymap
        self.visCube[self.channel] = self.modelvis
        self.dirtyvisCube[self.channel] = self.dirtyvis

    def _prepare_baselines(self):
        self.Nbas = self.Nant * (self.Nant - 1) // 2
        NBmax = self.Nbas
        self.B = np.zeros((NBmax, self.nH), dtype=np.float32)
        self.basnum = np.zeros((self.Nant, self.Nant - 1), dtype=np.int8)
        self.basidx = np.zeros((self.Nant, self.Nant), dtype=np.int8)
        self.antnum = np.zeros((NBmax, 2), dtype=np.int8)
        self.Gains = np.ones((self.Nbas, self.nH), dtype=np.complex64)
        self.Noise = np.zeros((self.Nbas, self.nH), dtype=np.complex64)
        self.Horig = np.linspace(self.Hcov[0], self.Hcov[1], self.nH)
        H = self.Horig[np.newaxis, :]
        self.H = [np.sin(H), np.cos(H)]
        bi = 0
        nii = [0 for n in range(self.Nant)]
        for n1 in range(self.Nant - 1):
            for n2 in range(n1 + 1, self.Nant):
                self.basnum[n1, nii[n1]] = np.int8(bi)
                self.basnum[n2, nii[n2]] = np.int8(bi)
                self.basidx[n1, n2] = np.int8(bi)
                self.antnum[bi] = [n1, n2]
                nii[n1] += 1
                nii[n2] += 1
                bi += np.int8(1)
        self.u = np.zeros((NBmax, self.nH))
        self.v = np.zeros((NBmax, self.nH))
        self.ravelDims = (NBmax, self.nH)

    def set_noise(self):
        if self.noise == 0.0:
            self.Noise[:] = 0.0
        else:
            # Inject in the noise array a random distribution 
            # with mean 0 and standard deviation equal to the noise
            # The distribution in the complex domain is scaled to the 
            # imaginary unit
            self.Noise[:] = np.random.normal(
                loc=0.0, scale=self.noise, size=np.shape(
                    self.Noise)) + 1.j * np.random.normal(
                        loc=0.0, scale=self.noise, size=np.shape(self.Noise))

    def _set_baselines(self, antidx=-1):
        if antidx == -1:
                bas2change = range(self.Nbas)
        elif antidx < self.Nant:
            bas2change = self.basnum[antidx].flatten()
        else:
            bas2change = []
        for currBas in bas2change:
            n1, n2 = self.antnum[currBas]
            self.B[currBas, 0] = -(self.antPos[n2][1] - self.antPos[n1][1]) \
                                 * self.trlat[0] / self.wavelength[2]
            self.B[currBas, 1] = (self.antPos[n2][0] - self.antPos[n1][0]) \
                                 / self.wavelength[2]
            self.B[currBas, 2] = (self.antPos[n2][1] - self.antPos[n1][1]) \
                                 * self.trlat[1] / self.wavelength[2]
            self.u[currBas, :] = -(self.B[currBas, 0] * self.H[0] + self.B[currBas, 1] * self.H[1])
            self.v[currBas, :] = -self.B[currBas, 0] * self.trdec[0] * self.H[1] \
                                 + self.B[currBas, 1] * self.trdec[0] * self.H[0] \
                                 + self.trdec[1] * self.B[currBas, 2]
    
    def _grid_uv(self, antidx=-1):
        
        if antidx == -1:
            bas2change = range(self.Nbas)
            self.pixpos = [[] for nb in bas2change]
            self.totsampling[:] = 0.0
            self.Gsampling[:] = 0.0
            self.noisemap[:] = 0.0
        elif antidx < self.Nant:
            bas2change = list(map(int, list(self.basnum[antidx].flatten())))
        else:
            bas2change = []
        # set the pixsize in the UV plane
        self.UVpixsize = 2. / (self.imsize * np.pi / 180. / 3600.)
        self.baseline_phases = {}
        self.bas2change = bas2change
        for nb in self.bas2change:
            # Computes pixel positions in the UV plane 
            pixU = np.rint(self.u[nb] / self.UVpixsize).flatten().astype(
                np.int32)
            pixV = np.rint(self.v[nb] / self.UVpixsize).flatten().astype(
                np.int32)
            # select pixels within the half field 
            goodpix = np.where(
                np.logical_and(
                    np.abs(pixU) < self.Nphf,
                    np.abs(pixV) < self.Nphf))[0]
            # added to introduce Atmospheric Errors 
            phase_nb = np.angle(self.Gains[nb, goodpix])
            self.baseline_phases[nb] = phase_nb
            # Isolates positives and negative pixels 
            pU = pixU[goodpix] + self.Nphf
            pV = pixV[goodpix] + self.Nphf
            mU = -pixU[goodpix] + self.Nphf
            mV = -pixV[goodpix] + self.Nphf
            if antidx != -1:
                # subtracting previous gains and sampling contributions based on
                # stored positions 
                self.totsampling[self.pixpos[nb][1], self.pixpos[nb][2]] -= 1.0
                self.totsampling[self.pixpos[nb][3], self.pixpos[nb][0]] -= 1.0
                self.Gsampling[self.pixpos[nb][1], 
                               self.pixpos[nb][2]] -= self.Gains[nb, goodpix]
                self.Gsampling[self.pixpos[nb][3], 
                               self.pixpos[nb][0]] -= np.conjugate(
                                   self.Gains[nb, goodpix])
                self.noisemap[self.pixpos[nb][1], self.pixpos[nb]
                              [2]] -= self.Noise[nb, goodpix] * np.abs(
                                  self.Gains[nb, goodpix])
                self.noisemap[self.pixpos[nb][3], 
                              self.pixpos[nb][0]] -= np.conjugate(
                                  self.Noise[nb, goodpix]) * np.abs(
                                      self.Gains[nb, goodpix])
            # updated pixel positions for current baseline 
            self.pixpos[nb] = [
                np.copy(pU),
                np.copy(pV),
                np.copy(mU),
                np.copy(mV)
            ]
            for pi, gp in enumerate(goodpix):
                # computes the absolute gains for the current baseline
                gabs = np.abs(self.Gains[nb, gp])
                pVi = pV[pi]
                mUi = mU[pi]
                mVi = mV[pi]
                pUi = pU[pi]
                self.totsampling[pVi, mUi] += 1.0
                self.totsampling[mVi, pUi] += 1.0
                self.Gsampling[pVi, mUi] += self.Gains[nb, gp]
                self.Gsampling[mVi, pUi] += np.conjugate(self.Gains[nb, gp])
                self.noisemap[pVi, mUi] += self.Noise[nb, gp] * gabs
                self.noisemap[mVi, pUi] += np.conjugate(
                    self.Noise[nb, gp]) * gabs
        self.robfac = (5. * 10.**(-self.robust))**2. * (
            2. * self.Nbas * self.nH) / np.sum(self.totsampling**2.)

    def _add_atmospheric_noise(self):
        for nb in self.bas2change:
            phase_rms = np.std(self.baseline_phases[nb]) # Standard deviation is RMS for phases
            random_phase_error = np.random.normal(scale=phase_rms)
            self.Gains[nb] *= np.exp(1j * random_phase_error)

    def _add_thermal_noise(self):
        mean_val = np.mean(self.img)
        #self.img += np.random.normal(scale=mean_val / self.snr)

    def _set_beam(self):
        denom = 1. + self.robfac * self.totsampling
        self.robustsamp[:] = self.totsampling / denom
        self.Grobustsamp[:] = self.Gsampling / denom
        self.GrobustNoise[:] = self.noisemap / denom
        self.beam[:] = np.fft.ifftshift(
            np.fft.ifft2(np.fft.fftshift(
                self.robustsamp))).real / (1. + self.W2W1)
        self.beamScale = np.max(self.beam[self.Nphf:self.Nphf +1, self.Nphf:self.Nphf + 1])
        self.beam[:] /= self.beamScale

    def _plot_beam(self):
        beamPlot = plt.figure(figsize=(8, 8))
        beamPlotPlot = plt.imshow(
            self.beam[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4],
            picker=True,
            interpolation='nearest',
            cmap=self.currcmap)
        beamText = plt.text(
            0.80,
            0.80,
            self.fmtB % (1.0, 0.0, 0.0),
            bbox=dict(facecolor='white', alpha=0.7))
        plt.ylabel('Dec offset (as)')
        plt.xlabel('RA offset (as)')
        plt.setp(beamPlotPlot,
                extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                        -self.Xaxmax / 2., self.Xaxmax / 2.))
        self.curzoom[0] = (self.Xaxmax / 2., -self.Xaxmax / 2.,
                           -self.Xaxmax / 2., self.Xaxmax / 2.)
        plt.title('DIRTY BEAM')
        plt.colorbar()
        nptot = np.sum(self.totsampling[:])
        beamPlotPlot.norm.vmin = np.min(self.beam)
        beamPlotPlot.norm.vmax = 1.0
        if np.sum(self.totsampling[self.Nphf - 4:self.Nphf + 4, self.Nphf -
                                   4:self.Nphf + 4]) == nptot:
            warn = 'WARNING!\nToo short baselines for such a small image\nPLEASE, INCREASE THE IMAGE SIZE!\nAND/OR DECREASE THE WAVELENGTH'
            beamText.set_text(warn)

        plt.savefig(os.path.join(self.plot_dir, 'beam_{}.png'.format(str(self.idx))))
        plt.close()

    def _plot_antennas(self):
        antPlot = plt.figure(figsize=(8, 8))
        toplot = np.array(self.antPos[:self.Nant])
        antPlotBas = plt.plot([0], [0], '-b')[0]
        antPlotPlot = plt.plot(toplot[:, 0], toplot[:, 1],
                                                    'o',
                                                    color='lime',
                                                    picker=5)[0]
        plt.xlim(-self.Xmax, self.Xmax)
        plt.ylim(-self.Xmax, self.Xmax)
        plt.xlabel('East-West offset (Km)')
        plt.ylabel('North-South offset (Km)')
        plt.title('Antenna Configuration')
        plt.savefig(os.path.join(self.plot_dir, 'antenna_config_{}.png'.format(str(self.idx))))
        plt.close()

    def _check_lfac(self):
        mw = 2. * self.Xmax / self.wavelength[2] / self.lfac
        if mw < 0.1 and self.lfac == 1.e6:
            self.lfac = 1.e3
            self.ulab = r'U (k$\lambda$)'
            self.vlab = r'V (k$\lambda$)'
        elif mw >= 100. and self.lfac == 1.e3:
            self.lfac = 1.e6
            self.ulab = r'U (M$\lambda$)'
            self.vlab = r'V (M$\lambda$)'

    def _plot_uv_coverage(self):
        self.ulab = r'U (k$\lambda$)'
        self.vlab = r'V (k$\lambda$)'
        UVPlot = plt.figure(figsize=(8, 8))
        UVPlotPlot = []
        toplotu = self.u.flatten() / self.lfac
        toplotv = self.v.flatten() / self.lfac
        UVPlotPlot.append(
            plt.plot(toplotu,
                             toplotv,
                             '.',
                             color='lime',
                             markersize=1,
                             picker=2)[0])
        UVPlotPlot.append(
            plt.plot(-toplotu,
                             -toplotv,
                             '.',
                             color='lime',
                             markersize=1,
                             picker=2)[0])
        plt.xlim((2. * self.Xmax / self.wavelength[2] / self.lfac,
                         -2. * self.Xmax / self.wavelength[2] / self.lfac))
        plt.ylim((2. * self.Xmax / self.wavelength[2] / self.lfac,
                         -2. * self.Xmax / self.wavelength[2] / self.lfac))
        plt.xlabel(self.ulab)
        plt.ylabel(self.vlab)
        plt.title('UV Coverage')
        plt.savefig(os.path.join(self.plot_dir, 'uv_coverage_{}.png'.format(str(self.idx))))
        plt.close()
   
    def _prepare_model(self):
        self.modelim = [np.zeros((self.Npix, self.Npix), dtype=np.float32) for i in [0, 1]]
        self.modelimTrue = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        dims = np.shape(self.img)
        d1 = self.img.shape[0]
        if d1 == self.Nphf:
            sh0 = (self.Nphf - dims[0]) // 2
            sh1 = (self.Nphf - dims[1]) // 2
            self.modelimTrue[sh0 + self.Np4:sh0 + self.Np4 + dims[0], sh1 + self.Np4:sh1 +
                                 self.Np4 + dims[1]] += self.zoomimg
        else:
            zoomimg = spndint.zoom(self.img, float(self.Nphf) / d1)
            zdims = np.shape(zoomimg)
            zd0 = min(zdims[0], self.Nphf)
            zd1 = min(zdims[1], self.Nphf)
            sh0 = (self.Nphf - zdims[0]) // 2
            sh1 = (self.Nphf - zdims[1]) // 2
            self.modelimTrue[sh0 + self.Np4:sh0 + self.Np4 + zd0, sh1 + self.Np4:sh1 +
                                 self.Np4 + zd1] += zoomimg[:zd0, :zd1]

        self.modelimTrue[self.modelimTrue < 0.0] = 0.0

    def _set_primary_beam(self):
        PB = 2. * (1220. * 180. / np.pi * 3600. * self.wavelength[2] /
               self.Diameters[0] / 2.3548)**2.
        beamImg = np.exp(self.distmat / PB)
        self.modelim[0][:] = self.modelimTrue * beamImg 
        self.modelfft = np.fft.fft2(np.fft.fftshift(self.modelim[0]))

    def _observe(self):
        self.dirtymap[:] = (np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(self.GrobustNoise) + self.modelfft * np.fft.ifftshift(self.Grobustsamp)))).real / (1. + self.W2W1)
        self.dirtymap /= self.beamScale
        min_dirty = np.min(self.dirtymap)
        if min_dirty < 0.0:
            self.dirtymap += np.abs(min_dirty)
        else:
            self.dirtymap -= min_dirty
        self.modelvis = np.fft.fftshift(self.modelfft)
        self.dirtyvis = np.fft.fftshift(np.fft.ifftshift(self.GrobustNoise) + self.modelfft * np.fft.ifftshift(self.Grobustsamp))

    def _savez_compressed_cubes(self):
        min_dirty = np.min(self.dirtyCube)
        if min_dirty < 0:
            self.dirtyCube += min_dirty
        max_dirty = np.sum(self.dirtyCube)
        max_clean = np.sum(self.modelCube)
        self.dirtyCube = self.dirtyCube / max_dirty 
        self.dirtyCube =  self.dirtyCube * max_clean
        np.savez_compressed(os.path.join(self.output_dir, 'clean-cube_{}.npz'.format(str(self.idx))), self.modelCube)
        np.savez_compressed(os.path.join(self.output_dir, 'dirty-cube_{}.npz'.format(str(self.idx))), self.dirtyCube)
        np.savez_compressed(os.path.join(self.output_dir, 'dirty-vis-cube_{}.npz'.format(str(self.idx))), self.dirtyvisCube)
        np.savez_compressed(os.path.join(self.output_dir, 'clean-vis-cube_{}.npz'.format(str(self.idx))), self.visCube)
        print(f'Total Flux detected in model cube: {round(np.sum(self.modelCube), 2)} Jy')
        print(f'Total Flux detected in dirty cube: {round(np.sum(self.dirtyCube), 2)} Jy')
    
    def _free_space(self):
        del self.modelCube
        del self.dirtyCube
        del self.dirtyvisCube
        del self.visCube

    def _plot_sim(self):
        simPlot, ax = plt.subplots(2, 2, figsize=(12, 12))
        sim_img = np.sum(self.modelCube, axis=0)
        simPlotPlot = ax[0, 0].imshow(
            np.power(
            sim_img[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4], self.gamma),
            picker=True,
            interpolation='nearest',
            vmin=0.0,
            vmax=np.max(sim_img)**self.gamma,
            cmap=self.currcmap)
        plt.setp(simPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.)) 
        ax[0, 0].set_ylabel('Dec offset (as)')
        ax[0, 0].set_xlabel('RA offset (as)')
        totflux = np.sum(sim_img[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4])
        ax[0, 0].set_title('MODEL IMAGE: %.2e Jy' % totflux)
        simPlotPlot.norm.vmin = np.min(sim_img)
        simPlotPlot.norm.vmax = np.max(sim_img)
        dirty_img = np.sum(self.dirtyCube, axis=0)
        dirtyPlotPlot = ax[0, 1].imshow(
            dirty_img[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4],
            picker=True,
            interpolation='nearest')
        plt.setp(dirtyPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.))
        ax[0, 1].set_ylabel('Dec offset (as)')
        ax[0, 1].set_xlabel('RA offset (as)')
        totflux = np.sum(dirty_img[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4])
        ax[0, 1].set_title('DIRTY IMAGE: %.2e Jy' % totflux)
        dirtyPlotPlot.norm.vmin = np.min(dirty_img)
        dirtyPlotPlot.norm.vmax = np.max(dirty_img)
        self.UVmax = self.Npix / 2. / self.lfac * self.UVpixsize
        self.UVSh = -self.UVmax / self.Npix
        toplot = np.sum(np.abs(self.visCube), axis=0)
        mval = np.min(toplot)
        Mval = np.max(toplot)
        dval = (Mval - mval) / 2.
        UVPlotFFTPlot = ax[1, 0].imshow(toplot,
                                        cmap=self.currcmap,
                                        vmin=0.0,
                                        vmax=Mval + dval,
                                        picker=5)
        plt.setp(UVPlotFFTPlot,
                    extent=(-self.UVmax + self.UVSh, self.UVmax + self.UVSh,
                            -self.UVmax - self.UVSh, self.UVmax - self.UVSh))

        ax[1, 0].set_ylabel('V (k$\\lambda$)')
        ax[1, 0].set_xlabel('U (k$\\lambda$)')
        ax[1, 0].set_title('MODEL VISIBILITY')

        toplot = np.sum(np.abs(self.dirtyvisCube), axis=0)
        mval = np.min(toplot)
        Mval = np.max(toplot)
        dval = (Mval - mval) / 2.
        UVPlotDirtyFFTPlot = ax[1, 1].imshow(toplot,
                                        cmap=self.currcmap,
                                        vmin=0.0,
                                        vmax=Mval + dval,
                                        picker=5)
        plt.setp(UVPlotDirtyFFTPlot,
                    extent=(-self.UVmax + self.UVSh, self.UVmax + self.UVSh,
                            -self.UVmax - self.UVSh, self.UVmax - self.UVSh))
        ax[1, 1].set_ylabel('V (k$\\lambda$)')
        ax[1, 1].set_xlabel('U (k$\\lambda$)')
        ax[1, 1].set_title('DIRTY VISIBILITY')
        plt.savefig(os.path.join(self.plot_dir, 'sim_{}.png'.format(str(self.idx))))
        plt.close()

        sim_spectrum = np.sum(self.modelCube, axis=(1, 2))
        dirty_spectrum = np.sum(self.dirtyCube, axis=(1, 2))
        wavelenghts = np.linspace(self.w_min, self.w_max, self.Nchan)
        x_ticks = np.round(wavelenghts, 2)
        specPlot, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].plot(wavelenghts, sim_spectrum)
        ax[0].set_ylabel('Jy/$pix^{2}$')
        ax[0].set_xlabel('$\\lambda$ [mm]')
        ax[0].set_title('MODEL SPECTRUM')
        ax[1].plot(wavelenghts, dirty_spectrum)
        ax[1].set_ylabel('Jy/$pix^{2}$')
        ax[1].set_xlabel('$\\lambda$ [mm]')
        ax[1].set_title('DIRTY SPECTRUM')
        plt.savefig(os.path.join(self.plot_dir, 'spectra_{}.png'.format(str(self.idx))))


