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
import astropy.coordinates as coord

def showError(message):
        raise Exception(message)


class Interferometer(object):

    def __init__(self, idx, master_path, output_path, model, 
                       integration_time, total_time, n_pix, 
                       n_channels,  bandwidth, central_freq, 
                       ra, dec, obs_date, antenna_array, antenna_diameter=12):
        self.master_path = master_path
        self.output_path = output_path
        self.plot_path = os.path.join(output_path, 'plots')
        self.integration_time = integration_time
        self.total_time = total_time
        self.nH = int(self.total_time / self.integration_time)
        self.atenna_diameter = antenna_diameter
        self.n_pix = n_pix
        self.central_freq = central_freq
        self.bandwidth = bandwidth
        self.n_channels = n_channels
        # Constants and conversions
        self.c_ms = c.to(U.m / U.s).value
        self.deg2rad = np.pi / 180.
        self.rad2deg = 180. / np.pi
        self.rad2arcsec = 3600. * self.rad2deg
        self.deltaAng = 1. * self.deg2rad 
        self.lat = 23.017469 * self.deg2rad 
        self.date_str = obs_date
        self.ra = ra
        self.dec = dec
        self._alma_radec_to_dec()
        print(self.Hcov)
        self.trlat [np.sin(self.lat), np.cos(self.lat)]
        self.trdec [np.sin(self.dec), np.cos(self.dec)]
        self.delta_nu = self._hz_to_m(self.bandwidth  / self.n_channels) # channel width in meters
        self.obs_wavelenghts = _get_observing_wavelengths() # observing wavelenghts in meters
        self.gamma = 0.5  # gamma correction to plot model 
        self.lambdafac = 1.e6 # scale factor from km to mm
        self.W2W1 = 1.0 # relative weight of subarrays
        self.hpbw = self._get_half_power_beam_width() # half power beam width in radians
         # coverage in hours, CHECK THIS
        self.antenna_array = antenna_arrays
        self._get_antenna_coords()
        self.fov = self._get_fov()
        self._readAntennas()
        self.model = model
        self.imsize = 1.5 * self.fov
        self.Xaxmax
        self.Nphf = self.Npix // 2
        self.robfac = 0.0
        self.currcmap = cm.jet
        self._prepareCubes()
        for channel in tqdm(range(self.n_channels)):
            self.wavelenghts = self.obs_wavelenghts[channel]
            self._prepareBeam()
            self._prepareBaselines()
            self._setBaselines()
            self._setBeam()
            self.modelim = model[channel]
            xx = np.linspace(-self.imsize / 2., self.imsize / 2., self.Npix)
            yy = np.ones(self.Npix, dtype=np.float32)
            self.distmat = (-np.outer(xx**2., yy) -
                        np.outer(yy, xx**2.)) * pixsize**2.
            self._setPrimaryBeam()
            self._obs_model()
        
            self.modelCube[channel] = self.modelim
            self.beamCube[channel] = self.beam
            self.dirtymapCube[channel] = self.dirtymap
            self.dirtyvisCube[channel] = self.dirtyvis
            self.modelvisCube[channel] = np.fft.fftshift(self.modelfft)
        self._plotAntennas()
        self._plotSim()
        self._savez_compressed_cubes()


    def _readAntennas(self):
        self.Xmax = 0.0
        for line in self.antenna_coordinates:
            Xmax = np.max(np.abs(antPos[-1] + [Xmax]))
        self.Xmax = Xmax
        cosW = -np.tan(self.lat) * np.tan(self.dec)
        if np.abs(cosW) < 1.0:
            Hhor = np.arccos(cosW)
        elif np.abs(self.lat - self.dec) > np.pi / 2.:
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
        self.Xmax = Xmax * 1.5

    def _get_antenna_coords(self):
        antenna_coordinates = pd.read_csv(os.path.join(master_path, 'antenna_config', 'antenna_coordinates.csv'))
        obs_antennas = antenna_array.split(' ')
        obs_antennas = [antenna.split(':')[0] for antenna in obs_antennas]
        obs_coordinates = antenna_coordinates[antenna_coordinates['name'].isin(obs_antennas)]
        antenna_coordinates = obs_coordinates.iloc[:, 'x', 'y', 'z'].values

        self.antenna_coordinates = antenna_coordinates * 1.e-3 # convert to km
        self.Nant = len(antenna_coordinates)
    
    def _get_observing_wavelengths(self):
        # returns the observing wavelenghts in meters
        w_min, w_max = [self._hz_to_m(freq) for freq in [self.central_freq - self.bandwidth / 2,    self.central_freq + self.bandwidth / 2]]
        obs_wavelengths= np.array([[wave - self.delta_nu /2, wave + self.delta_nu / 2] for wave in np.linspace(w_min, w_max, self.n_channels)])
        return obs_wavelengths

    def _hz_to_m(self, freq):
        return self.c_ms / freq
    
    def _hz_to_km (self, freq):
        return self._hz_to_m(freq) / 1000

    def _alma_radec_to_dec(self):
        """
            Converts RA and Dec to Lon and Lat for the ALMA array on a given date, 
            and calculates initial and final Hour Angles.

        Args:
            date_str (str): Date string in YYYY-MM-DD format.
            ra (float): Right Ascension in degrees.
            dec (float): Declination in degrees.
            total_observation_time (float): Total observation time in seconds.

        Returns:
            tuple: (longitude, latitude, initial_hour_angle, final_hour_angle) in degrees.
        """

         # Convert date string to Astropy Time object
        obstime = astropy.time.Time(self.date_str)
    
        # Define ALMA location (approximately)
        alma_loc = coord.EarthLocation(lon=-69.482606 * U.deg, lat=-23.017469  * U.deg, height=5000 * U.m)  # ALMA location

        # Create sky coordinate object
        sky_coord = coord.ICRS(ra=self.ra * U.deg, dec=self.dec * U.deg)

        # Transform to topocentric coordinates (ALMA frame)
        target_altaz = sky_coord.transform_to(alma_loc)

        # Extract Longitude and Latitude
        longitude = target_altaz.lon.degree
        latitude = target_altaz.lat.degree

        # Calculate sidereal rotation rate (Earth's rotation in radians per second)
        sidereal_rate = 2 * np.pi / (23.9305882 * U.hour)  # sidereal day in hours

        # Convert total observation time to hours
        observation_time_hours = total_observation_time / 3600

        # Calculate Local Sidereal Time (LST) at the start of observation (approximate)
        # Requires more precise calculations for real-world use
        local_sidereal_time = obstime.sidereal_time('apparent')

        # Calculate initial Hour Angle (assuming LST is constant during observation)
        initial_hour_angle = (ra - local_sidereal_time.hour) * 15

        # Calculate final Hour Angle (initial Hour Angle + observation time * sidereal rate)
        final_hour_angle = (initial_hour_angle + observation_time_hours * sidereal_rate.to(U.deg/U.s)) * 180 / np.pi
        self.Hcov = [initial_hour_angle.value, final_hour_angle.value]
        self.dec = latitude.value * self.dec2rad

    def _get_wavelength(self):
        # returns the wavelength in meters
        return self.c_ms / self.central_freq.to(U.Hz).value

    def _get_wavelenght_km(self):
        # return the wavelength in km
        return self._get_wavelength() / 1000

    def _get_half_power_beam_width(self, antenna_diameter: int = 12):
        # returns the half power beam width in radians
        return 1.02 * self._get_wavelength() / antenna_diameter

    def _get_fov(self, antenna_diameter: int = 12):
        # returns the field of view in arcseconds
        fov = 1.22 * self._get_wavelength() / antenna_diameter
        self.fov = fov * self.rad2arcsec

    def _prepareCubes(self):
        self.modelCube = np.zeros((self.n_channels, self.Npix, self.Npix, ), dtype=np.float32)
        self.beamCube = np.zeros((self.n_channels, self.Npix, self.Npix), dtype=np.float32)
        self.dirtymapCube = np.zeros((self.n_channels, self.Npix, self.Npix), dtype=np.float32)
        self.dirtyvisCube = np.zeros((self.n_channels, self.Npix, self.Npix), dtype=np.complex64)
        self.modelvisCube = np.zeros((self.n_channels, self.Npix, self.Npix), dtype=np.complex64)

    def _prepareBeam(self):
        self.beam = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.totsampling = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.dirtymap = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.noisemap = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.robustsamp = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.Gsampling = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.Grobustsamp = np.zeros((self.Npix, self.Npix), dtype=np.complex64)
        self.GrobustNoise = np.zeros((self.Npix, self.Npix),
                                     dtype=np.complex64)

    def _prepareBaselines(self):
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
        for n1 in range(Nant - 1):
            for n2 in range(n1 + 1, self.Nant):
                self.basnum[n1, nii[n1]] = np.int8(bi)
                self.basnum[n2, nii[n2]] = np.int8(bi)
                self.basidx[n1, n2] = np.int8(bi)
                self.antnum[bi] = [n1, n2]
                nii[n1] += 1
                nii[n2] += 1
                bi += np.int8(1)
        self.u = np.zeros((NBmax, nH))
        self.v = np.zeros((NBmax, nH))
        self.ravelDims = (NBmax, nH)

    def _setBaselines(self, antidxs=-1): 
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

    def _gridUV(self, antidx=-1):
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
        self.UVpixsize = 2. / (self.imsize * np.pi / 180. / 3600.)
        for nb in bas2change:
            pixU = np.rint(self.u[nb] / self.UVpixsize).flatten().astype(
                np.int32)
            pixV = np.rint(self.v[nb] / self.UVpixsize).flatten().astype(
                np.int32)
            goodpix = np.where(
                np.logical_and(
                    np.abs(pixU) < self.Nphf,
                    np.abs(pixV) < self.Nphf))[0]
            pU = pixU[goodpix] + self.Nphf
            pV = pixV[goodpix] + self.Nphf
            mU = -pixU[goodpix] + self.Nphf
            mV = -pixV[goodpix] + self.Nphf
            if antidx != -1:
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
            self.pixpos[nb] = [
                np.copy(pU),
                np.copy(pV),
                np.copy(mU),
                np.copy(mV)
            ]
            for pi, gp in enumerate(goodpix):
                gabs = np.abs(self.Gains[nb, gp])
                pVi = pV[pi]
                mUi = mU[pi]
                mVi = mV[pi]
                pUi = pU[pi]
                totsampling[pVi, mUi] += 1.0
                totsampling[mVi, pUi] += 1.0
                Gsampling[pVi, mUi] += self.Gains[nb, gp]
                Gsampling[mVi, pUi] += np.conjugate(self.Gains[nb, gp])
                noisemap[pVi, mUi] += self.Noise[nb, gp] * gabs
                noisemap[mVi, pUi] += np.conjugate(
                    self.Noise[nb, gp]) * gabs
        self.robfac = (5. * 10.**(-self.robust))**2. * (
            2. * self.Nbas * self.nH) / np.sum(self.totsampling**2.)    

    def _setPrimaryBeam(self):
        PB = 2. * (1220. * self.rad2arcsec * self.wavelength[2] /
                           self.antenna_diameter / 2.3548)**2.
        beamImg = np.exp(self.distmat / PB)
        self.modelim = self.modelimTrue * beamImg
        self.modelfft = np.fft.fft2(np.fft.fftshift(self.modelim))
        self.modelvis = np.fft.fftshift(modelfft)

    def _setBeam(self):

        self._gridUV(antidx=antidx)
        denom = 1. + self.robfac * self.totsampling
        self.robustsamp[:] = self.totsampling / denom
        self.Grobustsamp[:] = self.Gsampling / denom
        self.GrobustNoise[:] = self.noisemap / denom
        self.beam[:] = np.fft.ifftshift(
                    np.fft.ifft2(np.fft.fftshift(
                    self.robustsamp))).real / (1. + self.W2W1)
        self.beamScale = np.max(self.beam[self.Nphf:self.Nphf +
                                                  1, self.Nphf:self.Nphf + 1])
        self.beam[:] /= self.beamScale

    def _obs_model(self):
        self.dirtyvis[:] = np.fft.ifftshift(GrobustNoise) + modelfft * np.fft.ifftshift(Grobustsamp)
        self.dirtymap[:] = (np.fft.fftshift(
            np.fft.ifft2(dirtyvis))).real / (1. + W2W1)
        self.dirtymap /= beamScale 
       
    def _plotAntennas(self):
        antPlot = plt.figure(figsize=(8, 8))
        mw = 2. * self.Xmax / self.wavelength[2] / self.lambdafac
        if mw < 0.1 and self.lambdafac == 1.e6:
            self.lambdafac = 1.e3
            self.ulab = r'U (k$\lambda$)'
            self.vlab = r'V (k$\lambda$)'
        elif mw >= 100. and self.lambdafac == 1.e3:
            self.lambdafac = 1.e6
            self.ulab = r'U (M$\lambda$)'
            vlab = r'V (M$\lambda$)'
        toplot = np.array(self.antPos[:Nant])
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
        plt.savez_compressedfig(os.path.join(self.plot_path, 'antenna_config.png'))

        UVPlot = plt.figure(figsize=(8, 8))
        UVPlotPlot = []
        toplotu = self.u.flatten() / self.lambdafac
        toplotv = self.v.flatten() / self.lambdafac
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
        plt.xlim((2. * self.Xmax / self.wavelength[2] / self.lambdafac,
                         -2. * self.Xmax / self.wavelength[2] / self.lambdafac))
        plt.ylim((2. * self.Xmax / self.wavelength[2] / self.lambdafac,
                         -2. * self.Xmax / self.wavelength[2] / self.lambdafac))
        plt.xlabel(ulab)
        plt.ylabel(vlab)
        plt.title('UV Coverage')
        plt.save(os.path.join(self.plot_path, 'uv_coverage.png'))

    def _plotBeam(self):
        self.Np4 = self.Npix // 4
        beamPlot = plt.figure(figsize=(8, 8))
        beam_img = plt.avg(self.beamCube, axis=0)
        beamPlotPlot = plt.imshow(
            beam_img[Np4:Npix - Np4, Np4:Npix - Np4],
            picker=True,
            interpolation='near')
        plt.ylabel('Dec offset (as)')
        plt.xlabel('RA offset (as)')
        plt.setp(beamPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.)) 
        plt.title('DIRTY BEAM')
        nptot = np.sum(self.totsampling[:])
        beamPlotPlot.norm.vmin = np.min(beam_img)
        beamPlotPlot.norm.vmax = 1.0
        plt.save(os.path.join(self.plot_path, 'beam.png'))

    def _plotSim(self):
        self.Np4 = self.Npix // 4
        simPlot, ax = plt.subplots(2, 2, figsize=(12, 12))

        sim_img = np.sum(self.modelCube, axis=0)
        simPlotPlot = ax[0, 0].imshow(
            np.power(
            sim_img[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4], self.gamma),
            picker=True,
            interpolation='nearest',
            vmin=0.0,
            vmax=np.max(sim_img)**self.gamma,
            cmap=currcmap)
        plt.setp(simPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.)) 
        ax[0, 0].set_ylabel('Dec offset (as)')
        ax[0, 0].set_xlabel('RA offset (as)')
        totflux = np.sum(modelimTrue[Np4:Npix -Np4, Np4:Npix - Np4])
        ax[0, 0].set_title('MODEL IMAGE: %.2e Jy' % totflux)
        simPlotPlot.norm.vmin = np.min(sim_img)
        simPlotPlot.norm.vmax = np.max(sim_img)

        dirty_img = np.sum(self.dirtymapCube, axis=0)
        dirtyPlotPlot = ax[0, 1].imshow(
            dirty_img[Np4:Npix - Np4, Np4:Npix - Np4],
            picker=True,
            interpolation='near')
        plt.setp(dirtyPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.))
        ax[0, 1].set_ylabel('Dec offset (as)')
        ax[0, 1].set_xlabel('RA offset (as)')
        totflux = np.sum(dirty_img[Np4:Npix - Np4, Np4:Npix - Np4])
        ax[0, 1].set_title('DIRTY IMAGE: %.2e Jy' % totflux)
        dirtyPlotPlot.norm.vmin = np.min(dirty_img)
        dirtyPlotPlot.norm.vmax = np.max(dirty_img)

        self.UVmax = self.Npix / 2. / self.lambdafac * self.UVpixsize
        self.UVSh = -self.UVmax / self.Npix
        toplot = np.sum(np.abs(self.modelvisCube), axis=0)
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
        pl.setp(UVPlotDirtyFFTPlot,
                    extent=(-self.UVmax + self.UVSh, self.UVmax + self.UVSh,
                            -self.UVmax - self.UVSh, self.UVmax - self.UVSh))
        ax[1, 1].set_ylabel('V (k$\\lambda$)')
        ax[1, 1].set_xlabel('U (k$\\lambda$)')
        ax[1, 1].set_title('DIRTY VISIBILITY')
        plt.save(os.path.join(self.plot_path, 'sim.png'))

    def _savez_compressed_cubes(self):
        np.savez_compressed(os.path.join(self.output_path, 'modelCube.npz'), self.modelCube)
        np.savez_compressed(os.path.join(self.output_path, 'beamCube.npz'), self.beamCube)
        np.savez_compressed(os.path.join(self.output_path, 'dirtymapCube.npz'), self.dirtymapCube)
        np.savez_compressed(os.path.join(self.output_path, 'dirtyvisCube.npz'), self.dirtyvisCube)
        np.savez_compressed(os.path.join(self.output_path, 'modelvisCube.npz'), self.modelvisCube)


    


