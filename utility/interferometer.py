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
import pandas as pd

def showError(message):
        raise Exception(message)


""" class Interferometer(object):

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
        print(self.nH)
        self.antenna_diameter = antenna_diameter
        self.Npix = n_pix
        self.central_freq = central_freq
        self.bandwidth = bandwidth
        self.n_channels = n_channels
        # Constants and conversions
        self.robust = 0.0
        self.Hfac = np.pi / 180. * 15.
        self.c_ms = c.to(U.m / U.s).value
        self.deg2rad = np.pi / 180.
        self.rad2deg = 180. / np.pi
        self.rad2arcsec = 3600. * self.rad2deg
        self.deltaAng = 1. * self.deg2rad 
        self.lat = 23.017469 * self.deg2rad 
        self.date_str = obs_date
        self.ra = ra
        self.dec = dec * self.deg2rad
        alma_loc = EarthLocation.of_site('ALMA')
        self.lat = alma_loc.lat * self.deg2rad
        if np.abs(self.lat - self.dec >= np.pi / 2.):
           print("\nSource is either not observable or just at the horizon!\n\n")

        self.trlat = [np.sin(self.lat), np.cos(self.lat)]
        self.trdec = [np.sin(self.dec), np.cos(self.dec)]
        self.delta_nu = self._hz_to_m(self.bandwidth  / self.n_channels) # channel width in meters
        self.obs_wavelenghts = self._get_observing_wavelength() # observing wavelenghts in meters
        print("{:.2e}".format(self.obs_wavelenghts[0][0]), "{:.2e}".format(self.obs_wavelenghts[-1][1]))
        self.gamma = 0.5  # gamma correction to plot model 
        self.lambdafac = 1.e6 # scale factor from km to mm
        self.W2W1 = 1.0 # relative weight of subarrays
        self.hpbw = self._get_half_power_beam_width() # half power beam width in radians
         # coverage in hours, CHECK THIS
        self.antenna_array = antenna_array
        self._get_antenna_coords()
        self._get_fov()
        self._readAntennas()
        self.model = model
        self.imsize = 2 * self.fov
        self.pixsize = float(self.imsize) / self.Npix
        self.Xaxmax = self.imsize / 2.
        self.Nphf = self.Npix // 2
        self.robfac = 0.0
        self.currcmap = cm.jet
        self._prepareCubes()
        for channel in tqdm(range(self.n_channels)):
            self.wavelength = list(self.obs_wavelenghts[channel] * 1e-3)
            self.wavelength.append(
                                (self.wavelength[0] + self.wavelength[1]) / 2.)
            self.fmtB1 = r'$\lambda = $ %4.1fmm  ' % (self.wavelength[2] * 1.e6)
            self.fmtB = self.fmtB1 + "\n" + r'% 4.2f Jy/beam' + "\n" + r'$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f '
            self._prepareBeam()
            self._prepareBaselines()
            self._setBaselines()
            self._setBeam()
            self.modelim = model[channel]
            xx = np.linspace(-self.imsize / 2., self.imsize / 2., self.Npix)
            yy = np.ones(self.Npix, dtype=np.float32)
            self.distmat = (-np.outer(xx**2., yy) -
                        np.outer(yy, xx**2.)) * self.pixsize**2.
            self._setPrimaryBeam()
            self._obs_model()
        
            self.modelCube[channel] = self.modelim
            self.beamCube[channel] = self.beam
            self.dirtymapCube[channel] = self.dirtymap
            self.dirtyvisCube[channel] = self.dirtyvis
            self.modelvisCube[channel] = np.fft.fftshift(self.modelfft)
        self._plotAntennas()
        self._plotSim()
        self._plotBeam()
        self._savez_compressed_cubes()
        self._free_space()


    def _readAntennas(self):
        antPos = []
        Xmax = 0.0
        Hcov = [-1.0 * self.Hfac, 1.0 * self.Hfac]
        for line in self.antenna_coordinates:
            antPos.append([line[0] * 1e-3, line[1] * 1e-3])
            Xmax = np.max(np.abs(antPos[-1] + [Xmax]))
        self.Xmax = Xmax
        self.antPos = antPos
        self.Hcov = Hcov
        cosW = -np.tan(self.lat) * np.tan(self.dec)
        if np.abs(cosW) < 1.0:
            Hhor = np.arccos(cosW)
        elif np.abs(self.lat - self.dec) > np.pi / 2.:
            Hhor = 0
        else:
            Hhor = np.pi
        Hhor = Hhor.value
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
        antenna_coordinates = pd.read_csv(os.path.join(self.master_path, 'antenna_config', 'antenna_coordinates.csv'))
        obs_antennas = self.antenna_array.split(' ')
        obs_antennas = [antenna.split(':')[0] for antenna in obs_antennas]
        obs_coordinates = antenna_coordinates[antenna_coordinates['name'].isin(obs_antennas)]
        antenna_coordinates = obs_coordinates[['x', 'y']].values

        self.antenna_coordinates = antenna_coordinates * 1.e-3 # convert to km
        self.Nant = len(antenna_coordinates)
    
    def _get_observing_wavelength(self):
        
    
        # returns the observing wavelenghts in meters
        w_max, w_min = [self._hz_to_m(freq) for freq in [self.central_freq - self.bandwidth / 2,    self.central_freq + self.bandwidth / 2]]
        waves = np.linspace(w_min, w_max, self.n_channels + 1)
        obs_wavelengths= np.array([[waves[i], waves[i + 1] ] for i in range(len(waves) - 1)])
        return obs_wavelengths

    def _hz_to_m(self, freq):
        return self.c_ms / freq
    
    def _hz_to_km (self, freq):
        return self._hz_to_m(freq) / 1000

    def _alma_radec_to_dec(self):

        # Define ALMA location (approximately)
        
         # Convert date string to Astropy Time object
        obs_time = astropy.time.Time(self.date_str + 'T00:00:00', format='isot', scale='utc', location=alma_loc)
        #self.dec = self.dec * self.deg2rad
        #aa = AltAz(location=alma_loc, obstime=obs_time)

        # Create sky coordinate object
        #sky_coord = SkyCoord(ra=self.ra * U.deg, dec=self.dec * U.deg, frame='icrs')

        # Transform to topocentric coordinates (ALMA frame)
        #target_altaz = sky_coord.transform_to(aa)

        # Extract Longitude and Latitude
        #longitude = target_altaz.lon.degree
        #latitude = target_altaz.lat.degree

        # Calculate sidereal rotation rate (Earth's rotation in radians per second)
        #sidereal_rate_hours = 23.9344696

        # Convert total observation time to hours
        #observation_time_hours = self.total_time / 3600

        # Calculate Local Sidereal Time (LST) at the start of observation (approximate)
        # Requires more precise calculations for real-world use
        #local_sidereal_time = obs_time.sidereal_time('apparent')

        # Calculate initial Hour Angle (assuming LST is constant during observation)
        #initial_hour_angle = self.ra * self.deg2rad * 15. - local_sidereal_time.hour

        # Calculate final Hour Angle (initial Hour Angle + observation time * sidereal rate)
        #final_hour_angle = initial_hour_angle + (observation_time_hours * sidereal_rate_hours)

        #self.Hcov = [-12 * self.Hfac, 12 * self.Hfac]
        #self.dec * self.deg2rad

    def _get_wavelength(self):
        # returns the wavelength in meters
        return self.c_ms / self.central_freq

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

    def _setBaselines(self, antidx=-1): 
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
                self.totsampling[pVi, mUi] += 1.0
                self.totsampling[mVi, pUi] += 1.0
                self.Gsampling[pVi, mUi] += self.Gains[nb, gp]
                self.Gsampling[mVi, pUi] += np.conjugate(self.Gains[nb, gp])
                self.noisemap[pVi, mUi] += self.Noise[nb, gp] * gabs
                self.noisemap[mVi, pUi] += np.conjugate(
                    self.Noise[nb, gp]) * gabs
        self.robfac = (5. * 10.**(-self.robust))**2. * (
            2. * self.Nbas * self.nH) / np.sum(self.totsampling**2.)    

    def _setPrimaryBeam(self):
        PB = 2. * (1220. * self.rad2arcsec * self.wavelength[2] /
                           self.antenna_diameter / 2.3548)**2.
        beamImg = np.exp(self.distmat / PB)
        self.modelim = self.modelim * beamImg
        self.modelfft = np.fft.fft2(np.fft.fftshift(self.modelim))
        self.modelvis = np.fft.fftshift(self.modelfft)

    def _setBeam(self, antidx=-1):

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
        self.dirtyvis = np.fft.ifftshift(self.GrobustNoise) + self.modelfft * np.fft.ifftshift(self.Grobustsamp)
        self.dirtymap[:] = (np.fft.fftshift(
            np.fft.ifft2(self.dirtyvis))).real / (1. + self.W2W1)
        self.dirtymap /= self.beamScale 
       
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
        plt.savefig(os.path.join(self.plot_path, 'antenna_config.png'))

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
        plt.xlabel(self.ulab)
        plt.ylabel(self.vlab)
        plt.title('UV Coverage')
        plt.savefig(os.path.join(self.plot_path, 'uv_coverage.png'))
        plt.close()

    def _plotBeam(self):
        self.Np4 = self.Npix // 4
        beamPlot = plt.figure(figsize=(8, 8))
        beam_img = self.beamCube[self.n_channels // 2]
        beamPlotPlot = plt.imshow(
            beam_img[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4],
            picker=True,
            interpolation='nearest', 
            cmap=self.currcmap)
        beamText = plt.text(
                0.05,
                 0.80,
                self.fmtB % (1.0, 0.0, 0.0),
                bbox=dict(facecolor='white', alpha=0.7))
        plt.ylabel('Dec offset (as)')
        plt.xlabel('RA offset (as)')
        plt.setp(beamPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.)) 
        plt.title('DIRTY BEAM')
        nptot = np.sum(self.totsampling[:])
        beamPlotPlot.norm.vmin = np.min(beam_img)
        beamPlotPlot.norm.vmax = 1.0
        print(nptot)
        print(np.sum(self.totsampling[self.Nphf - 4:self.Nphf + 4, self.Nphf -
                           4:self.Nphf + 4]))
        plt.colorbar()
        plt.savefig(os.path.join(self.plot_path, 'beam.png'))
        plt.close()

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
            cmap=self.currcmap)
        plt.setp(simPlotPlot,
            extent=(self.Xaxmax / 2., -self.Xaxmax / 2.,
                    -self.Xaxmax / 2., self.Xaxmax / 2.)) 
        ax[0, 0].set_ylabel('Dec offset (as)')
        ax[0, 0].set_xlabel('RA offset (as)')
        totflux = np.sum(self.modelim[self.Np4:self.Npix - self.Np4, self.Np4:self.Npix - self.Np4])
        ax[0, 0].set_title('MODEL IMAGE: %.2e Jy' % totflux)
        simPlotPlot.norm.vmin = np.min(sim_img)
        simPlotPlot.norm.vmax = np.max(sim_img)

        dirty_img = np.sum(self.dirtymapCube, axis=0)
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
        plt.setp(UVPlotDirtyFFTPlot,
                    extent=(-self.UVmax + self.UVSh, self.UVmax + self.UVSh,
                            -self.UVmax - self.UVSh, self.UVmax - self.UVSh))
        ax[1, 1].set_ylabel('V (k$\\lambda$)')
        ax[1, 1].set_xlabel('U (k$\\lambda$)')
        ax[1, 1].set_title('DIRTY VISIBILITY')
        plt.savefig(os.path.join(self.plot_path, 'sim.png'))
        plt.close()

    def _savez_compressed_cubes(self):
        np.savez_compressed(os.path.join(self.output_path, 'modelCube.npz'), self.modelCube)
        np.savez_compressed(os.path.join(self.output_path, 'beamCube.npz'), self.beamCube)
        np.savez_compressed(os.path.join(self.output_path, 'dirtymapCube.npz'), self.dirtymapCube)
        np.savez_compressed(os.path.join(self.output_path, 'dirtyvisCube.npz'), self.dirtyvisCube)
        np.savez_compressed(os.path.join(self.output_path, 'modelvisCube.npz'), self.modelvisCube)

    def _free_space(self):
        del self.modelCube
        del self.beamCube
        del self.dirtymapCube
        del self.dirtyvisCube
        del self.modelvisCube
 """



class Interferometer():

    def __init__(self, idx, skymodel, main_dir, output_dir, dec, central_freq, band_range, fov, antenna_array):
        self.idx = idx
        self.skymodel = skymodel
        self.antenna_array = antenna_array
        # Constants
        self.c_ms = c.to(U.m / U.s).value
        # Directories
        self.main_dir = main_dir
        self.output_dir = output_dir
        self.plot_dir = os.path.join(output_dir, 'plots')
        # Parameters
        self.Hfac = np.pi / 180. * 15.
        self.deg2rad = np.pi / 180.
        self.curzoom = [0, 0, 0, 0]
        self.robust = 0.0
        self.deltaAng = 1. * self.deg2rad
        self.gamma = 0.5
        self.lfac = 1.e6
        # PLACEHOLDER MUST BE SUBSTITUTED WITH REAL NUMBER OF SCANS 
        self.nH = 200
        self.Hmax = np.pi
        self.lat = -23.028 * self.deg2rad
        self.trlat = [np.sin(self.lat), np.cos(self.lat)]
        self.Diameters = [12.0, 0]
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
        self._read_antennas()
        # Get the observing wavelengths for each channel 
        self._get_wavelengths()
        self._prepare_cubes()
        for channel in tqdm(range(self.Nchan)):
            self.channel = channel
            self._get_channel_range()
            self._prepare_2d_arrays()
            self._prepare_baselines()
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
        self.plot_sim()
        self.savez_compressed_cubes()
        self.free_space()

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
        Hcov = [-3.0 * self.Hfac, 3.0 * self.Hfac]
        for line in antenna_coordinates:
            antPos.append([line[0] * 1e-3, line[1] * 1e-3])
            Xmax = np.max(np.abs(antPos[-1] + [Xmax]))
        self.Xmax = Xmax
        self.antPos = antPos
        self.Hcov = Hcov
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
        w_max, w_min = [self._hz_to_m(freq) for freq in [self.central_freq - self.band_range / 2, self.central_freq + self.band_range / 2]]
        waves = np.linspace(w_min, w_max, self.Nchan + 1)
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
        self.modelCube[self.channel] = self.img
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
                self.totsampling[pVi, mUi] += 1.0
                self.totsampling[mVi, pUi] += 1.0
                self.Gsampling[pVi, mUi] += self.Gains[nb, gp]
                self.Gsampling[mVi, pUi] += np.conjugate(self.Gains[nb, gp])
                self.noisemap[pVi, mUi] += self.Noise[nb, gp] * gabs
                self.noisemap[mVi, pUi] += np.conjugate(
                    self.Noise[nb, gp]) * gabs
        self.robfac = (5. * 10.**(-self.robust))**2. * (
            2. * self.Nbas * self.nH) / np.sum(self.totsampling**2.)
    
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
            0.05,
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

    def _prepare_model(self):
        self.modelim = [np.zeros((self.Npix, self.Npix), dtype=np.float32) for i in [0, 1]]
        self.modelimTrue = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        dims = np.shape(self.img)
        d1 = self.img.shape[0]
        avimg = self.img
        if d1 == self.Nphf:
            sh0 = (self.Nphf - dims[0]) // 2
            sh1 = (self.Nphf - dims[1]) // 2
            self.modelimTrue[sh0 + self.Np4:sh0 + self.Np4 + dims[0], sh1 + self.Np4:sh1 +
                                 self.Np4 + dims[1]] += self.zoomimg
        else:
            zoomimg = spndint.zoom(avimg, float(self.Nphf) / d1)
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
        self.modelvis = np.fft.fftshift(self.modelfft)
        self.dirtyvis = np.fft.ifftshift(self.GrobustNoise) + self.modelfft * np.fft.ifftshift(self.Grobustsamp)

    def _savez_compressed_cubes(self):
        np.savez_compressed(os.path.join(self.output_dir, 'clean-cube_{}.npz'.format(str(self.idx))), self.modelCube)
        np.savez_compressed(os.path.join(self.output_dir, 'dirty-cube_{}.npz'.format(str(self.idx))), self.dirtyCube)
        np.savez_compressed(os.path.join(self.output_dir, 'dirty-vis-cube_{}.npz'.format(str(self.idx))), self.dirtyvisCube)
        np.savez_compressed(os.path.join(self.output_dir, 'clean-vis-cube_{}.npz'.format(str(self.idx))), self.visCube)
    
    def _free_space(self):
        del self.modelCube
        del self.dirtCube
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
