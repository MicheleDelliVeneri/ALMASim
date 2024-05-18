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

def showError(message):
        raise Exception(message)


class Interferometer(object):

    def __init__(self, master_path, output_path, model, integration_time, total_time, n_pix, n_channels,  bandwidth, central_freq, lat, dec, antenna_array, antenna_diameter=12):
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
        self.c_ms c.to(U.m / U.s).value
        self.deg2rad = np.pi / 180.
        self.rad2deg = 180. / np.pi
        self.rad2arcsec = 3600. * self.rad2deg
        self.deltaAng = 1. * self.deg2rad 
        self.lat = self.deg2rad * lat
        self.dec = self.deg2rad * dec
        self.trlat [np.sin(lat), np.cos(lat)]
        self.trdec [np.sin(dec), np.cos(dec)]

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
        self.readAntennas()
        self.model = model
        self.imsize = 1.5 * self.fov
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
            self.modelim = self.modelCube[channel]
            xx = np.linspace(-self.imsize / 2., self.imsize / 2., self.Npix)
            yy = np.ones(self.Npix, dtype=np.float32)
            self.distmat = (-np.outer(xx**2., yy) -
                        np.outer(yy, xx**2.)) * pixsize**2.
            self._setPrimaryBeam()
            self.obs_model()
        self.modelCube[channel] = self.modelim
        self.beamCube[channel] = self.beam
        self.dirtymapCube[channel] = self.dirtymap

    def readAntennas(self):
        self.Hcov = [-12.0 * Hfac, 12.0 * Hfac]
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
        self.beamCube = np.zeros((self.Npix, self.Npix), dtype=np.float32)
        self.dirtymapCube = np.zeros((self.Npix, self.Npix), dtype=np.float32)

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
        self.dirtymap[:] = (np.fft.fftshift(
            np.fft.ifft2(
            np.fft.ifftshift(GrobustNoise) + modelfft *
            np.fft.ifftshift(Grobustsamp)))).real / (1. + W2W1)
        self.dirtymap /= beamScale 
       
