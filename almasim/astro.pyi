from _typeshed import Incomplete
from astropy.constants import c as c
from astropy.coordinates import SkyCoord as SkyCoord
from astropy.cosmology import FlatLambdaCDM as FlatLambdaCDM
from astropy.io import fits as fits
from math import pi as pi
from tqdm import tqdm as tqdm

def compute_redshift(rest_frequency, observed_frequency): ...
def redshift_to_snapshot(redshift): ...
def get_data_from_hdf(file, snapshot): ...
def get_subhaloids_from_db(n, main_path, snapshot): ...
def partTypeNum(partType): ...
def gcPath(basePath, snapNum, chunkNum: int = 0): ...
def offsetPath(basePath, snapNum): ...
def loadObjects(basePath, snapNum, gName, nName, fields): ...
def loadSubhalos(basePath, snapNum, fields: Incomplete | None = None): ...
def loadHalos(basePath, snapNum, fields: Incomplete | None = None): ...
def loadHeader(basePath, snapNum): ...
def load(basePath, snapNum): ...
def loadSingle(basePath, snapNum, haloID: int = -1, subhaloID: int = -1): ...
def snapPath(basePath, snapNum, chunkNum: int = 0): ...
def snapPath2(basePath, snapNum, chunkNum: int = 0): ...
def getNumPart(header): ...
def loadSubset(basePath, snapNum, partType, fields: Incomplete | None = None, subset: Incomplete | None = None, mdi: Incomplete | None = None, sq: bool = True, float32: bool = False, outPath: Incomplete | None = None, api_key: Incomplete | None = None): ...
def download_groupcat(basePath, snapNum, fileNum, api_key) -> None: ...
def getSnapOffsets(basePath, snapNum, id, type, api_key): ...
def get_particles_num(basePath, outputPath, snapNum, subhaloID, tng_api_key): ...
def loadSubhalo(basePath, snapNum, id, partType, tng_api_key, fields: Incomplete | None = None): ...
def loadHalo(basePath, snapNum, id, partType, fields: Incomplete | None = None, api_key: Incomplete | None = None): ...
def read_line_emission_csv(path_line_emission_csv, sep: str = ';'): ...
def get_line_info(main_path, idxs: Incomplete | None = None): ...
def compute_rest_frequency_from_redshift(master_path, source_freq, redshift): ...
def write_sim_parameters(path, ra, dec, ang_res, vel_res, int_time, band, band_range, central_freq, redshift, line_fluxes, line_names, line_frequencies, continum, fov, beam_size, cell_size, n_pix, n_channels, snapshot, subhalo, lum_infrared, fwhm_z, source_type, fwhm_x: Incomplete | None = None, fwhm_y: Incomplete | None = None, angle: Incomplete | None = None) -> None: ...
def get_image_from_ssd(ra, dec, fov) -> None: ...
