from _typeshed import Incomplete
from martini.sources.sph_source import SPHSource

class myTNGSource(SPHSource):
    def __init__(
        self,
        snapNum,
        subID,
        basePath: Incomplete | None = None,
        distance=...,
        vpeculiar=...,
        rotation=...,
        ra=...,
        dec=...,
        api_key: Incomplete | None = None,
    ) -> None: ...

class DataCube:
    HIfreq: Incomplete
    stokes_axis: Incomplete
    px_size: Incomplete
    arcsec2_to_pix: Incomplete
    velocity_centre: Incomplete
    channel_width: Incomplete
    ra: Incomplete
    dec: Incomplete
    padx: int
    pady: int
    def __init__(
        self,
        n_px_x: int = 256,
        n_px_y: int = 256,
        n_channels: int = 64,
        px_size=...,
        channel_width=...,
        velocity_centre=...,
        ra=...,
        dec=...,
        stokes_axis: bool = False,
    ) -> None: ...
    def spatial_slices(self): ...
    def spectra(self): ...
    def freq_channels(self) -> None: ...
    def velocity_channels(self) -> None: ...
    def add_pad(self, pad) -> None: ...
    def drop_pad(self) -> None: ...
    def copy(self): ...
    def save_state(self, filename, overwrite: bool = False) -> None: ...
    @classmethod
    def load_state(cls, filename): ...

class Martini:
    quiet: Incomplete
    find_distance: Incomplete
    source: Incomplete
    datacube: Incomplete
    beam: Incomplete
    noise: Incomplete
    sph_kernel: Incomplete
    spectral_model: Incomplete
    inserted_mass: int
    def __init__(
        self,
        source: Incomplete | None = None,
        datacube: Incomplete | None = None,
        beam: Incomplete | None = None,
        noise: Incomplete | None = None,
        sph_kernel: Incomplete | None = None,
        spectral_model: Incomplete | None = None,
        quiet: bool = False,
        find_distance: bool = False,
    ) -> None: ...
    def convolve_beam(self) -> None: ...
    def add_noise(self) -> None: ...
    def insert_source_in_cube(
        self,
        skip_validation: bool = False,
        progressbar: Incomplete | None = None,
        ncpu: int = 1,
    ): ...
    def write_fits(
        self, filename, channels: str = "frequency", overwrite: bool = True
    ) -> None: ...
    def write_beam_fits(
        self, filename, channels: str = "frequency", overwrite: bool = True
    ) -> None: ...
    def write_hdf5(
        self,
        filename,
        channels: str = "frequency",
        overwrite: bool = True,
        memmap: bool = False,
        compact: bool = False,
    ): ...
    def reset(self) -> None: ...

def gaussian(x, amp, cen, fwhm): ...
def gaussian2d(x, y, amp, cen_x, cen_y, fwhm_x, fwhm_y, angle): ...
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
): ...
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
): ...
def interpolate_array(arr, n_px): ...
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
): ...
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
): ...
def insert_extended(
    terminal, datacube, tngpath, snapshot, subhalo_id, redshift, ra, dec, api_key, ncpu
): ...
def diffuse_signal(n_px): ...
def insert_diffuse(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_px, n_chan
): ...
def distance_1d(p1, p2): ...
def distance_2d(p1, p2): ...
def distance_3d(p1, p2): ...
def get_iou(bb1, bb2): ...
def get_iou_1d(bb1, bb2): ...
def get_pos(x_radius, y_radius, z_radius): ...
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
): ...
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
): ...
def get_datacube_header(datacube, obs_date): ...
def write_datacube_to_fits(
    datacube, filename, obs_date, channels: str = "frequency", overwrite: bool = True
) -> None: ...
