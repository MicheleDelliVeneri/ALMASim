from _typeshed import Incomplete
from martini.sources.sph_source import SPHSource

class MartiniMod:
    memory_limit: Incomplete
    def _evaluate_pixel_spectrum(
        self, ranks_and_ij_pxs, update_progress, progressbar=True
    ): ...
    def _insert_source_in_cube(
        self,
        update_progress=None,
        terminal=None,
        skip_validation=False,
        progressbar=None,
        ncpu=1,
        quiet=None,
    ): ...

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
    update_progress,
    terminal,
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
    update_progress,
    terminal,
    datacube,
    tngpath,
    snapshot,
    subhalo_id,
    redshift,
    ra,
    dec,
    api_key,
    ncpu,
): ...
def diffuse_signal(n_px): ...
def insert_diffuse(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_px, n_chan
): ...
def make_extended(
    imsize,
    powerlaw=2.0,
    theta=0.0,
    ellip=1.0,
    return_fft=False,
    full_fft=True,
    randomseed=32768324,
): ...
def molecular_cloud(n_px): ...
def insert_molecular_cloud(
    update_progress, datacube, continum, line_fluxes, pos_z, fwhm_z, n_pix, n_chan
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
