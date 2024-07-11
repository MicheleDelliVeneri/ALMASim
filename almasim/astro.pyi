from _typeshed import Incomplete

def compute_redshift(rest_frequency, observed_frequency): ...
def redshift_to_snapshot(redshift): ...
def get_data_from_hdf(file, snapshot): ...
def get_subhaloids_from_db(n, main_path, snapshot): ...
def read_line_emission_csv(path_line_emission_csv, sep: str = ";"): ...
def get_line_info(main_path, idxs: Incomplete | None = None): ...
def compute_rest_frequency_from_redshift(master_path, source_freq, redshift): ...
def write_sim_parameters(
    path,
    ra,
    dec,
    ang_res,
    vel_res,
    int_time,
    band,
    band_range,
    central_freq,
    redshift,
    line_fluxes,
    line_names,
    line_frequencies,
    continum,
    fov,
    beam_size,
    cell_size,
    n_pix,
    n_channels,
    snapshot,
    subhalo,
    lum_infrared,
    fwhm_z,
    source_type,
    fwhm_x: Incomplete | None = None,
    fwhm_y: Incomplete | None = None,
    angle: Incomplete | None = None,
) -> None: ...

# def get_image_from_ssd(ra, dec, fov) -> None: ...
