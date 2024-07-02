from _typeshed import Incomplete

def get_tap_service(): ...
def search_with_retry(service, query): ...
def get_science_types(): ...
def query_observations(member_ous_uid, target_name): ...
def query_all_targets(targets): ...
def query_by_science_type(science_keyword: Incomplete | None = None, scientific_category: Incomplete | None = None, band: Incomplete | None = None, fov_range: Incomplete | None = None, time_resolution_range: Incomplete | None = None, frequency_range: Incomplete | None = None): ...
def estimate_alma_beam_size(central_frequency_ghz, max_baseline_km, return_value: bool = True): ...
def get_fov_from_band(band, antenna_diameter: int = 12, return_value: bool = True): ...
def generate_antenna_config_file_from_antenna_array(antenna_array, master_path, output_dir) -> None: ...
def compute_distance(x1, y1, z1, x2, y2, z2): ...
def get_max_baseline_from_antenna_config(update_progress, antenna_config): ...
def get_max_baseline_from_antenna_array(antenna_array, master_path): ...
def get_band_range(band): ...
def get_band_central_freq(band): ...
def get_antennas_distances_from_reference(antenna_config): ...
def generate_prms(antbl, scaleF): ...
def simulate_atmospheric_noise(sim_output_dir, project, scale, ms, antennalist) -> None: ...
def simulate_gain_errors(ms, amplitude: float = 0.01): ...
