import numpy as np
from scipy.ndimage import zoom
from astropy import units as u
from martini import DataCube
import os
from skimage import io
import nifty8 as ift
from dask import delayed, compute
import random

# Utility Functions
def interpolate_array(arr, target_size):
    """Resize a 2D array while preserving aspect ratio."""
    zoom_factors = [target_size / dim for dim in arr.shape]
    return zoom(arr, zoom_factors)


def gaussian(x, amp, cen, fwhm):
    """Generates a 1D Gaussian."""
    sigma = fwhm / 2.35482  # Convert FWHM to standard deviation
    gaussian = amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)
    norm = np.sum(gaussian)
    return gaussian / norm if norm != 0 else gaussian


def diffuse_signal(n_px, seed=None, args=None):
    """
    Generates a normalized diffuse signal using a correlated random field.

    Parameters:
        n_px (int): Number of pixels along each dimension (must be positive).
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        args (dict, optional): Custom parameters for the correlated field.
            Defaults to a predefined configuration.

    Returns:
        np.ndarray: Normalized 2D diffuse signal of size (n_px, n_px).

    Raises:
        ValueError: If n_px is not greater than 0.
    """

    # Validate input
    if n_px <= 0:
        raise ValueError("`n_px` must be greater than 0 to generate a signal.")

    # Set random seed
    if seed is not None:
        ift.random.push_sseq(seed)
    else:
        ift.random.push_sseq(random.randint(1, 1000))
    # Define the space
    space = ift.RGSpace((2 * n_px, 2 * n_px))
    # Default arguments for the correlated field
    default_args = {
        "offset_mean": 24,
        "offset_std": (1, 0.1),
        "fluctuations": (5.0, 1.0),
        "loglogavgslope": (-3.5, 0.5),
        "flexibility": (1.2, 0.4),
        "asperity": (0.2, 0.2),
    }
    args = args or default_args  # Use passed arguments or fall back to default

    # Correlated field generation
    cf = ift.SimpleCorrelatedField(space, **args)
    exp_cf = ift.exp(cf)

    # Generate random positions and sample the field
    random_pos = ift.from_random(exp_cf.domain)
    sample = np.log(exp_cf(random_pos))
    data = sample.val[:n_px, :n_px]  # Slice to the desired size

    # Normalize the data
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min) if data_max != data_min else data

    return normalized_data


def molecular_cloud(n_px):
    powerlaw = random.random() * 3.0 + 1.5
    ellip = random.random() * 0.5 + 0.5
    theta = random.random() * 2 * np.pi
    im = make_extended(
        n_px,
        powerlaw=powerlaw,
        theta=theta,
        ellip=ellip,
        randomseed=random.randrange(10000),
    )
    im -= np.min(im)
    if np.max(im) > 0:  # Avoid division by zero
        im /= np.max(im)
    return im



def make_extended(imsize, powerlaw=2.0, theta=0.0, ellip=1.0, randomseed=None):
    """Generate a 2D power-law image with specified index and random phases."""
    np.random.seed(randomseed)
    freq_y, freq_x = np.meshgrid(
        np.fft.fftfreq(imsize), np.fft.rfftfreq(imsize), indexing="ij"
    )

    rr2 = freq_x**2 + (freq_y / ellip) ** 2
    rr2[0, 0] = np.inf  # avoid division by zero
    power_spectrum = rr2 ** (-powerlaw / 2.0)

    phase = np.exp(2j * np.pi * np.random.rand(*power_spectrum.shape))
    fft_image = power_spectrum * phase

    image = np.fft.irfft2(fft_image, s=(imsize, imsize))
    image = np.real(image)
    return image


@delayed
def delayed_model_insertion(template, line_flux, continuum):
    """Inserts a model slice into the datacube."""
    return template * (line_flux + continuum)


def insert_model(datacube, model_type, line_fluxes, continuum, **kwargs):
    """Inserts a specified model into the datacube using Dask for parallelization."""
    n_px = datacube.n_px_x
    n_chan = datacube.n_channels
    delayed_slices = []
    gs = np.zeros(n_chan)
    z_idxs = np.arange(n_chan)
    for z in range(len(line_fluxes)):
        gs += gaussian(z_idxs, line_fluxes[z], kwargs['pos_z'][z], kwargs['fwhm_z'][z])

    if model_type == "pointlike":
        pos_x, pos_y = kwargs["pos_x"], kwargs["pos_y"]
        for z in range(n_chan):
            slice_data = np.zeros((n_px, n_px))
            slice_data[pos_x, pos_y] = 1
            delayed_slices.append(
                delayed_model_insertion(slice_data, gs[z], continuum[z])
            )

    elif model_type == "gaussian":
        pos_x, pos_y = kwargs["pos_x"], kwargs["pos_y"]
        fwhm_x, fwhm_y = kwargs["fwhm_x"], kwargs["fwhm_y"]
        angle = kwargs.get("angle", 0)
        x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
        template = np.exp(
            -(
                ((x - pos_x) * np.cos(angle) - (y - pos_y) * np.sin(angle)) ** 2 / (2 * (fwhm_x / 2.35482) ** 2)
                + ((x - pos_x) * np.sin(angle) + (y - pos_y) * np.cos(angle)) ** 2 / (2 * (fwhm_y / 2.35482) ** 2)
            )
        )
        for z in range(n_chan):
            delayed_slices.append(
                delayed_model_insertion(template, gs[z], continuum[z])
            )

    elif model_type == "galaxy_zoo":
        template = load_template(kwargs.get("data_path"), n_px)
        if template is None:
            template = np.ones((n_px, n_px), dtype=np.float32) / n_px
            print("Warning: Using default uniform template.")

        for z in range(n_chan):
            delayed_slices.append(
                delayed_model_insertion(template, gs[z], continuum[z])
            )

    elif model_type == "diffuse":
        template = diffuse_signal(n_px)
        template -= np.min(template)
        template /= np.max(template)

        for z in range(n_chan):
            delayed_slices.append(
                delayed_model_insertion(template, line_fluxes[z], continuum[z])
            )

    elif model_type == "molecular_cloud":
        template = molecular_cloud(n_px)
        template -= np.min(template)
        template /= np.max(template)

        for z in range(n_chan):
            delayed_slices.append(
                delayed_model_insertion(template, line_fluxes[z], continuum[z])
            )

    elif model_type == "hubble":
        data_path = kwargs["data_path"]
        files = [file for file in os.listdir(data_path) if not file.startswith(".")]
        img_path = os.path.join(data_path, random.choice(files))
        template = load_template(data_path, n_px)
        if template is None:
            template = np.ones((n_px, n_px), dtype=np.float32) / n_px
            print("Warning: Using default uniform template.")
        for z in range(n_chan):
            delayed_slices.append(

                delayed_model_insertion(template, line_fluxes[z], continuum[z])
            )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    computed_slices = compute(*delayed_slices)
    datacube._array = np.stack(computed_slices) * u.Jy / u.pix**2
    return datacube


def generate_datacube(n_chan, n_px):
    """Generate an empty datacube using the Martini DataCube class."""
    datacube = DataCube(n_channels=n_chan, n_px_x=n_px, n_px_y=n_px)
    datacube._array = np.zeros((n_chan, n_px, n_px)) * u.Jy / u.pix**2
    return datacube


def compute_rest_frequency(redshift, observed_frequency):
    """Compute the rest frequency from the redshift."""
    return observed_frequency * (1 + redshift)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_1d(bb1, bb2):
    assert bb1["z1"] < bb1["z2"]
    assert bb2["z1"] < bb2["z2"]
    z_left = max(bb1["z1"], bb2["z1"])
    z_right = min(bb1["z2"], bb2["z2"])
    if z_right < z_left:
        return 0.0
    intersection = z_right - z_left
    bb1_area = bb1["z2"] - bb1["z1"]
    bb2_area = bb2["z2"] - bb2["z1"]
    union = bb1_area + bb2_area - intersection
    return intersection / union

def get_pos(x_radius, y_radius, z_radius):
    x = np.random.randint(-x_radius, x_radius)
    y = np.random.randint(-y_radius, y_radius)
    z = np.random.randint
    return x, y, z

def get_iou(bb1, bb2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    return intersection_area / float(bb1_area + bb2_area - intersection_area)

def get_iou_1d(bb1, bb2):
    """Calculate the IoU for 1D intervals."""
    z_left = max(bb1["z1"], bb2["z1"])
    z_right = min(bb1["z2"], bb2["z2"])
    if z_right < z_left:
        return 0.0
    intersection = z_right - z_left
    bb1_area = bb1["z2"] - bb1["z1"]
    bb2_area = bb2["z2"] - bb2["z1"]
    return intersection / (bb1_area + bb2_area - intersection)

def get_pos(x_radius, y_radius, z_radius):
    """Generate random positions within the given radius."""
    x = np.random.randint(-x_radius, x_radius)
    y = np.random.randint(-y_radius, y_radius)
    z = np.random.randint(-z_radius, z_radius)
    return x, y, z

def sample_positions(
    pos_x,
    pos_y,
    pos_z,
    n_components,
    xy_radius,
    z_radius,
    sep_xy,
    sep_z,
):
    """Sample valid positions for components within constraints."""
    sample = []
    i = 0
    n = 0
    while len(sample) < n_components and n < 1000:
        new_p = (
            np.random.randint(-xy_radius, xy_radius) + pos_x,
            np.random.randint(-xy_radius, xy_radius) + pos_y,
            np.random.randint(-z_radius, z_radius) + pos_z,
        )

        if len(sample) == 0:
            sample.append(new_p)
        else:
            spatial_distances = [
                np.sqrt((new_p[0] - p[0]) ** 2 + (new_p[1] - p[1]) ** 2) for p in sample
            ]
            freq_distances = [abs(new_p[2] - p[2]) for p in sample]
            if all(sd > sep_xy and fd > sep_z for sd, fd in zip(spatial_distances, freq_distances)):
                sample.append(new_p)
            else:
                n += 1
                continue
        i += 1
        n = 0

    return sample


def insert_serendipitous(
    datacube,
    continuum,
    line_fluxes,
    line_names,
    line_frequencies,
    pos_zs,
    fwhm_x,
    fwhm_y,
    fwhm_zs,
    n_px,
    n_chan,
    xy_radius,
    z_radius,
    sep_xy,
    sep_z,
    freq_sup,
    sim_params_path,
):
    """Insert serendipitous sources into the datacube."""
    n_sources = np.random.randint(1, 5)
    fwhm_xs = np.random.randint(1, fwhm_x, n_sources)
    fwhm_ys = np.random.randint(1, fwhm_y, n_sources)
    n_lines = np.random.randint(1, 3, n_sources)
    x, y = np.meshgrid(np.arange(n_px), np.arange(n_px))
    sample_coords = sample_positions(
        pos_x=n_px // 2,
        pos_y=n_px // 2,
        pos_z=n_chan // 2,
        n_components=n_sources,
        xy_radius=xy_radius,
        z_radius=z_radius,
        sep_xy=sep_xy,
        sep_z=sep_z,
    )

    with (open(sim_params_path, "a") as f):
        f.write(f"\n Injected {n_sources} serendipitous sources\n")
        for c_id, (pos_x, pos_y, pos_z) in enumerate(sample_coords):
            n_line = n_lines[c_id]
            peak_line_fluxes = np.random.uniform(np.min(line_fluxes), np.max(line_fluxes), n_line)
            s_line_names = line_names[:n_line]
            s_freq = [line_freq + (pos_z - pos_zs[0]) * freq_sup for line_freq in line_frequencies[:n_line]]
            fwhmsz = np.random.randint(2, np.max(fwhm_zs), n_line)
            angle = np.random.uniform(0, 2 * np.pi)
            z_idxs = np.arange(n_chan)
            gs = np.zeros(n_chan)
            for i in range(len(peak_line_fluxes)):
                gs += gaussian(z_idxs, peak_line_fluxes[c_id], pos_z[i], fwhmsz[i])
            f.write(f"RA: {pos_x}, DEC: {pos_y}\n")
            f.write(f"FWHM_x (pixels): {fwhm_xs[c_id]}\n")
            f.write(f"FWHM_y (pixels): {fwhm_ys[c_id]}\n")
            template = np.exp(
                -(
                        ((x - pos_x) * np.cos(angle) - (y - pos_y) * np.sin(angle)) ** 2 / (2 * (fwhm_x / 2.35482) ** 2)
                        + ((x - pos_x) * np.sin(angle) + (y - pos_y) * np.cos(angle)) ** 2 / (
                                    2 * (fwhm_y / 2.35482) ** 2)
                )
            )

            for i in range(len(s_freq)):
                f.write(
                    f"Line: {s_line_names[i]} - Frequency: {s_freq[i]} GHz - Flux: {peak_line_fluxes[i]} Jy - Width (Channels): {fwhmsz[i]}\n"
                )
            delayed_slices = []
            for z in range(n_chan):
                delayed_slices.append(
                    delayed_model_insertion(template, gs[z], continuum[z])
                )

            computed_slices = compute(*delayed_slices)
            datacube._array += np.stack(computed_slices)
    datacube._array = datacube._array * u.Jy / u.pix ** 2
    return datacube
