"""Serendipitous sources utilities."""
import math
import numpy as np
from typing import Optional, Any
from dask.distributed import Client

from .gaussian import GaussianSkyModel


def distance_1d(p1: float, p2: float) -> float:
    """Calculate 1D distance."""
    return math.sqrt((p1 - p2) ** 2)


def distance_2d(p1: tuple, p2: tuple) -> float:
    """Calculate 2D distance."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_iou(bb1: dict, bb2: dict) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_1d(bb1: dict, bb2: dict) -> float:
    """Calculate 1D IoU."""
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


def get_pos(x_radius: int, y_radius: int, z_radius: int) -> tuple:
    """Get random position within radius."""
    x = np.random.randint(-x_radius, x_radius)
    y = np.random.randint(-y_radius, y_radius)
    z = np.random.randint(-z_radius, z_radius)
    return (x, y, z)


def sample_positions(
    terminal: Optional[Any],
    pos_x: float,
    pos_y: float,
    pos_z: int,
    fwhm_x: int,
    fwhm_y: int,
    fwhm_z: float,
    n_components: int,
    fwhm_xs: np.ndarray,
    fwhm_ys: np.ndarray,
    fwhm_zs: np.ndarray,
    xy_radius: float,
    z_radius: float,
    sep_xy: int,
    sep_z: int,
) -> list:
    """Sample positions for serendipitous sources."""
    sample = []
    i = 0
    n = 0
    while (len(sample) < n_components) and (n < 1000):
        new_p = get_pos(int(xy_radius), int(xy_radius), int(z_radius))
        new_p = int(new_p[0] + pos_x), int(new_p[1] + pos_y), int(new_p[2] + pos_z)
        if len(sample) == 0:
            spatial_dist = distance_2d((new_p[0], new_p[1]), (pos_x, pos_y))
            freq_dist = distance_1d(new_p[2], pos_z)
            if spatial_dist < sep_xy or freq_dist < sep_z:
                n += 1
                continue
            else:
                spatial_iou = get_iou(
                    {
                        "x1": new_p[0] - fwhm_xs[i],
                        "x2": new_p[0] + fwhm_xs[i],
                        "y1": new_p[1] - fwhm_ys[i],
                        "y2": new_p[1] + fwhm_ys[i],
                    },
                    {
                        "x1": pos_x - fwhm_x,
                        "x2": pos_x + fwhm_x,
                        "y1": pos_y - fwhm_y,
                        "y2": pos_y + fwhm_y,
                    },
                )
                freq_iou = get_iou_1d(
                    {"z1": new_p[2] - fwhm_zs[i], "z2": new_p[2] + fwhm_zs[i]},
                    {"z1": pos_z - fwhm_z, "z2": pos_z + fwhm_z},
                )
                if spatial_iou > 0.1 or freq_iou > 0.1:
                    n += 1
                    continue
                else:
                    sample.append(new_p)
                    i += 1
                    n = 0
                    if terminal is not None:
                        terminal.add_log("Found {}st component".format(len(sample)))
        else:
            spatial_distances = [
                distance_2d((new_p[0], new_p[1]), (p[0], p[1])) for p in sample
            ]
            freq_distances = [distance_1d(new_p[2], p[2]) for p in sample]
            checks = [
                spatial_dist < sep_xy or freq_dist < sep_z
                for spatial_dist, freq_dist in zip(spatial_distances, freq_distances)
            ]
            if any(checks) is True:
                n += 1
                continue
            else:
                spatial_iou = [
                    get_iou(
                        {
                            "x1": new_p[0] - fwhm_xs[i],
                            "x2": new_p[0] + fwhm_xs[i],
                            "y1": new_p[1] - fwhm_ys[i],
                            "y2": new_p[1] + fwhm_ys[i],
                        },
                        {
                            "x1": p[0] - fwhm_xs[j],
                            "x2": p[0] + fwhm_xs[j],
                            "y1": p[1] - fwhm_ys[j],
                            "y2": p[1] + fwhm_ys[j],
                        },
                    )
                    for j, p in enumerate(sample)
                ]
                freq_iou = [
                    get_iou_1d(
                        {"z1": new_p[2] - fwhm_zs[i], "z2": new_p[2] + fwhm_zs[i]},
                        {"z1": p[2] - fwhm_zs[j], "z2": p[2] + fwhm_zs[j]},
                    )
                    for j, p in enumerate(sample)
                ]
                checks = [
                    spatial_iou > 0.1 or freq_iou > 0.1
                    for spatial_iou, freq_iou in zip(spatial_iou, freq_iou)
                ]
                if any(checks) is True:
                    n += 1
                    continue
                else:
                    i += 1
                    n = 0
                    sample.append(new_p)
                    if terminal is not None:
                        terminal.add_log("Found {}st component".format(len(sample)))

    return sample


def insert_serendipitous(
    terminal: Optional[Any],
    client: Client,
    update_progress: Optional[Any],
    datacube: Any,
    continum: np.ndarray,
    cont_sens: float,
    line_fluxes: np.ndarray,
    line_names: np.ndarray,
    line_frequencies: np.ndarray,
    freq_sup: float,
    pos_zs: list[int],
    fwhm_x: int,
    fwhm_y: int,
    fwhm_zs: list[float],
    n_px: int,
    n_chan: int,
    sim_params_path: str,
) -> Any:
    """Insert serendipitous sources into datacube."""
    wcs = datacube.wcs
    xy_radius = n_px / 4
    z_radius = n_chan / 2
    n_sources = np.random.randint(1, 5)
    # Generate fwhm for x and y
    fwhm_xs = np.random.randint(1, fwhm_x, n_sources)
    fwhm_ys = np.random.randint(1, fwhm_y, n_sources)
    # generate a random number of lines for each serendipitous source
    if len(line_fluxes) == 1:
        n_lines = np.array([1] * n_sources)
    else:
        n_lines = np.random.randint(1, 3, n_sources)
    # generate the width of the first line based on the first line of the central source
    if fwhm_zs[0] < 2:
        fwhm_zs[0] = 2
    s_fwhm_zs = np.random.randint(2, int(fwhm_zs[0]), n_sources)
    # get posx and poy of the centtral source
    pos_x, pos_y, _ = datacube.wcs.sub(3).wcs_world2pix(
        datacube.ra, datacube.dec, datacube.spectral_centre, 0
    )
    # get a mininum separation based on spatial dimensions
    sep_x, sep_z = np.random.randint(0, int(xy_radius)), np.random.randint(0, int(z_radius))
    # get the position of the first line of the central source
    pos_z = pos_zs[0]
    # get maximum continum value
    cont_peak = np.max(continum)
    # get serendipitous continum maximum
    serendipitous_norms = np.random.uniform(cont_sens, cont_peak, n_sources)
    # normalize continum to each serendipitous continum maximum
    serendipitous_conts = np.array(
        [
            continum * serendipitous_norm / cont_peak
            for serendipitous_norm in serendipitous_norms
        ]
    )
    # sample coordinates of the first line
    sample_coords = sample_positions(
        terminal,
        pos_x,
        pos_y,
        pos_z,
        fwhm_x,
        fwhm_y,
        fwhm_zs[0],
        n_sources,
        fwhm_xs,
        fwhm_ys,
        s_fwhm_zs,
        xy_radius,
        z_radius,
        sep_x,
        sep_z,
    )
    # get the rotation angles
    pas = np.random.randint(0, 360, n_sources)
    with open(sim_params_path, "a") as f:
        f.write("\n Injected {} serendipitous sources\n".format(n_sources))
        for c_id, choords in enumerate(sample_coords):
            n_line = n_lines[c_id]
            if terminal is not None:
                terminal.add_log(
                    "Simulating serendipitous source {} with {} lines".format(
                        c_id + 1, n_line
                    )
                )
            s_line_fluxes = np.random.uniform(cont_sens, np.max(line_fluxes), n_line)
            s_line_names = line_names[:n_line]
            if terminal is not None:
                for s_name, s_flux in zip(s_line_names, s_line_fluxes):
                    terminal.add_log("Line {} Flux: {}".format(s_name, s_flux))
            pos_x, pos_y, pos_z = choords
            delta = pos_z - pos_zs[0]
            pos_z = np.array([pos + delta for pos in pos_zs])[:n_line]
            s_ra, s_dec, _ = wcs.sub(3).wcs_pix2world(pos_x, pos_y, 0, 0)
            s_freq = np.array(
                [line_freq + delta * freq_sup for line_freq in line_frequencies]
            )[:n_line]
            fwhmsz = [s_fwhm_zs[0]]
            for _ in range(n_line - 1):
                fwhmsz.append(np.random.randint(2, np.random.choice(fwhm_zs, 1))[0])
            s_continum = serendipitous_conts[c_id]
            f.write("RA: {}\n".format(s_ra))
            f.write("DEC: {}\n".format(s_dec))
            f.write("FWHM_x (pixels): {}\n".format(fwhm_xs[c_id]))
            f.write("FWHM_y (pixels): {}\n".format(fwhm_ys[c_id]))
            f.write("Projection Angle: {}\n".format(pas[c_id]))
            for i in range(len(s_freq)):
                f.write(
                    f"Line: {s_line_names[i]} - Frequency: {s_freq[i]} GHz "
                    f"- Flux: {line_fluxes[i]} Jy - Width (Channels): {fwhmsz[i]}\n"
                )
            # Use GaussianSkyModel to insert serendipitous source
            gaussian_model = GaussianSkyModel(
                datacube=datacube,
                continuum=s_continum,
                line_fluxes=s_line_fluxes,
                pos_x=int(pos_x),
                pos_y=int(pos_y),
                pos_z=pos_z.tolist(),
                fwhm_x=int(fwhm_xs[c_id]),
                fwhm_y=int(fwhm_ys[c_id]),
                fwhm_z=fwhmsz,
                angle=int(pas[c_id]),
                n_px=n_px,
                n_chan=n_chan,
                client=client,
                update_progress=update_progress,
            )
            datacube = gaussian_model.insert()
    return datacube


