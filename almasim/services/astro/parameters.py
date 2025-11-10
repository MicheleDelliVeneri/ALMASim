"""Simulation parameter writing functions."""
import numpy as np


def write_sim_parameters(
    path,
    source_name,
    member_ouid,
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
    fwhm_x=None,
    fwhm_y=None,
    angle=None,
):
    """Write simulation parameters to a text file."""
    with open(path, "w") as f:
        f.write("Simulation Parameters:\n")
        f.write("Source Name: {}\n".format(source_name))
        f.write('Member OUID: "{}"\n'.format(member_ouid))
        f.write("RA: {}\n".format(ra))
        f.write("DEC: {}\n".format(dec))
        f.write("Band: {}\n".format(band))
        f.write("Bandwidth {}\n".format(band_range))
        f.write("Band Central Frequency: {}\n".format(central_freq))
        f.write("Pixel size: {}\n".format(cell_size))
        f.write("Beam Size: {}\n".format(beam_size))
        f.write("Fov: {}\n".format(fov))
        f.write("Angular Resolution: {}\n".format(ang_res))
        f.write("Velocity Resolution: {}\n".format(vel_res))
        f.write("Redshift: {}\n".format(redshift))
        f.write("Integration Time: {}\n".format(int_time))
        f.write("Cube Size: {} x {} x {} pixels\n".format(n_pix, n_pix, n_channels))
        f.write("Mean Continum Flux: {}\n".format(np.mean(continum)))
        f.write("Infrared Luminosity: {}\n".format(lum_infrared))
        if source_type == "gaussian":
            f.write("FWHM_x (pixels): {}\n".format(fwhm_x))
            f.write("FWHM_y (pixels): {}\n".format(fwhm_y))
        if (source_type == "gaussian") or (source_type == "extended"):
            f.write("Projection Angle: {}\n".format(angle))
        for i in range(len(line_fluxes)):
            f.write(
                f"Line: {line_names[i]} - Frequency: {line_frequencies[i]} GHz "
                f"- Flux: {line_fluxes[i]} Jy  - Width (Channels): {fwhm_z[i]}\n"
            )
        if snapshot is not None:
            f.write("TNG Snapshot ID: {}\n".format(snapshot))
            f.write("TNG Subhalo ID: {}\n".format(subhalo))
        f.close()


