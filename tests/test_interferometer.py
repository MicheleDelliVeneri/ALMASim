import pytest
from pytestqt.qtbot import QtBot
import almasim.alma as alma
import almasim.skymodels as skymodels
import almasim.astro as astro
import almasim.interferometer as interferometer
import almasim.ui as ui
from pathlib import Path
import inspect
import faulthandler
import os
import pandas as pd
import astropy.units as U
import numpy as np

faulthandler.enable()
os.environ["LC_ALL"] = "C"


@pytest.fixture
def test_interferometer(qtbot: QtBot):
    almasim = ui.ALMASimulator()
    qtbot.addWidget(almasim)
    main_path = os.path.sep + os.path.join(
        *str(Path(inspect.getfile(inspect.currentframe())).resolve()).split(
            os.path.sep
        )[:-2]
    )
    metadata_path = os.path.join(main_path, "almasim", "metadata", "qso_metadata.csv")
    line_path = os.path.join(main_path, "almasim")
    metadata = pd.read_csv(metadata_path)
    rest_frequency, line_names = astro.get_line_info(line_path)
    metadata = almasim.sample_given_redshift(metadata, 1, rest_frequency, False, None)
    metadata = metadata.iloc[0]
    assert len(metadata) > 0
    antenna_array = metadata["antenna_arrays"]
    ra = metadata["RA"]
    dec = metadata["Dec"]
    fov = metadata["FOV"]
    ang_res = metadata["Ang.res."]
    vel_res = metadata["Vel.res."]
    int_time = metadata["Int.Time"]
    freq = metadata["Freq"]
    freq_support = metadata["Freq.sup."]
    cont_sens = metadata["Cont_sens_mJybeam"]
    source_name = metadata["ALMA_source_name"]
    member_ouid = metadata["member_ous_uid"]
    alma.generate_antenna_config_file_from_antenna_array(
        antenna_array, os.path.join(main_path, "almasim"), main_path
    )
    antennalist = os.path.join(main_path, "antenna.cfg")
    ra = ra * U.deg
    dec = dec * U.deg
    fov = fov * 3600 * U.arcsec
    ang_res = ang_res * U.arcsec
    vel_res = vel_res * U.km / U.s
    int_time = int_time * U.s
    source_freq = freq * U.GHz

    band_range, central_freq, t_channels, delta_freq = almasim.freq_supp_extractor(
        freq_support, source_freq
    )
    max_baseline = alma.get_max_baseline_from_antenna_config(None, antennalist) * U.km
    beam_size = alma.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=False
    )
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    cont_sens = cont_sens * U.mJy / (U.arcsec**2)
    cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
    cont_sens = cont_sens_jy
    cell_size = beam_size / 5
    n_pix = 256
    n_channels = 256

    rest_frequency = metadata["rest_frequency"]
    redshift = metadata["redshift"]
    lum_infrared = 1e10
    source_type = "point"
    (
        continum,
        line_fluxes,
        line_names,
        redshift,
        line_frequency,
        source_channel_index,
        n_channels_nw,
        bandwidth,
        freq_sup_nw,
        cont_frequencies,
        fwhm_z,
        lum_infrared,
    ) = almasim.process_spectral_data(
        source_type,
        os.path.join(main_path, "almasim"),
        redshift,
        central_freq.value,
        band_range.value,
        source_freq.value,
        n_channels,
        lum_infrared,
        cont_sens.value,
        None,
        None,
        False,
    )
    datacube = skymodels.DataCube(
        n_px_x=n_pix,
        n_px_y=n_pix,
        n_channels=n_channels,
        px_size=cell_size,
        channel_width=delta_freq,
        spectral_centre=central_freq,
        ra=ra,
        dec=dec,
    )
    model = datacube._array.to_value(datacube._array.unit).T
    assert model.shape[0] > 0
    wcs = datacube.wcs
    if n_channels_nw != n_channels:
        freq_sup = freq_sup_nw * U.MHz
        n_channels = n_channels_nw
        band_range = n_channels * freq_sup
    datacube = skymodels.DataCube(
        n_px_x=n_pix,
        n_px_y=n_pix,
        n_channels=n_channels,
        px_size=cell_size,
        channel_width=delta_freq,
        spectral_centre=central_freq,
        ra=ra,
        dec=dec,
    )
    # testing point model
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
    pos_z = [int(index) for index in source_channel_index]
    datacube = skymodels.insert_pointlike(
        None,
        datacube,
        continum,
        line_fluxes,
        int(pos_x),
        int(pos_y),
        pos_z,
        fwhm_z,
        n_channels,
    )
    model = datacube._array.to_value(datacube._array.unit).T
    obs_date = metadata["Obs.date"]
    header = skymodels.get_datacube_header(datacube, obs_date)
    second2hour = 1 / 3600
    save_mode = "npz"
    inter = interferometer.Interferometer(
        0,
        model,
        os.path.join(main_path, "almasim"),
        main_path,
        ra,
        dec,
        central_freq,
        band_range,
        fov,
        antenna_array,
        cont_sens.value,
        int_time.value * second2hour,
        obs_date,
        header,
        save_mode,
        None,
        False, 
        0,
    )
    simulation_results = inter.run_interferometric_sim()
    assert simulation_results is not None
    os.remove(os.path.join(main_path, "antenna.cfg"))


def test(test_interferometer):
    return test_interferometer


if __name__ == "__main__":
    pytest.main(["-v", __file__])
