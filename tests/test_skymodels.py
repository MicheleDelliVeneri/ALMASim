import pytest
from pytestqt.qtbot import QtBot
import almasim.alma as alma
import almasim.skymodels as skymodels
import almasim.astro as astro
import almasim.ui as ui
from pathlib import Path
import inspect
import faulthandler
import os
import pandas as pd
import astropy.units as U
import numpy as np

faulthandler.enable()

@pytest.fixture
def test_skymodels(qtbot: QtBot):
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
    ra = metadata['RA']
    dec = metadata['Dec']
    fov = metadata['FOV']
    ang_res = metadata['Ang.res.']
    vel_res = metadata['Vel.res.']
    int_time = metadata['Int.Time']
    freq = metadata['Freq']
    freq_support = metadata['Freq.sup.']
    cont_sens = metadata['Cont_sens_mJybeam']
    alma.generate_antenna_config_file_from_antenna_array(
        antenna_array, os.path.join(main_path, "almasim"), main_path
    )
    antennalist = os.path.join(main_path, "antenna.cfg")
    second2hour = 1 / 3600
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
    
    if isinstance(rest_frequency, np.ndarray):
        rest_frequency = np.sort(np.array(rest_frequency))[0]
        rest_frequency = rest_frequency * U.GHz
        redshift = astro.compute_redshift(rest_frequency, source_freq)
    else:
        rest_frequency = (
                astro.compute_rest_frequency_from_redshift(
            os.path.join(main_dir, "almasim"), source_freq.value, redshift
            ) * U.GHz)

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
            main_path,
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
        velocity_centre=central_freq,
        ra=ra,
        dec=dec,
        )
    wcs = datacube.wcs
    fwhm_x, fwhm_y, angle = None, None, None
    if n_channels_nw != n_channels:
        freq_sup = freq_sup_nw * U.MHz
        n_channels = n_channels_nw
        band_range = n_channels * freq_sup

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
    pos_z = [int(index) for index in source_channel_index]
    fwhm_x = np.random.randint(3, 10)
    fwhm_y = np.random.randint(3, 10)
    angle = np.random.randint(0, 180)
    datacube = skymodels.insert_gaussian(
                None,
                datacube,
                continum,
                line_fluxes,
                int(pos_x),
                int(pos_y),
                pos_z,
                fwhm_x,
                fwhm_y,
                fwhm_z,
                angle,
                n_pix,
                n_channels,
            )
    datacube = skymodels.insert_diffuse(
                None,
                datacube,
                continum,
                line_fluxes,
                pos_z,
                fwhm_z,
                n_pix,
                n_channels,
            )
    #alaxy_zoo_path = os.path.join(os.path.expanduser('~'), 'TNGData')
    #almasim.galaxy_zoo_entry.setText(galaxy_zoo_path)
    #if not os.path.exists(galaxy_zoo_path):
    #    os.mkdir(galaxy_zoo_path)
    #almasim.download_galaxy_zoo()
    #snapshot = astro.redshift_to_snapshot(redshift)
    #tng_subhaloid = astro.get_subhaloids_from_db(
    #            1, line_path, snapshot
    #        )
    #tng_api_key = "8f578b92e700fae3266931f4d785f82c"
    #outpath = os.path.join(
    #    main_path, "TNG100-1", "output", "snapdir_0{}".format(snapshot)
    #        )
    #part_num = uas.get_particles_num(
    #            main_path, outpath, snapshot, int(tng_subhaloid), tng_api_key
    #        )
    
def test(test_skymodels):
    return test_skymodels

if __name__ == "__main__":
    pytest.main(["-v", __file__])

