import os
from pathlib import Path

import astropy.units as U
import numpy as np
import pandas as pd

from almasim.services.metadata.tap import service as alma
from almasim.services import astro
from almasim.services import interferometry as interferometer
from almasim import skymodels
from almasim.services import simulation as sim


def _load_metadata_row():
    repo_root = Path(__file__).resolve().parents[1]
    main_dir = repo_root / "src" / "almasim"
    metadata = pd.read_csv(repo_root / "data" / "qso_metadata.csv")
    rest_frequency, _ = astro.get_line_info(main_dir)
    sample = sim.sample_given_redshift(metadata, 1, rest_frequency, False, None)
    return main_dir, sample.iloc[0]


def test_interferometer_runs(tmp_path):
    main_dir, metadata = _load_metadata_row()
    antenna_array = metadata["antenna_arrays"]
    ra = metadata["RA"] * U.deg
    dec = metadata["Dec"] * U.deg
    fov = metadata["FOV"] * 3600 * U.arcsec
    ang_res = metadata["Ang.res."] * U.arcsec
    vel_res = metadata["Vel.res."] * U.km / U.s
    int_time = metadata["Int.Time"] * U.s
    freq = metadata["Freq"] * U.GHz
    freq_support = metadata["Freq.sup."]
    cont_sens = metadata["Cont_sens_mJybeam"] * U.mJy / (U.arcsec**2)

    alma.generate_antenna_config_file_from_antenna_array(
        antenna_array, str(main_dir), str(main_dir.parent)
    )
    antennalist = main_dir.parent / "antenna.cfg"

    from almasim.services.interferometry.frequency import freq_supp_extractor
    band_range, central_freq, t_channels, delta_freq = freq_supp_extractor(
        freq_support, freq
    )
    max_baseline = alma.get_max_baseline_from_antenna_config(None, antennalist) * U.km
    beam_size = alma.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=False
    )
    beam_area = 1.1331 * beam_size**2
    cont_sens_jy = (cont_sens * (np.pi * (beam_size / 2) ** 2)).to(U.Jy)
    cell_size = beam_size / 5
    n_pix = 128
    n_channels = 64

    rest_frequency = metadata["rest_frequency"]
    redshift = metadata["redshift"]
    lum_infrared = 1e10
    source_type = "point"
    from almasim.services.astro.spectral import process_spectral_data
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
    ) = process_spectral_data(
        source_type,
        str(main_dir),
        redshift,
        central_freq.value,
        band_range.value,
        freq.value,
        n_channels,
        lum_infrared,
        cont_sens_jy.value,
        line_names=None,
        n_lines=None,
        remote=False,
        line_width_range=(50.0, 300.0),
    )

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

    pos_x, pos_y, _ = datacube.wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
    pos_z = [int(index) for index in source_channel_index]
    model = skymodels.PointlikeSkyModel(
        datacube=datacube,
        continuum=continum,
        line_fluxes=line_fluxes,
        pos_x=int(pos_x),
        pos_y=int(pos_y),
        pos_z=pos_z,
        fwhm_z=fwhm_z,
        n_chan=n_channels,
    )
    datacube = model.insert()
    model = datacube._array.to_value(datacube._array.unit).T / beam_area.value

    obs_date = metadata["Obs.date"]
    header = skymodels.get_datacube_header(datacube, obs_date)
    second2hour = 1 / 3600
    snr = 1.3
    inter = interferometer.Interferometer(
        idx=0,
        skymodel=model,
        client=None,
        main_dir=str(main_dir),
        output_dir=str(tmp_path),
        ra=ra,
        dec=dec,
        central_freq=central_freq,
        bandwidth=band_range,
        fov=fov,
        antenna_array=antenna_array,
        noise=(np.min(line_fluxes) / beam_area.value) / snr,
        snr=snr,
        integration_time=int_time.value * second2hour,
        observation_date=obs_date,
        header=header,
        save_mode="npz",
        robust=0,
        terminal=None,
    )

    simulation_results = inter.run_interferometric_sim()
    assert simulation_results is not None
