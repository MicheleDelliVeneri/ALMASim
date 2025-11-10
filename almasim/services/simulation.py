"""Simulation helpers decoupled from the legacy PyQt UI."""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import astropy.units as U

from . import interferometry as uin
from . import astro as uas
from .interferometry import antenna as ual_antenna
from .interferometry.frequency import freq_supp_extractor
from .interferometry.utils import closest_power_of_2
from .utils import log_message, as_progress_emitter
from .astro.spectral import process_spectral_data
from .. import skymodels as usm

LogFn = Optional[Callable[[str], None]]


@dataclass
class SimulationParams:
    idx: int
    source_name: str
    member_ouid: str
    main_dir: str
    output_dir: str
    tng_dir: str
    galaxy_zoo_dir: str
    hubble_dir: str
    project_name: str
    ra: float
    dec: float
    band: float
    ang_res: float
    vel_res: float
    fov: float
    obs_date: str
    pwv: float
    int_time: float
    bandwidth: float
    freq: float
    freq_support: str
    cont_sens: float
    antenna_array: str
    n_pix: Optional[float]
    n_channels: Optional[int]
    source_type: str
    tng_api_key: Optional[str]
    ncpu: int
    rest_frequency: Any
    redshift: Any
    lum_infrared: Any
    snr: float
    n_lines: Any
    line_names: Any
    save_mode: str
    inject_serendipitous: bool
    remote: bool

    @classmethod
    def from_metadata_row(
        cls,
        row: pd.Series | Mapping[str, Any],
        *,
        idx: int,
        main_dir: Path | str,
        output_dir: Path | str,
        tng_dir: Path | str,
        galaxy_zoo_dir: Path | str,
        hubble_dir: Path | str,
        project_name: str,
        source_type: str = "point",
        snr: float = 1.3,
        save_mode: str = "npz",
        n_pix: Optional[float] = None,
        n_channels: Optional[int] = None,
        n_lines: Optional[Any] = None,
        line_names: Optional[Any] = None,
        rest_frequency: Optional[Any] = None,
        redshift: Optional[Any] = None,
        lum_infrared: Optional[Any] = None,
        ncpu: Optional[int] = None,
        tng_api_key: Optional[str] = None,
        inject_serendipitous: bool = False,
        remote: bool = False,
    ) -> "SimulationParams":
        """Build :class:`SimulationParams` from a metadata row."""

        def _as_dict(obj):
            if hasattr(obj, "to_dict"):
                return dict(obj.to_dict())
            if isinstance(obj, Mapping):
                return dict(obj)
            return dict(obj)

        payload = _as_dict(row)
        missing = object()

        def _resolve_path(path_like: Path | str) -> str:
            return str(Path(path_like).expanduser().resolve())

        def _get(keys, *, required=True, default=None):
            keys_seq = [keys] if isinstance(keys, str) else list(keys)
            for key in keys_seq:
                value = payload.get(key, missing)
                if value is missing:
                    continue
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                return value
            if required and default is None:
                raise KeyError(f"Missing required metadata column(s): {keys_seq}")
            return default

        def _float(keys):
            value = _get(keys)
            return float(value)

        source_name = _get(["ALMA_source_name", "source_name", "target_name"])
        member_ouid = _get(["member_ous_uid", "member_ouid"])
        band = _float(["Band", "band"])
        ra = _float("RA")
        dec = _float("Dec")
        ang_res = _float(["Ang.res.", "ang_res"])
        vel_res = _float(["Vel.res.", "vel_res"])
        fov = _float(["FOV", "fov"])
        obs_date = str(_get(["Obs.date", "obs_date"]))
        pwv = _float(["PWV", "pwv"])
        int_time = _float(["Int.Time", "int_time"])
        bandwidth = _float(["Bandwidth", "bandwidth"])
        freq = _float(["Freq", "frequency"])
        freq_support = str(_get(["Freq.sup.", "frequency_support"]))
        cont_sens = _float(["Cont_sens_mJybeam", "cont_sens"])
        antenna_array = str(_get(["antenna_arrays", "antenna_array"]))

        if rest_frequency is None:
            rest_frequency = _get("rest_frequency", required=False)
        if redshift is None:
            redshift = _get("redshift", required=False)
        if lum_infrared is None:
            lum_infrared = _get("lum_infrared", required=False)
        if n_lines is None:
            n_lines = _get("n_lines", required=False)
        if line_names is None:
            line_names = _get("line_names", required=False)
        ncpu = ncpu if ncpu is not None else (os.cpu_count() or 1)

        return cls(
            idx=idx,
            source_name=str(source_name),
            member_ouid=str(member_ouid),
            main_dir=_resolve_path(main_dir),
            output_dir=_resolve_path(output_dir),
            tng_dir=_resolve_path(tng_dir),
            galaxy_zoo_dir=_resolve_path(galaxy_zoo_dir),
            hubble_dir=_resolve_path(hubble_dir),
            project_name=project_name,
            ra=ra,
            dec=dec,
            band=band,
            ang_res=ang_res,
            vel_res=vel_res,
            fov=fov,
            obs_date=obs_date,
            pwv=pwv,
            int_time=int_time,
            bandwidth=bandwidth,
            freq=freq,
            freq_support=freq_support,
            cont_sens=cont_sens,
            antenna_array=antenna_array,
            n_pix=n_pix,
            n_channels=n_channels,
            source_type=source_type,
            tng_api_key=tng_api_key,
            ncpu=int(ncpu),
            rest_frequency=rest_frequency,
            redshift=redshift,
            lum_infrared=lum_infrared,
            snr=snr,
            n_lines=n_lines,
            line_names=line_names,
            save_mode=save_mode,
            inject_serendipitous=inject_serendipitous,
            remote=remote,
        )


# Progress emitter functions moved to services.utils


def run_simulation(
    params: SimulationParams,
    *,
    logger: LogFn = None,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_emitter=None,
    dask_client=None,
    terminal_logger=None,
    robust: float = 0.0,
    interferometer_progress_callback: Optional[Callable[[int], None]] = None,
    stop_requested: bool = False,
):
    """Execute the full ALMA simulation workflow."""
    remote = bool(params.remote)

    def log(message: str):
        log_message(logger, message, remote=remote)

    def status(message: str):
        if status_callback is not None:
            status_callback(message)

    def clean(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    line_names = clean(params.line_names)
    if isinstance(line_names, str):
        line_names = [
            token.strip(" \"'")
            for token in line_names.strip("[]").split(",")
            if token.strip()
        ]
    elif isinstance(line_names, np.ndarray):
        line_names = line_names.tolist()
    rest_frequency = clean(params.rest_frequency)
    redshift = clean(params.redshift)
    lum_infrared = clean(params.lum_infrared)
    n_pix = clean(params.n_pix)
    n_channels = clean(params.n_channels)
    n_lines = clean(params.n_lines)
    tng_api_key = clean(params.tng_api_key)

    if isinstance(line_names, float) and math.isnan(line_names):
        line_names = None

    progress_adapter = as_progress_emitter(progress_emitter)
    start = time.time()
    second2hour = 1 / 3600
    ra = params.ra * U.deg
    dec = params.dec * U.deg
    fov = params.fov * 3600 * U.arcsec
    ang_res = params.ang_res * U.arcsec
    vel_res = params.vel_res * U.km / U.s
    int_time = params.int_time * U.s
    source_freq = params.freq * U.GHz

    band_range, central_freq, t_channels, delta_freq = freq_supp_extractor(
        params.freq_support, source_freq
    )
    sim_output_dir = os.path.join(
        params.output_dir, f"{params.project_name}_{params.idx}"
    )
    os.makedirs(sim_output_dir, exist_ok=True)
    os.chdir(params.output_dir)

    if remote:
        print(f"RA: {ra}")
        print(f"DEC: {dec}")
        print(f"Integration Time: {int_time}")
    else:
        log(f"RA: {ra}")
        log(f"DEC: {dec}")
        log(f"Integration Time: {int_time}")

    antennalist = os.path.join(sim_output_dir, "antenna.cfg")
    ual_antenna.generate_antenna_config_file_from_antenna_array(
        params.antenna_array, params.main_dir, sim_output_dir
    )
    status("Computing Max baseline")
    max_baseline = (
        ual_antenna.get_max_baseline_from_antenna_config(progress_adapter, antennalist) * U.km
    )
    if remote:
        print(f"Field of view: {round(fov.value, 3)} arcsec")
    else:
        log(f"Field of view: {round(fov.value, 3)} arcsec")

    beam_size = ual_antenna.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=False
    )
    beam_area = 1.1331 * beam_size**2
    beam_solid_angle = np.pi * (beam_size / 2) ** 2
    cont_sens = params.cont_sens * U.mJy / (U.arcsec**2)
    cont_sens_jy = (cont_sens * beam_solid_angle).to(U.Jy)
    if remote:
        print(f"Minimum detectable continum: {cont_sens_jy}")
    else:
        log(f"Minimum detectable continum: {cont_sens_jy}")

    cell_size = beam_size / 5
    if n_pix is None:
        n_pix = closest_power_of_2(int(1.5 * fov.value / cell_size.value))
    else:
        n_pix = closest_power_of_2(int(n_pix))
        cell_size = fov / n_pix

    if n_channels is None:
        n_channels = t_channels
    else:
        band_range = n_channels * delta_freq

    if redshift is None:
        if isinstance(rest_frequency, (np.ndarray, list)):
            rest_frequency = np.sort(np.array(rest_frequency))[0]
        rest_frequency = rest_frequency * U.GHz
        redshift = uas.compute_redshift(rest_frequency, source_freq)
    else:
        rest_frequency = (
            uas.compute_rest_frequency_from_redshift(
                params.main_dir, source_freq.value, redshift
            )
            * U.GHz
        )

    status("Computing spectral lines and properties")
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
        params.source_type,
        params.main_dir,
        redshift,
        central_freq.value,
        band_range.value,
        source_freq.value,
        n_channels,
        lum_infrared,
        cont_sens_jy.value,
        line_names,
        n_lines,
        remote,
    )
    if n_channels_nw != n_channels:
        freq_sup = freq_sup_nw * U.MHz
        n_channels = n_channels_nw
        band_range = n_channels * freq_sup
    else:
        freq_sup = delta_freq

    if remote:
        print(f"Beam size: {round(beam_size.value, 4)} arcsec\n")
        print(f"Central Frequency: {central_freq}\n")
        print(f"Spectral Window: {round(band_range.value, 3)} GHz\n")
        print(f"Freq Support: {delta_freq}\n")
        print(f"Cube Dimensions: {n_pix} x {n_pix} x {n_channels}\n")
        print(f"Redshift: {round(redshift, 3)}\n")
        print(f"Source frequency: {round(source_freq.value, 2)} GHz\n")
        print(f"Band: {params.band}\n")
        print(f"Velocity resolution: {round(vel_res.value, 2)} Km/s\n")
        print(f"Angular resolution: {round(ang_res.value, 3)} arcsec\n")
        print(f"Infrared Luminosity: {lum_infrared:.2e}\n")
    else:
        log(f"Central Frequency: {central_freq}")
        log(f"Beam size: {round(beam_size.value, 4)} arcsec")
        log(f"Spectral Window: {band_range}")
        log(f"Freq Support: {delta_freq}")
        log(f"Cube Dimensions: {n_pix} x {n_pix} x {n_channels}")
        log(f"Redshift: {round(redshift, 3)}")
        log(f"Source frequency: {round(source_freq.value, 2)} GHz")
        log(f"Band: {params.band}")
        log(f"Velocity resolution: {round(vel_res.value, 2)} Km/s")
        log(f"Angular resolution: {round(ang_res.value, 3)} arcsec")
        log(f"Infrared Luminosity: {lum_infrared:.2e}")

    if params.source_type == "extended":
        snapshot = uas.redshift_to_snapshot(redshift)
        tng_subhaloid = uas.get_subhaloids_from_db(1, params.main_dir, snapshot)
    else:
        snapshot = None
        tng_subhaloid = None

    if isinstance(line_names, (list, np.ndarray)):
        for line_name, line_flux in zip(line_names, line_fluxes):
            log(
                f"Simulating Line {line_name} Flux: {line_flux:.3e} at z {redshift}"
            )
    elif line_names is not None:
        log(
            f"Simulating Line {line_names} Flux: {line_fluxes[0]} at z {redshift}"
        )

    if remote:
        print(f"Simulating Continum Flux: {np.mean(continum):.2e}")
        print(f"Continuum Sensitity: {cont_sens}")
        print("Generating skymodel cube ...\n")
    else:
        log(f"Simulating Continum Flux: {np.mean(continum):.2e}")
        log(f"Continuum Sensitity: {cont_sens}")
        log("Generating skymodel cube ...")

    datacube = usm.DataCube(
        n_px_x=n_pix,
        n_px_y=n_pix,
        n_channels=n_channels,
        px_size=cell_size,
        channel_width=delta_freq,
        spectral_centre=central_freq,
        ra=ra,
        dec=dec,
    )
    wcs = datacube.wcs
    fwhm_x = fwhm_y = angle = None
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
    shift_x = np.random.randint(
        0.1 * fov.value / cell_size.value, 1.5 * (fov.value / cell_size.value) - pos_x
    )
    shift_y = np.random.randint(
        0.1 * fov.value / cell_size.value, 1.5 * (fov.value / cell_size.value) - pos_y
    )
    pos_x = pos_x + (shift_x / 2) * random.choice([-1, 1])
    pos_y = pos_y + (shift_y / 2) * random.choice([-1, 1])
    pos_x = min(pos_x, n_pix - 1)
    pos_y = min(pos_y, n_pix - 1)
    pos_z = [int(index) for index in source_channel_index]
    log(f"Source Position (x, y): ({pos_x}, {pos_y})")

    if params.source_type == "point":
        status("Inserting Point Source Model")
        model = usm.PointlikeSkyModel(
            datacube=datacube,
            continuum=continum,
            line_fluxes=line_fluxes,
            pos_x=int(pos_x),
            pos_y=int(pos_y),
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_chan=n_channels,
            update_progress=progress_adapter,
        )
        datacube = model.insert()
    elif params.source_type == "gaussian":
        status("Inserting Gaussian Source Model")
        fwhm_x = np.random.randint(3, 10)
        fwhm_y = np.random.randint(3, 10)
        angle = np.random.randint(0, 180)
        model = usm.GaussianSkyModel(
            datacube=datacube,
            continuum=continum,
            line_fluxes=line_fluxes,
            pos_x=int(pos_x),
            pos_y=int(pos_y),
            pos_z=pos_z,
            fwhm_x=int(fwhm_x),
            fwhm_y=int(fwhm_y),
            fwhm_z=fwhm_z,
            angle=int(angle),
            n_px=int(n_pix),
            n_chan=n_channels,
            client=dask_client,
            update_progress=progress_adapter,
        )
        datacube = model.insert()
    elif params.source_type == "extended":
        status("Inserting Extended Source Model")
        model = usm.ExtendedSkyModel(
            datacube=datacube,
            tngpath=params.tng_dir,
            snapshot=snapshot,
            subhalo_id=int(tng_subhaloid),
            redshift=redshift,
            ra=ra,
            dec=dec,
            api_key=tng_api_key,
            client=dask_client,
            update_progress=progress_adapter,
            terminal=terminal_logger,
        )
        datacube = model.insert()
    elif params.source_type == "diffuse":
        status("Inserting Diffuse Source Model")
        model = usm.DiffuseSkyModel(
            datacube=datacube,
            continuum=continum,
            line_fluxes=line_fluxes,
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_px=int(n_pix),
            n_chan=n_channels,
            client=dask_client,
            update_progress=progress_adapter,
        )
        datacube = model.insert()
    elif params.source_type == "galaxy-zoo":
        status("Inserting Galaxy Zoo Source Model")
        galaxy_path = os.path.join(params.galaxy_zoo_dir, "images_gz2", "images")
        model = usm.GalaxyZooSkyModel(
            datacube=datacube,
            continuum=continum,
            line_fluxes=line_fluxes,
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_px=int(n_pix),
            n_chan=n_channels,
            data_path=galaxy_path,
            client=dask_client,
            update_progress=progress_adapter,
        )
        datacube = model.insert()
    elif params.source_type == "molecular":
        status("Inserting Molecular Cloud Source Model")
        model = usm.MolecularCloudSkyModel(
            datacube=datacube,
            continuum=continum,
            line_fluxes=line_fluxes,
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_px=int(n_pix),
            n_chan=n_channels,
            client=dask_client,
            update_progress=progress_adapter,
        )
        datacube = model.insert()
    elif params.source_type == "hubble-100":
        status("Insert Hubble Top 100 Source Model")
        hubble_path = os.path.join(params.hubble_dir, "top100")
        model = usm.HubbleSkyModel(
            datacube=datacube,
            continuum=continum,
            line_fluxes=line_fluxes,
            pos_z=pos_z,
            fwhm_z=fwhm_z,
            n_px=int(n_pix),
            n_chan=n_channels,
            data_path=hubble_path,
            client=dask_client,
            update_progress=progress_adapter,
        )
        datacube = model.insert()

    sim_params_path = os.path.join(
        params.output_dir, f"sim_params_{params.idx}.txt"
    )
    uas.write_sim_parameters(
        sim_params_path,
        params.source_name,
        params.member_ouid,
        ra,
        dec,
        ang_res,
        vel_res,
        int_time,
        params.band,
        band_range,
        central_freq,
        redshift,
        line_fluxes,
        line_names,
        line_frequency,
        continum,
        fov,
        beam_size,
        cell_size,
        n_pix,
        n_channels,
        snapshot,
        tng_subhaloid,
        lum_infrared,
        fwhm_z,
        params.source_type,
        fwhm_x,
        fwhm_y,
        angle,
    )

    if params.inject_serendipitous and not stop_requested:
        status("Inserting Serendipitous Sources")
        if params.source_type != "gaussian":
            fwhm_x = np.random.randint(3, 10)
            fwhm_y = np.random.randint(3, 10)
        datacube = usm.insert_serendipitous(
            terminal_logger,
            dask_client,
            progress_adapter,
            datacube,
            continum,
            cont_sens.value,
            line_fluxes,
            line_names,
            line_frequency,
            delta_freq.value,
            pos_z,
            fwhm_x,
            fwhm_y,
            fwhm_z,
            n_pix,
            n_channels,
            sim_params_path,
        )

    header = usm.get_datacube_header(datacube, params.obs_date)
    model = datacube._array.to_value(datacube._array.unit)
    if len(model.shape) == 4:
        model = model[0]
    dim1, dim2, dim3 = model.shape
    if dim1 == n_channels:
        model = model / beam_area.value
    else:
        model = model.T / beam_area.value
    totflux = np.sum(model)
    if remote:
        print(f"Total Flux injected in model cube: {round(totflux, 3)} Jy")
        print("Observing with ALMA")
    else:
        log(f"Total Flux injected in model cube: {round(totflux, 3)} Jy")
        log("Observing with ALMA")

    min_line_flux = np.min(line_fluxes)
    interferometer = uin.Interferometer(
        idx=params.idx,
        client=dask_client,
        skymodel=model,
        main_dir=params.main_dir,
        output_dir=params.output_dir,
        ra=ra,
        dec=dec,
        central_freq=central_freq,
        bandwidth=band_range,
        fov=fov.value,
        antenna_array=params.antenna_array,
        noise=0.1 * (min_line_flux / beam_area.value) / params.snr,
        snr=params.snr,
        integration_time=int_time.value * second2hour,
        observation_date=params.obs_date,
        header=header,
        save_mode=params.save_mode,
        robust=robust,
        logger=terminal_logger,
    )
    if interferometer_progress_callback is not None:
        interferometer.progress_signal.connect(interferometer_progress_callback)
    status("Observing with ALMA")
    simulation_results = interferometer.run_interferometric_sim()
    if remote:
        print("Finished")
    else:
        log("Finished")
    stop = time.time()
    log(
        "Execution took {} seconds".format(
            strftime("%H:%M:%S", gmtime(stop - start))
        )
    )
    return simulation_results
