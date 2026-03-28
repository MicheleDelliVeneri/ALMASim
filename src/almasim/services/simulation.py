"""Simulation helpers decoupled from the legacy PyQt UI."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import astropy.units as U
import h5py
from astropy.io import fits

from . import interferometry as uin
from . import astro as uas
from .imaging import build_image_products
from .observation_plan import build_single_pointing_observation_plan
from .interferometry import antenna as ual_antenna
from .interferometry.frequency import freq_supp_extractor
from .interferometry.utils import closest_power_of_2
from .utils import log_message, as_progress_emitter
from .astro.spectral import process_spectral_data
from .. import skymodels as usm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.compute.base import ComputationBackend
    from dask.distributed import Client

LogFn = Optional[Callable[[str], None]]
StopFn = Union[bool, Callable[[], bool]]


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
    snr: Optional[float]
    n_lines: Any
    line_names: Any
    save_mode: str
    persist: bool
    ml_dataset_path: Optional[str]
    inject_serendipitous: bool
    remote: bool = False
    observation_configs: Optional[Any] = None
    ground_temperature_k: float = 270.0
    correlator: Optional[str] = None
    elevation_deg: Optional[float] = None
    line_sens_10kms: Optional[float] = None
    source_offset_x_arcsec: float = 0.0
    source_offset_y_arcsec: float = 0.0
    background_mode: str = "none"
    background_level: float = 1.0
    background_seed: Optional[int] = None

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
        snr: Optional[float] = None,
        save_mode: str = "npz",
        persist: bool = True,
        ml_dataset_path: Optional[Path | str] = None,
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
        observation_configs: Optional[Any] = None,
        ground_temperature_k: float = 270.0,
        correlator: Optional[str] = None,
        elevation_deg: Optional[float] = None,
        source_offset_x_arcsec: float = 0.0,
        source_offset_y_arcsec: float = 0.0,
        background_mode: str = "none",
        background_level: float = 1.0,
        background_seed: Optional[int] = None,
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
        line_sens_10kms = _get(["Line_sens_10kms_mJybeam", "line_sens_10kms"], required=False)
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
            line_sens_10kms=(float(line_sens_10kms) if line_sens_10kms is not None else None),
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
            persist=bool(persist),
            ml_dataset_path=(
                _resolve_path(ml_dataset_path)
                if ml_dataset_path is not None
                else None
            ),
            inject_serendipitous=inject_serendipitous,
            remote=remote,
            observation_configs=observation_configs,
            ground_temperature_k=float(ground_temperature_k),
            correlator=correlator,
            elevation_deg=elevation_deg,
            source_offset_x_arcsec=float(source_offset_x_arcsec),
            source_offset_y_arcsec=float(source_offset_y_arcsec),
            background_mode=str(background_mode),
            background_level=float(background_level),
            background_seed=(
                int(background_seed) if background_seed is not None else None
            ),
        )


@dataclass
class CleanCubeStage:
    """Intermediate clean-cube stage output."""

    datacube: Any
    model_cube: np.ndarray
    header: Any
    output_dir_abs: str
    sim_output_dir: Optional[str]
    sim_params_payload: dict[str, Any]
    interferometer_kwargs: dict[str, Any]
    interferometer_runs: list[dict[str, Any]]
    total_power_runs: list[dict[str, Any]]
    metadata: dict[str, Any]
    observation_plan: dict[str, Any]
    channel_frequencies_hz: np.ndarray
    channel_width_hz: float
    cell_size_arcsec: float
    background_cube: Optional[np.ndarray]


def _json_safe(value: Any) -> Any:
    """Convert nested metadata values into JSON/HDF5-friendly objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, U.Quantity):
        return {"value": value.value, "unit": str(value.unit)}
    if isinstance(value, Path):
        return str(value)
    return value


def write_ml_dataset_shard(
    output_path: Path | str,
    *,
    clean_cube: np.ndarray,
    dirty_cube: np.ndarray,
    dirty_vis: np.ndarray,
    uv_mask_cube: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    """Persist a simulation sample as a single HDF5 shard for ML workflows."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("clean_cube", data=clean_cube, compression="gzip")
        h5f.create_dataset("dirty_cube", data=dirty_cube, compression="gzip")
        h5f.create_dataset("dirty_vis", data=dirty_vis, compression="gzip")
        h5f.create_dataset("uv_mask_cube", data=uv_mask_cube, compression="gzip")
        h5f.attrs["metadata_json"] = json.dumps(_json_safe(dict(metadata)))

    return str(output_path)


# Progress emitter functions moved to services.utils


def _get_client_from_backend(compute_backend: Optional["ComputationBackend"]) -> Optional["Client"]:
    """Extract Dask client from computation backend for backward compatibility.
    
    TODO: Update skymodels to use ComputationBackend directly.
    """
    if compute_backend is None:
        return None
    # For Dask backend, extract the client
    if hasattr(compute_backend, "client"):
        return compute_backend.client
    return None


def _count_antennas(antenna_array: str) -> int:
    """Count antennas encoded in an ALMA antenna-array string."""
    tokens = [token for token in str(antenna_array).split() if token.strip()]
    return max(len(tokens), 2)


def estimate_simulation_footprint(params: SimulationParams) -> dict[str, Any]:
    """Estimate cube dimensions and raw storage footprint before running a simulation."""
    source_freq = params.freq * U.GHz
    band_range, central_freq, t_channels, _delta_freq = freq_supp_extractor(
        params.freq_support, source_freq
    )
    observation_plan = build_single_pointing_observation_plan(params)
    max_baseline_km = max(
        ual_antenna.get_max_baseline_from_antenna_array(
            config.antenna_array,
            params.main_dir,
        )
        for config in observation_plan.configs
    )
    max_baseline = max_baseline_km * U.km
    fov = params.fov * 3600 * U.arcsec
    beam_size = ual_antenna.estimate_alma_beam_size(
        central_freq, max_baseline, return_value=False
    )
    cell_size = beam_size / 5

    if params.n_pix is None:
        n_pix = closest_power_of_2(int(1.5 * fov.value / cell_size.value))
    else:
        n_pix = closest_power_of_2(int(params.n_pix))

    n_channels = int(t_channels if params.n_channels is None else params.n_channels)
    voxels = int(n_channels) * int(n_pix) * int(n_pix)
    float32_gb = (voxels * np.dtype(np.float32).itemsize) / float(1024**3)
    complex64_gb = (voxels * np.dtype(np.complex64).itemsize) / float(1024**3)
    # Approximate raw footprint for the standard persisted image/cube products.
    estimated_standard_output_gb = 6.0 * float32_gb + complex64_gb

    return {
        "n_pix": int(n_pix),
        "n_channels": int(n_channels),
        "cube_shape": [int(n_channels), int(n_pix), int(n_pix)],
        "cube_voxels": voxels,
        "cell_size_arcsec": float(cell_size.to(U.arcsec).value),
        "beam_size_arcsec": float(beam_size.to(U.arcsec).value),
        "raw_single_cube_gb": float(float32_gb),
        "raw_complex_cube_gb": float(complex64_gb),
        "estimated_standard_output_gb": float(estimated_standard_output_gb),
        "note": "Raw uncompressed estimate. Actual NPZ/HDF5 output can be smaller due to compression.",
    }


def resolve_source_pixel_position(
    *,
    wcs: Any,
    ra: U.Quantity,
    dec: U.Quantity,
    central_freq: U.Quantity,
    n_pix: int,
    cell_size_arcsec: float,
    offset_x_arcsec: float = 0.0,
    offset_y_arcsec: float = 0.0,
) -> tuple[float, float]:
    """Resolve the source pixel position from phase center plus explicit offsets."""
    pos_x, pos_y, _ = wcs.sub(3).wcs_world2pix(ra, dec, central_freq, 0)
    pos_x += float(offset_x_arcsec) / max(float(cell_size_arcsec), 1e-12)
    pos_y += float(offset_y_arcsec) / max(float(cell_size_arcsec), 1e-12)
    pos_x = float(np.clip(pos_x, 0, n_pix - 1))
    pos_y = float(np.clip(pos_y, 0, n_pix - 1))
    return pos_x, pos_y


def _smoothed_positive_field(
    n_pix: int,
    rng: np.random.Generator,
    *,
    correlation_scale_pix: float,
) -> np.ndarray:
    """Generate a positive correlated 2D field using FFT low-pass smoothing."""
    white = rng.normal(size=(n_pix, n_pix))
    ky = np.fft.fftfreq(n_pix)
    kx = np.fft.fftfreq(n_pix)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    sigma_k = 1.0 / max(float(correlation_scale_pix), 1.0)
    low_pass = np.exp(-0.5 * k2 / max(sigma_k**2, 1e-12))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * low_pass)).astype(np.float32)
    field -= float(field.min())
    peak = float(field.max())
    if peak > 0.0:
        field /= peak
    return field


def generate_background_cube(
    *,
    mode: str,
    n_pix: int,
    n_channels: int,
    cell_size_arcsec: float,
    channel_frequencies_hz: np.ndarray,
    cont_sens_jy: float,
    level: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate an additive ALMA-band background cube in `(chan, y, x)` order."""
    mode = str(mode or "none").lower()
    if mode == "none" or level <= 0.0:
        return np.zeros((n_channels, n_pix, n_pix), dtype=np.float32)

    rng = np.random.default_rng(seed)
    nu0 = float(np.median(channel_frequencies_hz))
    spectral_ratio = np.asarray(channel_frequencies_hz, dtype=float) / max(nu0, 1e-12)
    cube = np.zeros((n_channels, n_pix, n_pix), dtype=np.float32)

    if mode in {"blank_field_dsfg", "combined"}:
        field_arcmin2 = ((n_pix * cell_size_arcsec) / 60.0) ** 2
        expected_sources = max(1.0, 30.0 * field_arcmin2 * max(level, 0.25))
        n_sources = int(min(250, rng.poisson(expected_sources)))
        for _ in range(n_sources):
            x = int(rng.integers(0, n_pix))
            y = int(rng.integers(0, n_pix))
            base_flux = cont_sens_jy * level * float(np.exp(rng.normal(-0.4, 0.9)))
            alpha = float(rng.uniform(2.5, 4.0))
            spectrum = base_flux * spectral_ratio**alpha
            cube[:, y, x] += spectrum.astype(np.float32)

    if mode in {"dusty_diffuse", "combined"}:
        base_field = _smoothed_positive_field(
            n_pix,
            rng,
            correlation_scale_pix=max(4.0, n_pix / 10.0),
        )
        diffuse_amp = cont_sens_jy * 0.5 * max(level, 0.1)
        beta = 3.0
        for channel in range(n_channels):
            cube[channel] += (
                diffuse_amp * spectral_ratio[channel] ** beta * base_field
            ).astype(np.float32)

    return cube


def _channel_first_to_datacube_layout(cube: np.ndarray, datacube_array: Any) -> np.ndarray:
    """Match a channel-first cube to the current datacube array layout."""
    datacube_shape = tuple(datacube_array.shape)
    if datacube_shape[0] == cube.shape[0]:
        return cube
    return np.transpose(cube, (2, 1, 0))


def _create_sky_model(
    source_type: str,
    common_params: dict,
    params: SimulationParams,
    pos_x: float,
    pos_y: float,
    n_pix: int,
    snapshot: Optional[int],
    tng_subhaloid: Optional[int],
    redshift: float,
    ra: U.Quantity,
    dec: U.Quantity,
    tng_api_key: Optional[str],
    terminal_logger: Optional[Any],
    client: Optional["Client"],
) -> Any:
    """Create and return the appropriate sky model based on source type."""
    source_type_handlers = {
        "point": lambda: usm.PointlikeSkyModel(
            **common_params,
            pos_x=int(pos_x),
            pos_y=int(pos_y),
        ),
        "gaussian": lambda: usm.GaussianSkyModel(
            **common_params,
            pos_x=int(pos_x),
            pos_y=int(pos_y),
            fwhm_x=int(np.random.randint(3, 10)),
            fwhm_y=int(np.random.randint(3, 10)),
            angle=int(np.random.randint(0, 180)),
            n_px=int(n_pix),
            client=client,
        ),
        "extended": lambda: usm.ExtendedSkyModel(
            datacube=common_params["datacube"],
            tngpath=params.tng_dir,
            snapshot=snapshot,
            subhalo_id=int(tng_subhaloid) if tng_subhaloid else 0,
            redshift=redshift,
            ra=ra,
            dec=dec,
            api_key=tng_api_key,
            client=client,
            update_progress=common_params["update_progress"],
            terminal=terminal_logger,
        ),
        "diffuse": lambda: usm.DiffuseSkyModel(
            **common_params,
            n_px=int(n_pix),
            client=client,
        ),
        "galaxy-zoo": lambda: usm.GalaxyZooSkyModel(
            **common_params,
            n_px=int(n_pix),
            data_path=os.path.join(params.galaxy_zoo_dir, "images_gz2", "images"),
            client=client,
        ),
        "molecular": lambda: usm.MolecularCloudSkyModel(
            **common_params,
            n_px=int(n_pix),
            client=client,
        ),
        "hubble-100": lambda: usm.HubbleSkyModel(
            **common_params,
            n_px=int(n_pix),
            data_path=os.path.join(params.hubble_dir, "top100"),
            client=client,
        ),
    }

    handler = source_type_handlers.get(source_type)
    if handler is None:
        raise ValueError(f"Unknown source type: {source_type}")
    
    return handler()


def generate_clean_cube(
    params: SimulationParams,
    *,
    logger: LogFn = None,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_emitter=None,
    compute_backend=None,
    terminal_logger=None,
    stop_requested: StopFn = False,
) -> CleanCubeStage:
    """Generate the clean sky cube and derived observation metadata."""
    remote = bool(params.remote)

    def log(message: str):
        log_message(logger, message, remote=remote)

    def status(message: str):
        if status_callback is not None:
            status_callback(message)

    def is_stop_requested() -> bool:
        if callable(stop_requested):
            return bool(stop_requested())
        return bool(stop_requested)

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
    if is_stop_requested():
        raise RuntimeError("Simulation cancelled")
    observation_plan = build_single_pointing_observation_plan(params)
    output_dir_abs = os.path.abspath(params.output_dir)
    persist_outputs = bool(params.persist and params.save_mode != "memory")
    sim_output_dir = (
        os.path.join(output_dir_abs, f"{params.project_name}_{params.idx}")
        if persist_outputs
        else None
    )
    if sim_output_dir is not None:
        os.makedirs(sim_output_dir, exist_ok=True)

    if remote:
        print(f"RA: {ra}")
        print(f"DEC: {dec}")
        print(f"Integration Time: {int_time}")
    else:
        log(f"RA: {ra}")
        log(f"DEC: {dec}")
        log(f"Integration Time: {int_time}")

    status("Computing Max baseline")
    max_baseline_km = max(
        ual_antenna.get_max_baseline_from_antenna_array(
            config.antenna_array,
            params.main_dir,
        )
        for config in observation_plan.configs
    )
    max_baseline = max_baseline_km * U.km
    if progress_adapter is not None:
        progress_adapter.emit(100)
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
        if rest_frequency is None:
            # For point sources, default to redshift=0 (local universe)
            # For other source types, this should be provided
            if params.source_type == "point":
                redshift = 0.0
                log("No redshift or rest_frequency provided, using default redshift=0 for point source")
            else:
                raise ValueError(
                    "Either 'redshift' or 'rest_frequency' must be provided. "
                    f"Cannot compute redshift without rest frequency for source type '{params.source_type}'."
                )
        
        if rest_frequency is not None:
            if isinstance(rest_frequency, (np.ndarray, list)):
                rest_frequency = np.sort(np.array(rest_frequency))[0]
            rest_frequency = rest_frequency * U.GHz
            redshift = uas.compute_redshift(rest_frequency, source_freq)
    
    # Compute rest_frequency from redshift if not already set
    # Check if rest_frequency is None or hasn't been converted to a Quantity yet
    if not isinstance(rest_frequency, U.Quantity):
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
    pos_x, pos_y = resolve_source_pixel_position(
        wcs=wcs,
        ra=ra,
        dec=dec,
        central_freq=central_freq,
        n_pix=n_pix,
        cell_size_arcsec=float(cell_size.to(U.arcsec).value),
        offset_x_arcsec=params.source_offset_x_arcsec,
        offset_y_arcsec=params.source_offset_y_arcsec,
    )
    pos_z = [int(index) for index in source_channel_index]
    log(
        "Source Position (x, y): "
        f"({pos_x:.2f}, {pos_y:.2f}) with offsets "
        f"({params.source_offset_x_arcsec:.2f}\", {params.source_offset_y_arcsec:.2f}\")"
    )

    # Common parameters for all sky models
    common_params = {
        "datacube": datacube,
        "continuum": continum,
        "line_fluxes": line_fluxes,
        "pos_z": pos_z,
        "fwhm_z": fwhm_z,
        "n_chan": n_channels,
        "update_progress": progress_adapter,
    }

    # Get client from backend if available (for backward compatibility with skymodels)
    client_for_skymodel = _get_client_from_backend(compute_backend)

    # Create and insert sky model
    status("Creating sky model")
    model = _create_sky_model(
        source_type=params.source_type,
        common_params=common_params,
        params=params,
        pos_x=pos_x,
        pos_y=pos_y,
        n_pix=n_pix,
        snapshot=snapshot,
        tng_subhaloid=tng_subhaloid,
        redshift=redshift,
        ra=ra,
        dec=dec,
        tng_api_key=tng_api_key,
        terminal_logger=terminal_logger,
        client=client_for_skymodel,
    )
    datacube = model.insert()

    channel_width_hz = delta_freq.to(U.Hz).value
    channel_offsets = (
        np.arange(n_channels, dtype=float) - (float(n_channels) - 1.0) / 2.0
    ) * channel_width_hz
    channel_frequencies_hz = central_freq.to(U.Hz).value + channel_offsets

    background_cube = generate_background_cube(
        mode=params.background_mode,
        n_pix=n_pix,
        n_channels=n_channels,
        cell_size_arcsec=float(cell_size.to(U.arcsec).value),
        channel_frequencies_hz=channel_frequencies_hz,
        cont_sens_jy=float(cont_sens_jy.value),
        level=params.background_level,
        seed=params.background_seed,
    )
    if np.any(background_cube):
        datacube._array += (
            _channel_first_to_datacube_layout(background_cube, datacube._array)
            * U.Jy
            * U.pix**-2
        )
        log(
            f"Injected background sky: mode={params.background_mode}, "
            f"level={params.background_level:.2f}, "
            f"total_flux={float(np.sum(background_cube)):.3e} Jy"
        )
    
    # Extract fwhm_x, fwhm_y, angle for parameter writing (if gaussian)
    if params.source_type == "gaussian":
        fwhm_x = getattr(model, "fwhm_x", None)
        fwhm_y = getattr(model, "fwhm_y", None)
        angle = getattr(model, "angle", None)
    else:
        fwhm_x = fwhm_y = angle = None

    if params.inject_serendipitous and not is_stop_requested():
        status("Inserting Serendipitous Sources")
        if params.source_type != "gaussian" or fwhm_x is None:
            fwhm_x = np.random.randint(3, 10)
            fwhm_y = np.random.randint(3, 10)
        datacube = usm.insert_serendipitous(
            terminal_logger,
            client_for_skymodel,
            progress_adapter,
            datacube,
            continum,
            cont_sens.value,
            line_fluxes,
            line_names,
            line_frequency,
            delta_freq.value,
            pos_z,
            fwhm_x or np.random.randint(3, 10),
            fwhm_y or np.random.randint(3, 10),
            fwhm_z,
            n_pix,
            n_channels,
            None,
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

    line_flux_magnitudes = np.abs(np.asarray(line_fluxes, dtype=float))
    nonzero_line_fluxes = line_flux_magnitudes[line_flux_magnitudes > 0]
    min_line_flux = (
        float(np.min(nonzero_line_fluxes))
        if nonzero_line_fluxes.size > 0
        else 0.0
    )
    sim_params_payload = {
        "sim_params_path": os.path.join(output_dir_abs, f"sim_params_{params.idx}.txt"),
        "source_name": params.source_name,
        "member_ouid": params.member_ouid,
        "ra": ra,
        "dec": dec,
        "ang_res": ang_res,
        "vel_res": vel_res,
        "int_time": int_time,
        "band": params.band,
        "band_range": band_range,
        "central_freq": central_freq,
        "redshift": redshift,
        "line_fluxes": line_fluxes,
        "line_names": line_names,
        "line_frequency": line_frequency,
        "continuum": continum,
        "fov": fov,
        "beam_size": beam_size,
        "cell_size": cell_size,
        "n_pix": n_pix,
        "n_channels": n_channels,
        "snapshot": snapshot,
        "tng_subhaloid": tng_subhaloid,
        "lum_infrared": lum_infrared,
        "fwhm_z": fwhm_z,
        "source_type": params.source_type,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "angle": angle,
    }
    metadata = {
        "idx": params.idx,
        "project_name": params.project_name,
        "source_name": params.source_name,
        "member_ouid": params.member_ouid,
        "source_type": params.source_type,
        "save_mode": params.save_mode,
        "persist": persist_outputs,
        "ra_deg": params.ra,
        "dec_deg": params.dec,
        "band": params.band,
        "ang_res_arcsec": params.ang_res,
        "vel_res_kms": params.vel_res,
        "fov_arcsec": fov.value,
        "obs_date": params.obs_date,
        "pwv": params.pwv,
        "int_time_s": params.int_time,
        "bandwidth_ghz": params.bandwidth,
        "source_freq_ghz": params.freq,
        "freq_support": params.freq_support,
        "cont_sens": params.cont_sens,
        "antenna_array": params.antenna_array,
        "n_pix": n_pix,
        "n_channels": n_channels,
        "central_freq_ghz": central_freq.to(U.GHz).value,
        "band_range_ghz": band_range.to(U.GHz).value,
        "channel_width_hz": delta_freq.to(U.Hz).value,
        "beam_size_arcsec": beam_size.to(U.arcsec).value,
        "cell_size_arcsec": cell_size.to(U.arcsec).value,
        "redshift": redshift,
        "rest_frequency_ghz": rest_frequency.to(U.GHz).value,
        "line_names": _json_safe(line_names),
        "line_fluxes": _json_safe(line_fluxes),
        "line_frequency_hz": _json_safe(line_frequency),
        "continuum": _json_safe(continum),
        "observation_plan": observation_plan.as_dict(),
        "source_offset_x_arcsec": params.source_offset_x_arcsec,
        "source_offset_y_arcsec": params.source_offset_y_arcsec,
        "source_pixel_x": float(pos_x),
        "source_pixel_y": float(pos_y),
        "background_mode": params.background_mode,
        "background_level": params.background_level,
        "background_seed": params.background_seed,
        "background_total_flux_jy": float(np.sum(background_cube)),
    }
    effective_snr = params.snr
    if effective_snr is None:
        if (
            params.line_sens_10kms is not None
            and params.cont_sens is not None
            and params.cont_sens > 0.0
        ):
            effective_snr = float(
                np.clip(params.line_sens_10kms / params.cont_sens, 1.0, 100.0)
            )
            log(
                "Auto-derived SNR from metadata sensitivities "
                f"(line/continuum): {effective_snr:.3f}"
            )
        elif params.cont_sens is not None and params.cont_sens > 0.0:
            effective_snr = 5.0
            log(
                "Auto-derived SNR fallback from metadata continuum sensitivity: 5.000"
            )
        else:
            effective_snr = 1.3
            log("Falling back to default SNR=1.3 because auto-derivation was not possible")
    else:
        effective_snr = float(effective_snr)

    metadata["effective_snr"] = effective_snr
    metadata["snr_mode"] = "auto" if params.snr is None else "manual"
    base_noise_reference = 0.1 * (min_line_flux / beam_area.value) / effective_snr
    interferometer_runs = []
    total_power_runs = []
    for config in observation_plan.configs:
        raw_noise_profile = uin.compute_channel_noise(
            uin.NoiseModelConfig(
                pwv_mm=params.pwv,
                ground_temperature_k=params.ground_temperature_k,
            ),
            channel_frequencies_hz,
            channel_width_hz,
            config.total_time_s,
            observation_plan.elevation_deg,
            antenna_diameter_m=config.antenna_diameter_m,
            n_antennas=_count_antennas(config.antenna_array),
        )
        noise_profile = uin.calibrate_noise_profile(
            raw_noise_profile,
            reference_noise=base_noise_reference,
        )
        if config.array_type == "TP":
            total_power_runs.append(
                {
                    "config_name": config.name,
                    "array_type": config.array_type,
                    "antenna_array": config.antenna_array,
                    "antenna_diameter_m": config.antenna_diameter_m,
                    "integration_time_s": config.total_time_s,
                    "noise": noise_profile,
                }
            )
        else:
            interferometer_runs.append(
                {
                    "idx": params.idx,
                    "skymodel": model,
                    "main_dir": params.main_dir,
                    "output_dir": output_dir_abs,
                    "ra": ra,
                    "dec": dec,
                    "central_freq": central_freq,
                    "bandwidth": band_range,
                    "fov": fov.value,
                    "antenna_array": config.antenna_array,
                    "noise": noise_profile,
                    "snr": effective_snr,
                    "integration_time": config.total_time_s / 3600.0,
                    "observation_date": params.obs_date,
                    "header": header,
                    "save_mode": params.save_mode,
                    "persist": persist_outputs,
                    "antenna_diameter_m": config.antenna_diameter_m,
                    "config_name": config.name,
                    "array_type": config.array_type,
                }
            )
    interferometer_kwargs = interferometer_runs[0] if interferometer_runs else {}

    return CleanCubeStage(
        datacube=datacube,
        model_cube=model,
        header=header,
        output_dir_abs=output_dir_abs,
        sim_output_dir=sim_output_dir,
        sim_params_payload=sim_params_payload,
        interferometer_kwargs=interferometer_kwargs,
        interferometer_runs=interferometer_runs,
        total_power_runs=total_power_runs,
        metadata=metadata,
        observation_plan=observation_plan.as_dict(),
        channel_frequencies_hz=channel_frequencies_hz,
        channel_width_hz=channel_width_hz,
        cell_size_arcsec=cell_size.to(U.arcsec).value,
        background_cube=background_cube.astype(np.float32),
    )


def simulate_observation(
    clean_cube_stage: CleanCubeStage,
    *,
    compute_backend=None,
    robust: float = 0.0,
    terminal_logger=None,
    interferometer_progress_callback: Optional[Callable[[int], None]] = None,
) -> dict[str, Any]:
    """Run the interferometric observation stage from a prepared clean cube."""
    per_config_results = []
    total_configs = max(
        len(clean_cube_stage.interferometer_runs) + len(clean_cube_stage.total_power_runs),
        1,
    )
    progress_index = 0

    for config_index, interferometer_kwargs in enumerate(clean_cube_stage.interferometer_runs):
        interferometer = uin.Interferometer(
            backend=compute_backend,
            robust=robust,
            logger=terminal_logger,
            **interferometer_kwargs,
        )
        if interferometer_progress_callback is not None:
            def scaled_progress(progress: int, *, idx=progress_index):
                base = idx / total_configs
                scaled = int(round((base + (progress / 100.0) / total_configs) * 100))
                interferometer_progress_callback(scaled)
            interferometer.progress_signal.connect(scaled_progress)
        config_result = interferometer.run_interferometric_sim()
        config_result["config_name"] = interferometer_kwargs.get("config_name")
        config_result["array_type"] = interferometer_kwargs.get("array_type")
        per_config_results.append(config_result)
        progress_index += 1

    int_results = None
    if per_config_results:
        int_results = uin.combine_interferometric_results(
            per_config_results,
            config_weights=[
                max(float(run["integration_time"]), 0.0) for run in clean_cube_stage.interferometer_runs
            ],
        )

    tp_per_config_results = []
    for tp_run in clean_cube_stage.total_power_runs:
        tp_result = uin.simulate_total_power_observation(
            clean_cube_stage.model_cube,
            freqs_hz=clean_cube_stage.channel_frequencies_hz,
            cell_size_arcsec=clean_cube_stage.cell_size_arcsec,
            noise_profile=tp_run["noise"],
            antenna_diameter_m=tp_run["antenna_diameter_m"],
            config_name=tp_run["config_name"],
            array_type=tp_run["array_type"],
        )
        tp_per_config_results.append(tp_result)
        if interferometer_progress_callback is not None:
            scaled = int(round(((progress_index + 1) / total_configs) * 100))
            interferometer_progress_callback(scaled)
        progress_index += 1

    tp_results = None
    if tp_per_config_results:
        tp_results = uin.combine_total_power_results(
            tp_per_config_results,
            config_weights=[
                max(float(run["integration_time_s"]), 0.0)
                for run in clean_cube_stage.total_power_runs
            ],
        )

    measurement_results: dict[str, Any] = {
        "observation_plan": clean_cube_stage.observation_plan,
        "int_results": int_results,
        "tp_results": tp_results,
        "model_cube": clean_cube_stage.model_cube,
    }
    if int_results is not None:
        measurement_results.update(int_results)
    if tp_results is not None:
        measurement_results.update(
            {
                key: value
                for key, value in tp_results.items()
                if key.startswith("tp_")
            }
        )
    return measurement_results


def image_products(
    simulation_results: dict[str, Any],
    *,
    reconstruction_epsilon: float = 1e-3,
) -> dict[str, Any]:
    """Construct image-domain products from measurement-stage outputs."""
    simulation_results = dict(simulation_results)
    products = build_image_products(
        int_results=simulation_results.get("int_results"),
        tp_results=simulation_results.get("tp_results"),
        reconstruction_epsilon=reconstruction_epsilon,
    )
    simulation_results.update(products)
    return simulation_results


def _save_optional_cube(
    output_dir: str,
    stem: str,
    idx: int,
    cube: np.ndarray | None,
    save_mode: str,
    header: Any,
) -> None:
    """Persist an optional cube using the repository's existing save formats."""
    if cube is None:
        return
    cube = np.asarray(cube)
    if save_mode == "npz":
        np.savez_compressed(os.path.join(output_dir, f"{stem}_{idx}.npz"), cube)
    elif save_mode == "h5":
        with h5py.File(os.path.join(output_dir, f"{stem}_{idx}.h5"), "w") as handle:
            handle.create_dataset(stem.replace("-", "_"), data=cube)
    elif save_mode == "fits":
        fits.PrimaryHDU(header=header.copy(), data=cube).writeto(
            os.path.join(output_dir, f"{stem}_{idx}.fits"),
            overwrite=True,
        )


def export_results(
    params: SimulationParams,
    clean_cube_stage: CleanCubeStage,
    simulation_results: dict[str, Any],
    *,
    logger: LogFn = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Persist optional file exports for simulation results."""
    remote = bool(params.remote)

    def log(message: str):
        log_message(logger, message, remote=remote)

    def status(message: str):
        if status_callback is not None:
            status_callback(message)

    exported_results = dict(simulation_results)
    exported_results["background_cube"] = clean_cube_stage.background_cube
    persist_outputs = bool(params.persist and params.save_mode != "memory")

    status("Exporting results")
    if persist_outputs:
        if clean_cube_stage.sim_output_dir is not None:
            antenna_config_paths = []
            for index, observation_config in enumerate(clean_cube_stage.observation_plan["configs"]):
                if len(clean_cube_stage.observation_plan["configs"]) == 1:
                    output_dir = clean_cube_stage.sim_output_dir
                    output_name = "antenna.cfg"
                else:
                    output_dir = os.path.join(
                        clean_cube_stage.sim_output_dir,
                        f"config_{index}",
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    output_name = "antenna.cfg"
                ual_antenna.generate_antenna_config_file_from_antenna_array(
                    observation_config["antenna_array"],
                    params.main_dir,
                    output_dir,
                )
                antenna_config_paths.append(os.path.join(output_dir, output_name))
            exported_results["antenna_config_path"] = antenna_config_paths[0]
            exported_results["antenna_config_paths"] = antenna_config_paths
            _save_optional_cube(
                clean_cube_stage.sim_output_dir,
                "int-image-cube",
                params.idx,
                exported_results.get("int_image_cube"),
                params.save_mode,
                clean_cube_stage.header,
            )
            _save_optional_cube(
                clean_cube_stage.sim_output_dir,
                "tp-image-cube",
                params.idx,
                exported_results.get("tp_image_cube"),
                params.save_mode,
                clean_cube_stage.header,
            )
            _save_optional_cube(
                clean_cube_stage.sim_output_dir,
                "tp-int-image-cube",
                params.idx,
                exported_results.get("tp_int_image_cube"),
                params.save_mode,
                clean_cube_stage.header,
            )
            _save_optional_cube(
                clean_cube_stage.sim_output_dir,
                "tp-dirty-cube",
                params.idx,
                exported_results.get("tp_dirty_cube"),
                params.save_mode,
                clean_cube_stage.header,
            )
            _save_optional_cube(
                clean_cube_stage.sim_output_dir,
                "background-cube",
                params.idx,
                clean_cube_stage.background_cube,
                params.save_mode,
                clean_cube_stage.header,
            )

        payload = clean_cube_stage.sim_params_payload
        uas.write_sim_parameters(
            payload["sim_params_path"],
            payload["source_name"],
            payload["member_ouid"],
            payload["ra"],
            payload["dec"],
            payload["ang_res"],
            payload["vel_res"],
            payload["int_time"],
            payload["band"],
            payload["band_range"],
            payload["central_freq"],
            payload["redshift"],
            payload["line_fluxes"],
            payload["line_names"],
            payload["line_frequency"],
            payload["continuum"],
            payload["fov"],
            payload["beam_size"],
            payload["cell_size"],
            payload["n_pix"],
            payload["n_channels"],
            payload["snapshot"],
            payload["tng_subhaloid"],
            payload["lum_infrared"],
            payload["fwhm_z"],
            payload["source_type"],
            payload["fwhm_x"],
            payload["fwhm_y"],
            payload["angle"],
        )
        exported_results["sim_params_path"] = payload["sim_params_path"]
    else:
        log("Skipping on-disk simulation exports (pure Python mode)")

    if params.ml_dataset_path:
        ml_dataset_path = write_ml_dataset_shard(
            params.ml_dataset_path,
            clean_cube=exported_results["model_cube"],
            dirty_cube=exported_results["dirty_cube"],
            dirty_vis=exported_results["dirty_vis"],
            uv_mask_cube=exported_results["uv_mask_cube"],
            metadata=clean_cube_stage.metadata,
        )
        exported_results["ml_dataset_path"] = ml_dataset_path

    return exported_results


def run_simulation(
    params: SimulationParams,
    *,
    logger: LogFn = None,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_emitter=None,
    compute_backend=None,
    terminal_logger=None,
    robust: float = 0.0,
    interferometer_progress_callback: Optional[Callable[[int], None]] = None,
    stop_requested: StopFn = False,
):
    """Execute the full ALMA simulation workflow."""
    remote = bool(params.remote)

    def log(message: str):
        log_message(logger, message, remote=remote)

    def is_stop_requested() -> bool:
        if callable(stop_requested):
            return bool(stop_requested())
        return bool(stop_requested)

    start = time.time()
    if is_stop_requested():
        raise RuntimeError("Simulation cancelled")
    if status_callback is not None:
        status_callback("Generating clean cube")
    clean_cube_stage = generate_clean_cube(
        params,
        logger=logger,
        status_callback=status_callback,
        progress_emitter=progress_emitter,
        compute_backend=compute_backend,
        terminal_logger=terminal_logger,
        stop_requested=is_stop_requested,
    )
    if is_stop_requested():
        raise RuntimeError("Simulation cancelled")
    if status_callback is not None:
        status_callback("Running interferometric simulation")
    simulation_results = simulate_observation(
        clean_cube_stage,
        compute_backend=compute_backend,
        robust=robust,
        terminal_logger=terminal_logger,
        interferometer_progress_callback=interferometer_progress_callback,
    )
    if is_stop_requested():
        raise RuntimeError("Simulation cancelled")
    if status_callback is not None:
        status_callback("Reconstructing image products")
    simulation_results = image_products(simulation_results)
    if is_stop_requested():
        raise RuntimeError("Simulation cancelled")
    exported_results = export_results(
        params,
        clean_cube_stage,
        simulation_results,
        logger=logger,
        status_callback=status_callback,
    )
    if is_stop_requested():
        raise RuntimeError("Simulation cancelled")

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
    return exported_results
