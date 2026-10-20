"""Build the model cube: sky model, background and serendipitous sources (MADIV-28 T5.7)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast

import astropy.units as U
import numpy as np

from ... import skymodels as usm
from ...skymodels.registry import SKYMODELS
from .. import astro as uas
from ..external_skymodel import is_external_source_type, load_external_sky_model
from .background import generate_background_cube
from .context import RunContext
from .geometry import CubeGeometry, resolve_source_pixel_position
from .params import LogFn, SimulationParams, _clean
from .spectral_setup import SpectralSetup

if TYPE_CHECKING:
    from dask.distributed import Client

    from ..compute.base import ComputationBackend


@dataclass
class SkyCube:
    """The model cube of a run and what was put into it."""

    datacube: Any
    header: Any
    background_cube: np.ndarray
    external_input_metadata: Optional[dict[str, Any]]
    pos_x: float
    pos_y: float
    fwhm_x: Any
    fwhm_y: Any
    angle: Any
    snapshot: Optional[int]
    tng_subhaloid: Any


def _get_client_from_backend(
    compute_backend: Optional["ComputationBackend"],
) -> Optional["Client"]:
    """Extract Dask client from computation backend for backward compatibility.

    TODO: Update skymodels to use ComputationBackend directly.
    """
    if compute_backend is None:
        return None
    # For Dask backend, extract the client
    if hasattr(compute_backend, "client"):
        return compute_backend.client
    return None


def _ensure_morphology_dataset(params: "SimulationParams", log: LogFn = None) -> None:
    """Ensure image assets exist for image-based sky models (galaxy-zoo / hubble).

    Downloads the morphology images from Kaggle when missing, using
    ``params.kaggle_token`` (a path to a ``kaggle.json`` file or its directory) if
    provided, otherwise the Kaggle defaults.
    """
    st = str(params.source_type).lower()

    def _say(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    dataset = SKYMODELS[st].dataset if st in SKYMODELS else None
    if dataset == "galaxy-zoo":
        data_path = os.path.join(params.galaxy_zoo_dir, "images_gz2", "images")
        if os.path.isdir(data_path) and os.listdir(data_path):
            return
        from ...skymodels.datasets.galaxy_zoo import download_galaxy_zoo

        _say(f"Fetching Galaxy Zoo 2 images into {params.galaxy_zoo_dir}")
        download_galaxy_zoo(params.galaxy_zoo_dir, token=params.kaggle_token)
    elif dataset == "hubble-100":
        data_path = os.path.join(params.hubble_dir, "top100")
        if os.path.isdir(data_path) and os.listdir(data_path):
            return
        from ...skymodels.datasets.hubble import download_hubble_top100

        _say(f"Fetching Hubble Top-100 images into {data_path}")
        download_hubble_top100(data_path, token=params.kaggle_token)


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
            snapshot=cast(int, snapshot),  # set for extended sources
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


def build_sky_cube(
    params: SimulationParams,
    geometry: CubeGeometry,
    spectral: SpectralSetup,
    *,
    cont_sens: U.Quantity,
    cont_sens_jy: U.Quantity,
    ctx: RunContext,
    progress_adapter: Any = None,
    compute_backend: Any = None,
    terminal_logger: Any = None,
) -> SkyCube:
    """The datacube holding the source, the background and any serendipitous sources."""
    n_pix, n_channels = geometry.n_pix, geometry.n_channels
    central_freq, delta_freq, cell_size = (
        geometry.central_freq,
        geometry.delta_freq,
        geometry.cell_size,
    )
    ra = params.ra * U.deg
    dec = params.dec * U.deg
    redshift = spectral.redshift
    continum, line_fluxes, line_names = (
        spectral.continuum,
        spectral.line_fluxes,
        spectral.line_names,
    )
    is_external_model = is_external_source_type(params.source_type)
    log = ctx.log

    if params.source_type == "extended":
        snapshot = uas.redshift_to_snapshot(redshift)
        tng_subhaloid = uas.get_subhaloids_from_db(1, params.main_dir, snapshot)
    else:
        snapshot = None
        tng_subhaloid = None

    if isinstance(line_names, (list, np.ndarray)):
        for line_name, line_flux in zip(line_names, line_fluxes):
            log(f"Simulating Line {line_name} Flux: {line_flux:.3e} at z {redshift}")
    elif line_names is not None:
        log(f"Simulating Line {line_names} Flux: {line_fluxes[0]} at z {redshift}")

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
    pos_z = [int(index) for index in spectral.source_channel_index]
    log(
        "Source Position (x, y): "
        f"({pos_x:.2f}, {pos_y:.2f}) with offsets "
        f'({params.source_offset_x_arcsec:.2f}", {params.source_offset_y_arcsec:.2f}")'
    )

    # Get client from backend if available (for backward compatibility with skymodels)
    client_for_skymodel = _get_client_from_backend(compute_backend)
    external_input_metadata = None
    channel_frequencies_hz = geometry.channel_frequencies_hz()

    if is_external_model:
        ctx.status("Loading external sky model")
        external_payload = load_external_sky_model(
            source_type=params.source_type,
            skymodel_path=params.external_skymodel_path,
            component_table_path=params.external_component_table_path,
            target_npix=int(n_pix),
            target_nchannels=int(n_channels),
            alignment_mode=params.external_alignment_mode,
            header_mode=params.external_header_mode,
            header_overrides=params.external_header_overrides,
            target_header=usm.get_datacube_header(datacube, params.obs_date),
            target_wcs=wcs,
            channel_frequencies_hz=channel_frequencies_hz,
        )
        usm.add_channel_first(datacube, external_payload.cube)
        header = external_payload.header
        external_input_metadata = external_payload.metadata
    else:
        # Common parameters for all sky models
        common_params = {
            "datacube": datacube,
            "continuum": continum,
            "line_fluxes": line_fluxes,
            "pos_z": pos_z,
            "fwhm_z": spectral.fwhm_z,
            "n_chan": n_channels,
            "update_progress": progress_adapter,
        }

        # For image-based morphologies, make sure the image assets exist,
        # downloading them from Kaggle (optionally using params.kaggle_token).
        _ensure_morphology_dataset(params, log)

        # Create and insert sky model
        ctx.status("Creating sky model")
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
            tng_api_key=_clean(params.tng_api_key),
            terminal_logger=terminal_logger,
            client=client_for_skymodel,
        )
        datacube = model.insert()
        header = usm.get_datacube_header(datacube, params.obs_date)

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
        usm.add_channel_first(datacube, background_cube)
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

    if params.inject_serendipitous and not ctx.stopped():
        ctx.status("Inserting Serendipitous Sources")
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
            spectral.line_frequency,
            delta_freq.value,
            pos_z,
            fwhm_x or np.random.randint(3, 10),
            fwhm_y or np.random.randint(3, 10),
            spectral.fwhm_z,
            n_pix,
            n_channels,
            None,
        )

    if not is_external_model:
        header = usm.get_datacube_header(datacube, params.obs_date)
    return SkyCube(
        datacube=datacube,
        header=header,
        background_cube=background_cube,
        external_input_metadata=external_input_metadata,
        pos_x=pos_x,
        pos_y=pos_y,
        fwhm_x=fwhm_x,
        fwhm_y=fwhm_y,
        angle=angle,
        snapshot=snapshot,
        tng_subhaloid=tng_subhaloid,
    )
