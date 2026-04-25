"""Component tests for sky model generation."""

import astropy.units as U
import pytest

from almasim import skymodels
from almasim.services.astro.spectral import process_spectral_data


class InlineClient:
    """Minimal synchronous Dask-like client for component tests."""

    def compute(self, tasks):
        if isinstance(tasks, list):
            return [
                (task.compute(scheduler="synchronous") if hasattr(task, "compute") else task)
                for task in tasks
            ]
        return tasks.compute(scheduler="synchronous") if hasattr(tasks, "compute") else tasks

    def gather(self, futures):
        return futures if isinstance(futures, list) else [futures]


@pytest.fixture
def sample_datacube():
    """Create a sample datacube for testing."""
    return skymodels.DataCube(
        n_px_x=64,
        n_px_y=64,
        n_channels=32,
        px_size=0.1 * U.arcsec,
        channel_width=0.1 * U.GHz,
        spectral_centre=100.0 * U.GHz,
        ra=0.0 * U.deg,
        dec=0.0 * U.deg,
    )


@pytest.fixture
def sample_spectral_data(main_dir):
    """Create sample spectral data for testing."""
    redshift = 0.1
    central_frequency = 100.0  # GHz
    delta_freq = 10.0  # GHz
    source_frequency = 100.0  # GHz
    n_channels = 32
    lum_infrared = 1e10
    cont_sens = 0.1  # Jy

    result = process_spectral_data(
        source_type="point",
        master_path=main_dir,
        redshift=redshift,
        central_frequency=central_frequency,
        delta_freq=delta_freq,
        source_frequency=source_frequency,
        n_channels=n_channels,
        lum_infrared=lum_infrared,
        cont_sens=cont_sens,
        line_names=None,
        n_lines=1,
        remote=False,
    )

    return result


@pytest.mark.component
def test_pointlike_skymodel_generation(sample_datacube, sample_spectral_data):
    """Test generating a pointlike sky model."""
    (
        continum,
        line_fluxes,
        line_names,
        redshift,
        line_frequency,
        source_channel_index,
        n_channels_nw,
        bandwidth,
        freq_support,
        cont_frequencies,
        fwhm_z,
        lum_infrared,
    ) = sample_spectral_data

    pos_x, pos_y, _ = sample_datacube.wcs.sub(3).wcs_world2pix(
        0.0 * U.deg, 0.0 * U.deg, 100.0 * U.GHz, 0
    )
    pos_z = [int(idx) for idx in source_channel_index]

    model = skymodels.PointlikeSkyModel(
        datacube=sample_datacube,
        continuum=continum,
        line_fluxes=line_fluxes,
        pos_x=int(pos_x),
        pos_y=int(pos_y),
        pos_z=pos_z,
        fwhm_z=fwhm_z,
        n_chan=32,
    )

    result = model.insert()
    assert result is not None
    assert hasattr(result, "_array")
    array = result._array.to_value(result._array.unit)
    assert array.shape[0] == 32 or array.shape[2] == 32


@pytest.mark.component
def test_gaussian_skymodel_generation(sample_datacube, sample_spectral_data):
    """Test generating a Gaussian sky model."""
    (
        continum,
        line_fluxes,
        line_names,
        redshift,
        line_frequency,
        source_channel_index,
        n_channels_nw,
        bandwidth,
        freq_support,
        cont_frequencies,
        fwhm_z,
        lum_infrared,
    ) = sample_spectral_data

    pos_x, pos_y, _ = sample_datacube.wcs.sub(3).wcs_world2pix(
        0.0 * U.deg, 0.0 * U.deg, 100.0 * U.GHz, 0
    )
    pos_z = [int(idx) for idx in source_channel_index]

    model = skymodels.GaussianSkyModel(
        datacube=sample_datacube,
        continuum=continum,
        line_fluxes=line_fluxes,
        pos_x=int(pos_x),
        pos_y=int(pos_y),
        pos_z=pos_z,
        fwhm_x=5,
        fwhm_y=5,
        fwhm_z=fwhm_z,
        angle=45,
        n_px=64,
        n_chan=32,
        client=InlineClient(),
    )

    result = model.insert()
    assert result is not None
    assert hasattr(result, "_array")


@pytest.mark.component
def test_diffuse_skymodel_generation(sample_datacube, sample_spectral_data):
    """Test generating a diffuse sky model."""
    (
        continum,
        line_fluxes,
        line_names,
        redshift,
        line_frequency,
        source_channel_index,
        n_channels_nw,
        bandwidth,
        freq_support,
        cont_frequencies,
        fwhm_z,
        lum_infrared,
    ) = sample_spectral_data

    pos_z = [int(idx) for idx in source_channel_index]

    model = skymodels.DiffuseSkyModel(
        datacube=sample_datacube,
        continuum=continum,
        line_fluxes=line_fluxes,
        pos_z=pos_z,
        fwhm_z=fwhm_z,
        n_px=64,
        n_chan=32,
        client=InlineClient(),
    )

    result = model.insert()
    assert result is not None
    assert hasattr(result, "_array")


@pytest.mark.component
@pytest.mark.slow
def test_serendipitous_sources_insertion(sample_datacube, sample_spectral_data, tmp_path):
    """Test inserting serendipitous sources into a datacube."""
    (
        continum,
        line_fluxes,
        line_names,
        redshift,
        line_frequency,
        source_channel_index,
        n_channels_nw,
        bandwidth,
        freq_support,
        cont_frequencies,
        fwhm_z,
        lum_infrared,
    ) = sample_spectral_data

    pos_z = [int(idx) for idx in source_channel_index]

    # Ensure fwhm_z values are valid (> 2) for serendipitous insertion
    # The function requires fwhm_zs[0] > 2 for np.random.randint(2, int(fwhm_zs[0]), ...)
    fwhm_z_valid = [max(3.0, fz) for fz in fwhm_z]

    # First create a base model
    pos_x, pos_y, _ = sample_datacube.wcs.sub(3).wcs_world2pix(
        0.0 * U.deg, 0.0 * U.deg, 100.0 * U.GHz, 0
    )

    base_model = skymodels.PointlikeSkyModel(
        datacube=sample_datacube,
        continuum=continum,
        line_fluxes=line_fluxes,
        pos_x=int(pos_x),
        pos_y=int(pos_y),
        pos_z=pos_z,
        fwhm_z=fwhm_z_valid,
        n_chan=32,
    )
    datacube = base_model.insert()

    # Now add serendipitous sources (requires Dask client)
    sim_params_path = tmp_path / "sim_params.txt"
    sim_params_path.write_text("test")

    result = skymodels.insert_serendipitous(
        terminal=None,
        client=InlineClient(),
        update_progress=None,
        datacube=datacube,
        continum=continum,
        cont_sens=0.1,
        line_fluxes=line_fluxes,
        line_names=line_names,
        line_frequencies=line_frequency,  # Note: plural form
        freq_sup=0.1,  # Note: freq_sup not delta_freq
        pos_zs=pos_z,  # Note: plural form
        fwhm_x=5,
        fwhm_y=5,
        fwhm_zs=fwhm_z_valid,  # Note: plural form, ensure >= 2
        n_px=64,
        n_chan=32,  # Note: n_chan not n_channels
        sim_params_path=str(sim_params_path),
    )

    assert result is not None
    assert hasattr(result, "_array")


@pytest.mark.component
def test_datacube_creation():
    """Test creating a DataCube with various parameters."""
    datacube = skymodels.DataCube(
        n_px_x=128,
        n_px_y=128,
        n_channels=64,
        px_size=0.05 * U.arcsec,
        channel_width=0.05 * U.GHz,
        spectral_centre=150.0 * U.GHz,
        ra=10.0 * U.deg,
        dec=-20.0 * U.deg,
    )

    assert datacube.n_px_x == 128
    assert datacube.n_px_y == 128
    assert datacube.n_channels == 64
    assert datacube.px_size == 0.05 * U.arcsec
    assert hasattr(datacube, "wcs")
    assert hasattr(datacube, "_array")


@pytest.mark.component
def test_datacube_header_generation(sample_datacube):
    """Test generating FITS header from datacube."""
    obs_date = "2020-01-01"
    header = skymodels.get_datacube_header(sample_datacube, obs_date)

    assert header is not None
    assert "NAXIS" in header
    # Check for date-related fields (MJD-OBS is used instead of DATE-OBS)
    assert "MJD-OBS" in header or "DATE-OBS" in header or "DATE" in header
