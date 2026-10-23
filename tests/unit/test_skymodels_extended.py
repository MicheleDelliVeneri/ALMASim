"""Unit tests for almasim.skymodels.extended."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import astropy.units as U
import numpy as np
import pytest

from almasim.skymodels.extended import ExtendedSkyModel, insert_source, inserted_hi_mass


@pytest.fixture(autouse=True)
def _mass_from_mock(monkeypatch):
    """The mocked martini runs below carry their inserted mass as ``inserted_mass``."""
    monkeypatch.setattr("almasim.skymodels.extended.inserted_hi_mass", lambda M: M.inserted_mass)


# ===========================================================================
# insert_source (a real martini insertion, no TNG download)
# ===========================================================================


@pytest.mark.unit
def test_insert_source_puts_most_of_a_small_galaxy_in_the_cube(monkeypatch):
    """martini grids a toy disc into the cube, and the mass bookkeeping sees it."""
    from martini import demo_source

    monkeypatch.undo()  # the real inserted_hi_mass, not _mass_from_mock's
    np.random.seed(0)  # demo_source draws from the global generator
    source = demo_source(N=200)
    terminal = MagicMock()

    M = insert_source(terminal, source, 64, 64, 95 * U.kHz, 20 * U.arcsec)

    cube = M.datacube._array
    assert cube.shape == (64, 64, 64)
    assert cube.unit == U.Jy / U.arcsec**2
    assert np.isfinite(cube.value).all() and cube.value.max() > 0
    ratio = (inserted_hi_mass(M) / source.input_mass).to_value(U.one)
    assert 0.5 < ratio < 1.05
    assert "Mass in cube" in terminal.add_log.call_args.args[0]


# ===========================================================================
# ExtendedSkyModel initialization
# ===========================================================================


@pytest.mark.unit
def test_extended_skymodel_init_stores_attributes():
    """ExtendedSkyModel.__init__ stores all supplied attributes."""
    mock_datacube = MagicMock()
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 16

    model = ExtendedSkyModel(
        datacube=mock_datacube,
        tngpath="/tmp/tng",
        snapshot=99,
        subhalo_id=42,
        redshift=0.3,
        ra=10.0 * U.deg,
        dec=-5.0 * U.deg,
        api_key="test_key",
    )

    assert model.tngpath == "/tmp/tng"
    assert model.snapshot == 99
    assert model.subhalo_id == 42
    assert model.redshift == 0.3
    assert model.api_key == "test_key"


@pytest.mark.unit
def test_extended_skymodel_init_optional_defaults():
    """terminal defaults to None."""
    mock_datacube = MagicMock()
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 16

    model = ExtendedSkyModel(
        datacube=mock_datacube,
        tngpath="/tmp/tng",
        snapshot=99,
        subhalo_id=42,
        redshift=0.3,
        ra=10.0 * U.deg,
        dec=-5.0 * U.deg,
        api_key=None,
    )

    assert model.terminal is None


# ===========================================================================
# ExtendedSkyModel.insert (heavily mocked)
# ===========================================================================


def _build_extended_model_with_mocked_insert_tng(
    mass_ratio_sequence: list[float],
):
    """Build an ExtendedSkyModel and patch insert_tng to control mass_ratio."""
    mock_datacube = MagicMock()
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 16
    mock_datacube.channel_width = 0.1 * U.GHz
    mock_terminal = MagicMock()

    model = ExtendedSkyModel(
        datacube=mock_datacube,
        tngpath="/tmp/tng",
        snapshot=99,
        subhalo_id=42,
        redshift=0.3,
        ra=10.0 * U.deg,
        dec=-5.0 * U.deg,
        api_key=None,
        terminal=mock_terminal,
    )
    return model, mass_ratio_sequence


@pytest.mark.unit
def test_extended_skymodel_insert_calls_insert_tng():
    """insert() should call insert_tng at least once."""
    model, _ = _build_extended_model_with_mocked_insert_tng([60.0])

    mock_M = MagicMock()
    mock_M.inserted_mass = 60.0 * U.Msun
    mock_M.source.input_mass = 100.0 * U.Msun
    mock_M.datacube = model.datacube

    with patch("almasim.skymodels.extended.insert_tng", return_value=mock_M) as mock_tng:
        result = model.insert()

    mock_tng.assert_called_once()
    assert result is not None


@pytest.mark.unit
def test_extended_skymodel_insert_retries_when_mass_ratio_low():
    """insert() retries with increased distance when mass_ratio < 50."""
    model, _ = _build_extended_model_with_mocked_insert_tng([5.0, 60.0])

    call_count = {"n": 0}
    mass_ratios = [5.0 * U.Msun, 60.0 * U.Msun]  # 5% then 60%
    input_mass = 100.0 * U.Msun

    def fake_insert_tng(*args, **kwargs):
        m = MagicMock()
        m.inserted_mass = mass_ratios[min(call_count["n"], len(mass_ratios) - 1)]
        m.source.input_mass = input_mass
        m.datacube = model.datacube
        call_count["n"] += 1
        return m

    with patch("almasim.skymodels.extended.insert_tng", side_effect=fake_insert_tng):
        result = model.insert()

    assert call_count["n"] >= 2
    assert result is not None


@pytest.mark.unit
def test_extended_skymodel_insert_terminal_logging():
    """When terminal is provided, log messages are emitted."""
    model, _ = _build_extended_model_with_mocked_insert_tng([60.0])

    mock_M = MagicMock()
    mock_M.inserted_mass = 60.0 * U.Msun
    mock_M.source.input_mass = 100.0 * U.Msun
    mock_M.datacube = model.datacube

    with patch("almasim.skymodels.extended.insert_tng", return_value=mock_M):
        model.insert()

    # terminal.add_log should have been called
    assert model.terminal.add_log.called


@pytest.mark.unit
def test_extended_skymodel_insert_no_terminal():
    """insert() runs without errors when terminal is None."""
    mock_datacube = MagicMock()
    mock_datacube.n_px_x = 32
    mock_datacube.n_channels = 16
    mock_datacube.channel_width = 0.1 * U.GHz

    model = ExtendedSkyModel(
        datacube=mock_datacube,
        tngpath="/tmp/tng",
        snapshot=99,
        subhalo_id=42,
        redshift=0.3,
        ra=10.0 * U.deg,
        dec=-5.0 * U.deg,
        api_key=None,
        terminal=None,
    )

    mock_M = MagicMock()
    mock_M.inserted_mass = 60.0 * U.Msun
    mock_M.source.input_mass = 100.0 * U.Msun
    mock_M.datacube = mock_datacube

    with patch("almasim.skymodels.extended.insert_tng", return_value=mock_M):
        result = model.insert()

    assert result is not None


@pytest.mark.unit
def test_extended_skymodel_insert_distance_scaling_low_mass_ratio():
    """insert() uses large distance multiplier when mass_ratio < 10."""
    model, _ = _build_extended_model_with_mocked_insert_tng([3.0, 3.0, 60.0])

    call_count = {"n": 0}
    masses = [3.0 * U.Msun, 3.0 * U.Msun, 60.0 * U.Msun]

    def fake_insert_tng(*args, **kwargs):
        m = MagicMock()
        idx = min(call_count["n"], len(masses) - 1)
        m.inserted_mass = masses[idx]
        m.source.input_mass = 100.0 * U.Msun
        m.datacube = model.datacube
        call_count["n"] += 1
        return m

    with patch("almasim.skymodels.extended.insert_tng", side_effect=fake_insert_tng):
        result = model.insert()

    assert call_count["n"] >= 3
    assert result is not None


@pytest.mark.unit
def test_extended_insert_uses_the_cube_cell_and_the_source_distance():
    """The TNG cube takes the run's pixel size, and the source starts at its own distance."""
    model, _ = _build_extended_model_with_mocked_insert_tng([60.0])
    model.datacube.px_size = 0.25 * U.arcsec
    mock_M = MagicMock()
    mock_M.inserted_mass = 60.0 * U.Msun
    mock_M.source.input_mass = 100.0 * U.Msun

    with patch("almasim.skymodels.extended.insert_tng", return_value=mock_M) as mock_tng:
        model.insert()

    args = mock_tng.call_args.args
    assert args[-1] == 0.25 * U.arcsec
    distance = args[6]
    assert 1000 < distance < 1400, "comoving distance of z = 0.3, not a fixed 50 Mpc"


@pytest.mark.unit
def test_extended_insert_starts_at_least_one_megaparsec_away():
    model, _ = _build_extended_model_with_mocked_insert_tng([60.0])
    model.redshift = 0.0
    mock_M = MagicMock()
    mock_M.inserted_mass = 60.0 * U.Msun
    mock_M.source.input_mass = 100.0 * U.Msun

    with patch("almasim.skymodels.extended.insert_tng", return_value=mock_M) as mock_tng:
        model.insert()

    assert mock_tng.call_args.args[6] == 1.0
