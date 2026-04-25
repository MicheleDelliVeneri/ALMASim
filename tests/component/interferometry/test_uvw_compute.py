from pathlib import Path

import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

from almasim.services.interferometry.baselines import generate_via_astropy, pairwise_baselines
from almasim.services.products.ms_io import export_native_ms


ANTENNA_CONFIG = (
    Path(__file__).parents[3] / "src" / "almasim" / "antenna_config" / "antenna_coordinates.csv"
)
OBS_TIME = Time("2024-01-01T00:00:00", scale="utc")
RA_RAD = 3.26 * u.rad
DEC_RAD = -1.05 * u.rad


def read_coordinates(coordinates_file: Path, n_antennas: int = 6) -> np.ndarray:
    return np.loadtxt(
        coordinates_file,
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3),
        max_rows=n_antennas,
    )


def _casa_to_cartesian_m(casa_direction: dict) -> np.ndarray:
    sph = coord.SphericalRepresentation(
        lon=np.asarray(casa_direction["m0"]["value"]) * u.Unit(casa_direction["m0"]["unit"]),
        lat=np.asarray(casa_direction["m1"]["value"]) * u.Unit(casa_direction["m1"]["unit"]),
        distance=np.asarray(casa_direction["m2"]["value"]) * u.Unit(casa_direction["m2"]["unit"]),
    )
    return sph.represent_as(coord.CartesianRepresentation).xyz.to_value(u.m).T


def generate_via_casa(antenna_positions_m: np.ndarray, ra: u.Quantity, dec: u.Quantity, time: Time) -> np.ndarray:
    try:
        import casatools
    except ImportError:
        pytest.skip("casatools not available")

    me = casatools.measures()
    qa = casatools.quanta()
    qq = qa.quantity

    me.doframe(me.observatory("ALMA"))
    me.doframe(me.epoch("UTC", qq(float(time.mjd), "d")))
    me.doframe(
        me.direction(
            "J2000",
            qq(float(ra.to_value(u.rad)), "rad"),
            qq(float(dec.to_value(u.rad)), "rad"),
        )
    )

    antpos_casa = me.position(
        "ITRF",
        qq(antenna_positions_m[:, 0], "m"),
        qq(antenna_positions_m[:, 1], "m"),
        qq(antenna_positions_m[:, 2], "m"),
    )
    antpos_baseline = me.asbaseline(antpos_casa)
    antpos_uvw_casa = me.touvw(antpos_baseline)[0]
    ant_xyz = _casa_to_cartesian_m(antpos_uvw_casa)
    # CASA's Cartesian axis order for this conversion is rotated relative to
    # Astropy's UVW convention, so rotate columns to compare like-for-like.
    baselines = pairwise_baselines(ant_xyz - ant_xyz[0])
    return np.roll(baselines, shift=1, axis=1)


@pytest.mark.component
def test_uvw_astropy_matches_casa_reference():
    coordinates_m = read_coordinates(ANTENNA_CONFIG)
    baselines_astropy = generate_via_astropy(coordinates_m, RA_RAD, DEC_RAD, OBS_TIME)
    baselines_casa = generate_via_casa(coordinates_m, RA_RAD, DEC_RAD, OBS_TIME)

    assert baselines_astropy.shape == baselines_casa.shape
    np.testing.assert_allclose(baselines_astropy, baselines_casa, rtol=1e-5, atol=5e-2)


@pytest.mark.component
def test_uvw_roundtrip_read_with_python_casacore(tmp_path: Path):

    coordinates_m = read_coordinates(ANTENNA_CONFIG)
    baselines_astropy = generate_via_astropy(coordinates_m, RA_RAD, DEC_RAD, OBS_TIME)
    nrows = baselines_astropy.shape[0]
    n_antennas = coordinates_m.shape[0]

    ms_path = tmp_path / "uvw_reference.ms"
    visibility_table = {
        "uvw_m": baselines_astropy,
        "antenna1": np.zeros(nrows, dtype=np.int32),
        "antenna2": np.ones(nrows, dtype=np.int32),
        "time_mjd_s": np.full(nrows, OBS_TIME.mjd * 86400.0, dtype=np.float64),
        "interval_s": np.full(nrows, 1.0, dtype=np.float64),
        "exposure_s": np.full(nrows, 1.0, dtype=np.float64),
        "data": np.zeros((nrows, 1, 1), dtype=np.complex64),
        "model_data": np.zeros((nrows, 1, 1), dtype=np.complex64),
        "flag": np.zeros((nrows, 1, 1), dtype=bool),
        "weight": np.ones((nrows, 1), dtype=np.float32),
        "sigma": np.ones((nrows, 1), dtype=np.float32),
        "channel_freq_hz": np.array([230e9], dtype=np.float64),
        "antenna_names": [f"ANT{i:02d}" for i in range(n_antennas)],
        "antenna_positions_m": coordinates_m,
        "field_ra_rad": float(RA_RAD.to_value(u.rad)),
        "field_dec_rad": float(DEC_RAD.to_value(u.rad)),
        "observation_date": "2024-01-01",
    }
    try: 
        export_native_ms(
            ms_path=ms_path,
            visibility_table=visibility_table,
            project_name="uvw-test",
            source_name="uvw-test-source",
            telescope_name="ALMA",
        )
    except ImportError as e:
        pytest.skip(str(e))
    try:
        from casacore.tables import table
    except ImportError:
        pytest.skip("python-casacore not available")

    tb = table(str(ms_path), readonly=True)
    uvw_read = tb.getcol("UVW").T
    tb.close()

    assert uvw_read.shape == baselines_astropy.shape
    np.testing.assert_allclose(uvw_read, baselines_astropy, rtol=1e-10, atol=1e-10)