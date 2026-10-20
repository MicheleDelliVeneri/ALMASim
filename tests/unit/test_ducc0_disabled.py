"""The ducc0 channel imager stays disabled until its uv geometry is fixed (MADIV-28 T1.3).

It passed the simulator's local (x, y) antenna offsets in km to
generate_via_astropy, which reads two columns as (lat, lon) in degrees: a 500 m
baseline became ~55 km and, with the km wavelengths, uv distances came out
~1e5 too large. It also imaged one snapshot instead of the hour-angle track.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from typer.testing import CliRunner

from almasim import cli
from almasim.services.interferometry.core import Interferometer

pytestmark = pytest.mark.unit


def test_the_simulator_refuses_ducc0():
    args = {
        name: MagicMock()
        for name in (
            "skymodel",
            "backend",
            "ra",
            "dec",
            "central_freq",
            "bandwidth",
            "header",
        )
    }
    with pytest.raises(ValueError, match="ducc0.* is disabled"):
        Interferometer(
            idx=0,
            main_dir="/nonexistent",
            output_dir="/nonexistent",
            fov=10.0,
            antenna_array="A001",
            noise=0.0,
            snr=1.0,
            integration_time=1.0,
            observation_date="2024-01-01",
            save_mode="memory",
            robust=0.0,
            imaging_algorithm="ducc0",
            **args,
        )


def test_the_cli_offers_only_legacy(tmp_path):
    result = CliRunner().invoke(
        cli.app,
        [
            "simulation",
            "run",
            "--metadata-path",
            str(tmp_path / "m.csv"),
            "--imaging-algorithm",
            "ducc0",
        ],
    )
    assert result.exit_code == 2
    assert "--imaging-algorithm must be one of: legacy" in result.output


def test_set_primary_beam_no_longer_writes_a_beam_into_the_header():
    from astropy.io import fits

    from almasim.services.interferometry.imaging import set_primary_beam

    header = fits.Header()
    n = 8
    set_primary_beam(
        header, -np.ones((n, n)), [1e-6, 1e-6, 1e-6], [12.0], [np.ones((n, n))], np.ones((n, n))
    )
    assert "BMAJ" not in header and "BMIN" not in header
