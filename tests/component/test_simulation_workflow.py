"""Component tests for simulation workflow."""

import pandas as pd

from almasim.services.simulation import SimulationParams
from almasim.services.astro.spectral import sample_given_redshift
from almasim.services import astro


def test_simulation_params_from_metadata_row(tmp_path, main_dir, test_data_dir):
    """Test creating SimulationParams from metadata row."""
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    rest_frequency, _ = astro.get_line_info(main_dir)
    sample = sample_given_redshift(metadata, 1, rest_frequency, False, None)
    row = sample.iloc[0]

    params = SimulationParams.from_metadata_row(
        row,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path,
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
    )

    assert params.source_name == row["ALMA_source_name"]
    assert params.member_ouid == row["member_ous_uid"]
    assert params.band == row["Band"]
    assert params.ra == row["RA"]
    assert params.dec == row["Dec"]


def test_simulation_params_overrides(tmp_path, main_dir, test_data_dir):
    """Test SimulationParams with overrides."""
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    rest_frequency, _ = astro.get_line_info(main_dir)
    sample = sample_given_redshift(metadata, 1, rest_frequency, False, None)
    row = sample.iloc[0]

    params = SimulationParams.from_metadata_row(
        row,
        idx=0,
        main_dir=main_dir,
        output_dir=tmp_path,
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="test",
        source_type="gaussian",
        n_pix=256,
        n_channels=64,
    )

    assert params.source_type == "gaussian"
    assert params.n_pix == 256
    assert params.n_channels == 64
