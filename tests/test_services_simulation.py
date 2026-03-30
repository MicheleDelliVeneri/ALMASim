from pathlib import Path

import pandas as pd

from almasim.services import astro
from almasim.services import simulation as sim
from almasim.services.simulation import SimulationParams


def _sample_metadata_row():
    repo_root = Path(__file__).resolve().parents[1]
    main_dir = repo_root / "src" / "almasim"
    metadata = pd.read_csv(repo_root / "data" / "qso_metadata.csv")
    rest_frequency, _ = astro.get_line_info(main_dir)
    from almasim.services.astro.spectral import sample_given_redshift
    sample = sample_given_redshift(metadata, 1, rest_frequency, False, None)
    return main_dir, sample.iloc[0]


def test_from_metadata_row_populates_required_fields(tmp_path):
    main_dir, row = _sample_metadata_row()
    params = SimulationParams.from_metadata_row(
        row,
        idx=5,
        main_dir=main_dir,
        output_dir=tmp_path,
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="demo",
    )

    assert params.source_name == row["ALMA_source_name"]
    assert params.member_ouid == row["member_ous_uid"]
    assert params.freq_support == row["Freq.sup."]
    assert params.band == row["Band"]
    assert params.ra == row["RA"]
    assert params.redshift == row["redshift"]
    assert params.rest_frequency == row["rest_frequency"]
    assert params.n_pix is None
    assert params.n_channels is None
    assert params.save_mode == "npz"
    assert params.ncpu >= 1
    assert params.main_dir == str(main_dir.resolve())
    assert params.output_dir == str(tmp_path.resolve())


def test_from_metadata_row_honours_overrides(tmp_path):
    main_dir, row = _sample_metadata_row()
    params = SimulationParams.from_metadata_row(
        row,
        idx=9,
        main_dir=main_dir,
        output_dir=tmp_path,
        tng_dir=tmp_path / "tng",
        galaxy_zoo_dir=tmp_path / "galaxy_zoo",
        hubble_dir=tmp_path / "hubble",
        project_name="demo",
        source_type="extended",
        snr=3.0,
        save_mode="fits",
        n_pix=512,
        n_channels=128,
        n_lines=2,
        line_names=["CO(3-2)", "CII"],
        rest_frequency=123.0,
        redshift=0.42,
        lum_infrared=5.0e11,
        ncpu=4,
        tng_api_key="secret",
        inject_serendipitous=True,
        remote=True,
    )

    assert params.source_type == "extended"
    assert params.snr == 3.0
    assert params.save_mode == "fits"
    assert params.n_pix == 512
    assert params.n_channels == 128
    assert params.n_lines == 2
    assert params.line_names == ["CO(3-2)", "CII"]
    assert params.rest_frequency == 123.0
    assert params.redshift == 0.42
    assert params.lum_infrared == 5.0e11
    assert params.ncpu == 4
    assert params.tng_api_key == "secret"
    assert params.inject_serendipitous is True
    assert params.remote is True
