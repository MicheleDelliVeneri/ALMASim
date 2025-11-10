"""Component tests for spectral data processing."""
import pytest
import numpy as np
from pathlib import Path

from almasim.services.astro.spectral import (
    sample_given_redshift,
    process_spectral_data,
    find_compatible_lines,
    sed_reading,
)


def test_sample_given_redshift(test_data_dir):
    """Test sampling metadata by redshift."""
    import pandas as pd
    
    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    rest_frequency = np.array([100.0, 200.0, 300.0])
    
    sample = sample_given_redshift(
        metadata,
        n=5,
        rest_frequency=rest_frequency,
        extended=False,
        zmax=2.0,
    )
    
    assert len(sample) <= 5
    assert "redshift" in sample.columns
    assert "rest_frequency" in sample.columns
    assert all(sample["redshift"] >= 0)


def test_process_spectral_data(main_dir):
    """Test processing spectral data."""
    source_type = "point"
    redshift = 0.1
    central_frequency = 100.0  # GHz
    delta_freq = 10.0  # GHz
    source_frequency = 100.0  # GHz
    n_channels = 32
    lum_infrared = 1e10
    cont_sens = 0.1  # Jy
    
    result = process_spectral_data(
        source_type,
        main_dir,
        redshift,
        central_frequency,
        delta_freq,
        source_frequency,
        n_channels,
        lum_infrared,
        cont_sens,
        line_names=None,
        n_lines=1,
        remote=False,
    )
    
    (
        continum,
        line_fluxes,
        line_names,
        redshift_out,
        line_frequency,
        line_indexes,
        n_channels_out,
        bandwidth,
        freq_support,
        cont_frequencies,
        fwhms,
        lum_infrared_out,
    ) = result
    
    assert len(continum) == n_channels
    assert len(line_fluxes) > 0
    assert redshift_out == redshift
    assert n_channels_out == n_channels
    assert lum_infrared_out >= lum_infrared


def test_sed_reading(main_dir):
    """Test reading SED templates."""
    sed, flux_infrared, lum_infrared = sed_reading(
        source_type="point",
        path=main_dir / "brightnes",
        cont_sens=0.1,
        freq_min=90.0,
        freq_max=110.0,
        remote=False,
        lum_infrared=1e10,
        redshift=0.1,
    )
    
    assert "GHz" in sed.columns
    assert "Jy" in sed.columns
    assert flux_infrared > 0
    assert lum_infrared > 0


