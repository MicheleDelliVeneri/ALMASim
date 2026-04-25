"""Component tests for spectral data processing."""

from almasim.services.astro.spectral import (
    process_spectral_data,
    sample_given_redshift,
    sed_reading,
)


def test_sample_given_redshift(test_data_dir, main_dir):
    """Test sampling metadata by redshift."""
    import pandas as pd

    from almasim.services.astro.lines import get_line_info

    metadata = pd.read_csv(test_data_dir / "qso_metadata.csv")
    # Get actual rest frequencies from the calibrated lines file
    # These should be higher than observed frequencies (which are ~250-500 GHz)
    rest_frequency, _ = get_line_info(main_dir)
    # Use a subset of rest frequencies that are higher than typical observed frequencies
    # CO lines are typically 100-500 GHz rest, so use those
    rest_frequency = rest_frequency[rest_frequency > 100.0][:3]  # Get 3 frequencies > 100 GHz

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
