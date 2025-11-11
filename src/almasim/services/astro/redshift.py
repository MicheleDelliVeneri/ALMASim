"""Redshift calculation functions."""
import astropy.units as U


def compute_redshift(rest_frequency, observed_frequency):
    """
    Computes the redshift of a source given the rest frequency and the observed frequency.

    Args:
        rest_frequency (astropy Unit): Rest frequency of the source in GHz.
        observed_frequency (astropy Unit): Observed frequency of the source in GHz.

    Returns:
        float: Redshift of the source.

    Raises:
        ValueError: If either input argument is non-positive.
    """
    # Input validation
    if rest_frequency <= 0 or observed_frequency <= 0:
        raise ValueError("Rest and observed frequencies must be positive values.")
    if rest_frequency < observed_frequency:
        raise ValueError("Observed frequency must be lower than the rest frequency.")

    # Compute redshift
    redshift = (
        rest_frequency.value - observed_frequency.value
    ) / observed_frequency.value
    return redshift


