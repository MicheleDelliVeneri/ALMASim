"""Unit tests for TNG snapshot functions."""

from almasim.services.astro.tng import redshift_to_snapshot


def test_redshift_to_snapshot_z0():
    """Test snapshot lookup for redshift 0."""
    snapshot = redshift_to_snapshot(0.0)
    assert snapshot == 99


def test_redshift_to_snapshot_z1():
    """Test snapshot lookup for redshift 1."""
    snapshot = redshift_to_snapshot(1.0)
    assert snapshot == 50


def test_redshift_to_snapshot_z2():
    """Test snapshot lookup for redshift 2."""
    snapshot = redshift_to_snapshot(2.0)
    assert snapshot == 33


def test_redshift_to_snapshot_intermediate():
    """Test snapshot lookup for intermediate redshift."""
    snapshot = redshift_to_snapshot(0.5)
    assert snapshot in range(50, 99)  # Should be between z=1 and z=0
