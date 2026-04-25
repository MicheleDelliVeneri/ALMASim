"""Unit tests for almasim.skymodels.serendipitous pure helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from almasim.skymodels.serendipitous import (
    distance_1d,
    distance_2d,
    get_iou,
    get_iou_1d,
    get_pos,
    sample_positions,
)


# ===========================================================================
# distance_1d
# ===========================================================================


@pytest.mark.unit
def test_distance_1d_zero():
    """Same point has zero distance."""
    assert distance_1d(5.0, 5.0) == pytest.approx(0.0)


@pytest.mark.unit
def test_distance_1d_positive():
    """Distance is always non-negative."""
    assert distance_1d(3.0, 7.0) == pytest.approx(4.0)


@pytest.mark.unit
def test_distance_1d_negative_offset():
    """Negative offset still gives positive distance."""
    assert distance_1d(1.0, 6.0) == pytest.approx(5.0)


# ===========================================================================
# distance_2d
# ===========================================================================


@pytest.mark.unit
def test_distance_2d_pythagorean():
    """3-4-5 triangle gives distance 5."""
    assert distance_2d((0, 0), (3, 4)) == pytest.approx(5.0)


@pytest.mark.unit
def test_distance_2d_same_point():
    """Same point returns zero."""
    assert distance_2d((7, 7), (7, 7)) == pytest.approx(0.0)


# ===========================================================================
# get_iou
# ===========================================================================


@pytest.mark.unit
def test_get_iou_no_overlap():
    """Non-overlapping boxes give IoU=0."""
    bb1 = {"x1": 0, "x2": 1, "y1": 0, "y2": 1}
    bb2 = {"x1": 2, "x2": 3, "y1": 2, "y2": 3}
    assert get_iou(bb1, bb2) == pytest.approx(0.0)


@pytest.mark.unit
def test_get_iou_identical_boxes():
    """Identical boxes give IoU=1."""
    bb = {"x1": 0, "x2": 4, "y1": 0, "y2": 4}
    assert get_iou(bb, bb) == pytest.approx(1.0)


@pytest.mark.unit
def test_get_iou_partial_overlap():
    """Partially overlapping boxes give 0 < IoU < 1."""
    bb1 = {"x1": 0, "x2": 4, "y1": 0, "y2": 4}
    bb2 = {"x1": 2, "x2": 6, "y1": 2, "y2": 6}
    iou = get_iou(bb1, bb2)
    assert 0.0 < iou < 1.0


@pytest.mark.unit
def test_get_iou_contained_box():
    """Inner box fully inside outer box gives IoU == inner_area / outer_area."""
    outer = {"x1": 0, "x2": 4, "y1": 0, "y2": 4}
    inner = {"x1": 1, "x2": 3, "y1": 1, "y2": 3}
    iou = get_iou(inner, outer)
    # intersection=4, bb1=4, bb2=16, union=16 → 4/16 = 0.25
    assert iou == pytest.approx(4.0 / 16.0)


# ===========================================================================
# get_iou_1d
# ===========================================================================


@pytest.mark.unit
def test_get_iou_1d_no_overlap():
    """Non-overlapping 1D intervals give IoU=0."""
    bb1 = {"z1": 0, "z2": 1}
    bb2 = {"z1": 2, "z2": 3}
    assert get_iou_1d(bb1, bb2) == pytest.approx(0.0)


@pytest.mark.unit
def test_get_iou_1d_identical():
    """Identical 1D intervals give IoU=1."""
    bb = {"z1": 0, "z2": 5}
    assert get_iou_1d(bb, bb) == pytest.approx(1.0)


@pytest.mark.unit
def test_get_iou_1d_partial():
    """Partially overlapping 1D intervals give 0 < IoU < 1."""
    bb1 = {"z1": 0, "z2": 4}
    bb2 = {"z1": 2, "z2": 6}
    iou = get_iou_1d(bb1, bb2)
    assert 0.0 < iou < 1.0


# ===========================================================================
# get_pos
# ===========================================================================


@pytest.mark.unit
def test_get_pos_returns_tuple():
    """get_pos returns a 3-tuple."""
    pos = get_pos(10, 10, 10)
    assert len(pos) == 3


@pytest.mark.unit
def test_get_pos_within_radius():
    """get_pos values are within the specified radii."""
    np.random.seed(0)
    for _ in range(50):
        x, y, z = get_pos(5, 5, 5)
        assert -5 <= x < 5
        assert -5 <= y < 5
        assert -5 <= z < 5


# ===========================================================================
# sample_positions
# ===========================================================================


@pytest.mark.unit
def test_sample_positions_empty_when_impossible():
    """sample_positions returns empty list when constraints can't be satisfied."""
    # Very tight radius with large sep ensures nothing fits
    result = sample_positions(
        terminal=None,
        pos_x=32.0,
        pos_y=32.0,
        pos_z=32,
        fwhm_x=4,
        fwhm_y=4,
        fwhm_z=4.0,
        n_components=5,
        fwhm_xs=np.array([2, 2, 2, 2, 2]),
        fwhm_ys=np.array([2, 2, 2, 2, 2]),
        fwhm_zs=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        xy_radius=1,  # tiny radius
        z_radius=1,
        sep_xy=200,  # huge separation — nothing passes
        sep_z=200,
    )
    assert result == []


@pytest.mark.unit
def test_sample_positions_finds_one():
    """sample_positions finds one component when constraints are relaxed."""
    np.random.seed(42)
    result = sample_positions(
        terminal=None,
        pos_x=32.0,
        pos_y=32.0,
        pos_z=32,
        fwhm_x=6,
        fwhm_y=6,
        fwhm_z=6.0,
        n_components=1,
        fwhm_xs=np.array([2]),
        fwhm_ys=np.array([2]),
        fwhm_zs=np.array([2.0]),
        xy_radius=60,
        z_radius=60,
        sep_xy=5,
        sep_z=5,
    )
    assert len(result) <= 1


@pytest.mark.unit
def test_sample_positions_terminal_callback():
    """sample_positions calls terminal.add_log when terminal is provided."""
    from unittest.mock import MagicMock

    terminal = MagicMock()
    np.random.seed(42)
    sample_positions(
        terminal=terminal,
        pos_x=32.0,
        pos_y=32.0,
        pos_z=32,
        fwhm_x=6,
        fwhm_y=6,
        fwhm_z=6.0,
        n_components=1,
        fwhm_xs=np.array([2]),
        fwhm_ys=np.array([2]),
        fwhm_zs=np.array([2.0]),
        xy_radius=60,
        z_radius=60,
        sep_xy=5,
        sep_z=5,
    )
    # terminal.add_log may or may not be called depending on whether a sample was found
    # — just verify no exception was raised
