"""User text is quoted safely in ADQL (MADIV-28 T1.14)."""

from __future__ import annotations

import pytest

from almasim.services.metadata.tap.service import (
    ExclusionFilters,
    InclusionFilters,
    _build_exclusion_conditions,
    _build_inclusion_conditions,
)

pytestmark = pytest.mark.unit

_NO_DEFAULTS = dict(public_only=False, science_only=False, exclude_mosaic=False)


def test_quotes_in_text_are_doubled_not_closing_the_literal():
    conds = _build_inclusion_conditions(
        InclusionFilters(source_name="x' OR 'a'='a", proposal_id=["2023.1.0'1"], **_NO_DEFAULTS)
    )
    assert "target_name LIKE '%x'' OR ''a''=''a%'" in conds
    assert "proposal_id IN ('2023.1.0''1')" in conds


def test_exclusions_are_quoted_too():
    conds = _build_exclusion_conditions(ExclusionFilters(source_name=["O'Brien"]))
    assert conds == ["LOWER(target_name) NOT LIKE LOWER('%O''Brien%')"]


def test_plain_text_is_unchanged():
    conds = _build_inclusion_conditions(
        InclusionFilters(source_name="NGC 253", qa2_status=["T", "X"], **_NO_DEFAULTS)
    )
    assert "target_name LIKE '%NGC 253%'" in conds
    assert "qa2_passed IN ('T', 'X')" in conds


@pytest.mark.parametrize("bad", ["1) OR (1=1", "nan", float("inf")])
def test_ranges_must_be_finite_numbers(bad):
    with pytest.raises(ValueError):
        _build_inclusion_conditions(InclusionFilters(fov_range=(0, bad), **_NO_DEFAULTS))
