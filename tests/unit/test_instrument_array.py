"""The shared ALMA array-type rule (MADIV-28 T1.6)."""

import pytest

from almasim.services.instrument.array import derive_array_type
from almasim.services.metadata import derive_array_type as metadata_derive_array_type
from almasim.services.metadata.tap.queries import _derive_array_type as queries_derive
from almasim.services.observation_plan import derive_array_type as plan_derive_array_type

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("antennas", "expected"),
    [
        ("DA41 DV02", "12m"),
        ("CM01 CM02", "7m"),
        ("PM01", "TP"),
        ("DA41 CM01", "12m+7m"),
        ("DA41 PM01", "12m+TP"),
        ("CM01 PM01", "7m+TP"),
        ("DA41 CM01 PM01", "12m+7m+TP"),
        ("da41 cm01", "12m+7m"),
        ("", ""),
        ("XX01", ""),
        (None, ""),
    ],
)
def test_derive_array_type(antennas, expected):
    assert derive_array_type(antennas) == expected


def test_metadata_and_tap_use_the_shared_rule():
    assert metadata_derive_array_type is derive_array_type
    assert queries_derive is derive_array_type


def test_observation_plan_reports_tp_with_12m_and_7m():
    # The planner's own copy returned "12m+7m" here: its 12m+7m+TP branch was
    # unreachable because the 12m+7m test came first.
    assert plan_derive_array_type("DA41 CM01 PM01") == "12m+7m+TP"


def test_observation_plan_falls_back_to_12m():
    assert plan_derive_array_type("") == "12m"
    assert plan_derive_array_type("XX01") == "12m"
