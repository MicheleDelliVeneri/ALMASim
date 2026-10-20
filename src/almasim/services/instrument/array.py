"""ALMA array types from antenna names."""

from __future__ import annotations

# Antenna name prefixes of each array: DA/DV are 12 m dishes, CM the 7 m
# dishes of the ACA, PM the total-power antennas.
_ARRAY_PREFIXES = (("12m", ("DA", "DV")), ("7m", ("CM",)), ("TP", ("PM",)))


def derive_array_type(antennas: object) -> str:
    """The arrays present in an antenna string, '+'-joined: '12m', '7m+TP', '12m+7m+TP', ...

    Returns "" for an empty or non-string value, or when no ALMA antenna is
    recognised.
    """
    if not isinstance(antennas, str) or not antennas:
        return ""
    upper = antennas.upper()
    return "+".join(
        array for array, prefixes in _ARRAY_PREFIXES if any(p in upper for p in prefixes)
    )
