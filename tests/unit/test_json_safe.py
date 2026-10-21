"""json_safe turns nested metadata into JSON values (MADIV-28 T8.2: a direct test)."""

import json
from pathlib import Path

import astropy.units as U
import numpy as np
import pytest

from almasim.services.json_safe import json_safe

pytestmark = pytest.mark.unit


def test_nested_numpy_quantities_and_paths_become_json():
    value = {
        1: (np.int64(3), np.float32(0.5)),
        "q": 2.0 * U.GHz,
        "arr": np.arange(3),
        "path": Path("/a/b"),
        "other": object,
        "ok": [True, None, "s"],
    }
    safe = json_safe(value)
    assert safe["1"] == [3, 0.5]
    assert safe["q"] == {"value": 2.0, "unit": "GHz"}
    assert safe["arr"] == [0, 1, 2]
    assert safe["path"] == "/a/b"
    assert safe["other"] == str(object)
    json.dumps(safe)


def test_array_likes_go_through_tolist():
    class ArrayLike:
        def tolist(self):
            return [np.int32(1), 2]

    assert json_safe(ArrayLike()) == [1, 2]
