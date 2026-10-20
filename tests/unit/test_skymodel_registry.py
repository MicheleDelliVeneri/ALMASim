"""The sky-model registry (MADIV-28 T5.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from almasim.skymodels.registry import SKYMODELS, skymodel_spec

pytestmark = pytest.mark.unit

BRIGHTNESS_DIR = Path(__file__).resolve().parents[2] / "src" / "almasim" / "brightnes"


def test_every_model_family_has_a_bundled_sed_template():
    for name, spec in SKYMODELS.items():
        if spec.external:
            assert spec.sed_template is None, name
        else:
            assert (BRIGHTNESS_DIR / spec.sed_template).is_file(), name


def test_lookup_is_case_insensitive_and_names_the_choices():
    assert skymodel_spec("Galaxy-Zoo").dataset == "galaxy-zoo"
    with pytest.raises(ValueError, match="point, gaussian"):
        skymodel_spec("spiral")


def test_the_cli_and_the_external_list_come_from_the_registry():
    from almasim import cli_simulation
    from almasim.services.external_skymodel import EXTERNAL_SOURCE_TYPES

    assert cli_simulation._SOURCE_TYPES == [
        "point",
        "gaussian",
        "diffuse",
        "galaxy-zoo",
        "molecular",
        "hubble-100",
    ]
    assert EXTERNAL_SOURCE_TYPES == {
        "external-fits-image",
        "external-fits-cube",
        "external-components",
    }


def test_sed_reading_refuses_external_and_unknown_types(tmp_path):
    from almasim.services.astro.spectral import sed_reading

    for source_type in ("external-fits-cube", "spiral"):
        with pytest.raises(ValueError):
            sed_reading(source_type, tmp_path, 1.0, 100.0, 101.0, remote=False)
