"""The sky-model families ALMASim can simulate, in one table (MADIV-28 T5.6).

The CLI and the backend validate ``source_type`` against ``SKYMODELS``; the
spectral setup takes the SED template from it, and the run fetches the image
dataset a family draws from. The builders themselves stay in the simulation
(``_create_sky_model``), which needs the run's state.
"""

from __future__ import annotations

from dataclasses import dataclass

_STAR_FORMING = "SED_low_z_warm_star_forming_galaxy.dat"
_TYPE2_AGN = "SED_low_z_type2_AGN.dat"


@dataclass(frozen=True)
class SkyModelSpec:
    """One sky-model family."""

    label: str
    # SED template under the brightness directory; None for an external input.
    sed_template: str | None
    # Redshift used when the metadata gives none.
    default_redshift: float = 0.05
    # Image dataset the model draws its morphology from ("galaxy-zoo", "hubble-100").
    dataset: str | None = None
    # The cube comes from the user's file or component table, not from a model.
    external: bool = False
    # Offered by ``almasim simulation run --source-type``.
    in_cli: bool = True


SKYMODELS: dict[str, SkyModelSpec] = {
    "point": SkyModelSpec("Point", _TYPE2_AGN),
    "gaussian": SkyModelSpec("Gaussian", _TYPE2_AGN),
    "extended": SkyModelSpec("Extended (TNG)", _STAR_FORMING, 10**-4, in_cli=False),
    "diffuse": SkyModelSpec("Diffuse", _STAR_FORMING, 10**-4),
    "galaxy-zoo": SkyModelSpec("Galaxy Zoo", _STAR_FORMING, 10**-4, dataset="galaxy-zoo"),
    "molecular": SkyModelSpec("Molecular", _STAR_FORMING, 10**-4),
    "hubble-100": SkyModelSpec("Hubble", _STAR_FORMING, 10**-4, dataset="hubble-100"),
    "external-fits-image": SkyModelSpec("External FITS image", None, external=True),
    "external-fits-cube": SkyModelSpec("External FITS cube", None, external=True),
    "external-components": SkyModelSpec("External component table", None, external=True),
}


def skymodel_spec(source_type: str) -> SkyModelSpec:
    """The family for ``source_type`` (case-insensitive); ValueError for an unknown one."""
    spec = SKYMODELS.get(str(source_type).lower())
    if spec is None:
        raise ValueError(
            f"Unknown source type {source_type!r}; expected one of: {', '.join(SKYMODELS)}"
        )
    return spec
