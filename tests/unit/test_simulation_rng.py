"""One seed drives every random draw of a run (MADIV-28 T5.1)."""

import numpy as np
import pytest

from almasim.services.interferometry.simulator import ChannelImagingConfig
from almasim.services.simulation.rng import run_rng, stream_entropy

pytestmark = pytest.mark.unit


def test_a_seed_gives_the_same_independent_streams():
    assert np.array_equal(run_rng(5, "sky").random(4), run_rng(5, "sky").random(4))
    assert not np.array_equal(run_rng(5, "sky").random(4), run_rng(5, "interferometer").random(4))
    assert not np.array_equal(run_rng(5, "sky").random(4), run_rng(6, "sky").random(4))
    assert stream_entropy(5, "interferometer", 2) == (5, 1, 2)


def test_without_a_seed_every_stream_is_fresh():
    assert stream_entropy(None, "sky") is None
    assert not np.array_equal(run_rng(None, "sky").random(4), run_rng(None, "sky").random(4))


def _config(entropy):
    return ChannelImagingConfig(
        Npix=8, Nant=2, Hcov=[0.0, 1.0], nH=2, antPos=[], robfac=0.0, trlat=[0.0, 1.0],
        trdec=[0.0, 1.0], Diameters=[12.0, 12.0], imsize=1.0, Xmax=1.0, lfac=1.0,
        distmat=np.zeros((8, 8)), Nphf=8, Np4=2, zooming=0, header={}, robust=0.0,
        scan_time_s=1.0, noise_entropy=entropy,
    )  # fmt: skip


def test_each_channel_has_its_own_reproducible_noise():
    config = _config((5, 1, 0))
    assert np.array_equal(config.channel_rng(3).random(4), config.channel_rng(3).random(4))
    assert not np.array_equal(config.channel_rng(3).random(4), config.channel_rng(4).random(4))
    unseeded = _config(None)
    assert not np.array_equal(unseeded.channel_rng(3).random(4), unseeded.channel_rng(3).random(4))


def test_image_templates_are_chosen_reproducibly(tmp_path):
    from almasim.skymodels.image_template import ImageTemplateSkyModel

    for name in ["c.png", "a.png", "b.png"]:
        (tmp_path / name).write_bytes(b"")
    model = ImageTemplateSkyModel.__new__(ImageTemplateSkyModel)
    model.data_path = str(tmp_path)
    assert list(model._image_files()) == ["a.png", "b.png", "c.png"]
    picks = [np.random.default_rng(1).choice(model._image_files()) for _ in range(2)]
    assert picks[0] == picks[1]


def test_serendipitous_uniform_accepts_a_reversed_range():
    """A continuum sensitivity above the brightest line used to crash Generator.uniform."""
    from almasim.skymodels.serendipitous import _uniform

    values = _uniform(np.random.default_rng(0), 2.0, 1.0, 50)
    assert values.shape == (50,)
    assert np.all((values > 1.0) & (values <= 2.0))
