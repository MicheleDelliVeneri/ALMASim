"""Jobs: ids, resources and manifests (web console plan, J1.1)."""

from __future__ import annotations

import pytest

from almasim.jobs import Job, JobState, Resources, ResourcesError, is_job_id, new_job_id
from almasim.jobs.model import MANIFEST_SCHEMA

pytestmark = pytest.mark.unit


def _job(**overrides) -> Job:
    fields = {
        "id": new_job_id(),
        "owner": "astro",
        "argv": ["image", "image-from-ms", "params.csv", "out", "--cube"],
        "executor": "slurm",
    }
    fields.update(overrides)
    return Job(**fields)


def test_job_ids_are_ulids_that_sort_by_time():
    early = new_job_id(now_ms=1_000, entropy=bytes(10))
    later = new_job_id(now_ms=2_000, entropy=bytes(10))
    assert len(early) == 26 and is_job_id(early)
    assert early < later
    assert new_job_id() != new_job_id()
    assert new_job_id(now_ms=0, entropy=bytes(10)) == "0" * 26


def test_job_id_inputs_are_checked():
    with pytest.raises(ValueError):
        new_job_id(now_ms=-1)
    with pytest.raises(ValueError):
        new_job_id(entropy=b"short")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("partition", "zwickyA-CPU\n#SBATCH --uid=0"),
        ("partition", "two words"),
        ("account", "a;b"),
        ("cpus", 0),
        ("cpus", True),
        ("cpus", "4"),
        ("memory", "16GB\n"),
        ("memory", "lots"),
        ("time", "2h"),
        ("time", "02:00:00\n"),
    ],
)
def test_resources_refuse_values_slurm_would_not_take(field, value):
    with pytest.raises(ResourcesError, match=field):
        Resources(**{field: value})


def test_resources_take_slurm_forms():
    r = Resources(
        partition="zwickyA-CPU", account="ska.madiv", cpus=4, memory="16G", time="2-00:00:00"
    )
    assert Resources.from_dict(r.to_dict()) == r
    assert Resources.from_dict({"cpus": 2, "unknown": 1}).cpus == 2


def test_a_job_round_trips_through_its_manifest_form():
    job = _job(resources=Resources(cpus=4), params={"cube": True}, parents=[new_job_id()])
    again = Job.from_dict(job.to_dict())
    assert again == job
    assert again.to_dict()["state"] == "pending"
    assert again.schema == MANIFEST_SCHEMA


def test_the_command_is_the_words_before_the_first_option():
    assert _job().command == "image image-from-ms"
    assert _job(argv=["metadata", "--band", "3"]).command == "metadata"
    assert _job(argv=["clean", "ms.ms"]).command == "clean ms.ms"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": "not-an-id"}, "ULID"),
        ({"executor": "kubernetes"}, "executor"),
        ({"argv": []}, "argv"),
        ({"argv": ["image", 3]}, "argv"),
    ],
)
def test_a_job_checks_its_fields(overrides, message):
    with pytest.raises(ValueError, match=message):
        _job(**overrides)


def test_a_newer_manifest_is_refused():
    data = _job().to_dict()
    data["schema"] = MANIFEST_SCHEMA + 1
    with pytest.raises(ValueError, match="newer"):
        Job.from_dict(data)


def test_finished_states():
    assert {s for s in JobState if s.finished} == {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.LOST,
    }
