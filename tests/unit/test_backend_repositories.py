"""The backend's repositories and the cache query on SQLite (MADIV-28 T6.10)."""

from __future__ import annotations

import importlib
import uuid
from datetime import datetime

import pytest

from almasim.services.metadata.filters import ExclusionFilters, InclusionFilters

pytestmark = pytest.mark.unit


@pytest.fixture
def db(backend_db):
    models = backend_db.models
    models.Base.metadata.create_all(
        backend_db.session.get_bind(),
        tables=[models.SimulationJob.__table__, models.SimulationLog.__table__],
    )
    return backend_db


def test_simulation_repository_round_trip(db):
    repository = importlib.import_module("app.db.repositories").SimulationRepository(db.session)
    sim_id = uuid.uuid4()
    job = repository.create_simulation_job(
        "NGC 1068", simulation_id=sim_id, status="queued", line_names=["CO(3-2)"]
    )
    assert repository.get_simulation_by_id(sim_id).id == job.id

    running = repository.update_simulation_status(sim_id, "running", progress=0.5, message="go")
    assert running.started_at is not None and running.progress == 0.5
    done = repository.update_simulation_status(sim_id, "completed", current_step="Done", error=None)
    assert done.completed_at is not None and done.current_step == "Done"
    assert repository.update_simulation_status(uuid.uuid4(), "failed") is None

    repository.add_simulation_log(sim_id, "first")
    repository.add_simulation_log(sim_id, "second", level="WARNING")
    assert {log.message for log in repository.get_simulation_logs(sim_id)} == {"first", "second"}

    other = uuid.uuid4()
    repository.create_simulation_job("Sun", simulation_id=other, status="failed")
    assert {s.simulation_id for s in repository.list_simulations()} == {sim_id, other}
    assert [s.simulation_id for s in repository.list_simulations(status="failed")] == [other]
    assert len(repository.list_simulations(limit=1, offset=1)) == 1
    assert len(repository.list_simulations(status="completed", offset=1)) == 0


@pytest.fixture
def cache(backend_db):
    csv = importlib.import_module("app.db.csv_importer").CSVImporter(backend_db.session)

    def add(uid, target="NGC 1068", **row):
        observation = csv.parse_csv_row(
            {"member_ous_uid": uid, "ALMA_source_name": target, "Band": "6", **row}, "test"
        )
        observation.science_keywords.append(csv.get_or_create_keyword("Galaxies"))
        observation.scientific_category = csv.get_or_create_category("Active galaxies")
        backend_db.session.add(observation)
        backend_db.session.commit()
        return observation

    backend_db.add = add
    return backend_db


def _uids(observations):
    return sorted(observation.member_ous_uid for observation in observations)


def test_cache_query_ranges_arrays_and_exclusions(cache):
    a = cache.add("uid://A/a", antenna_arrays="A001:DA41 A002:CM01", **{"FOV": "20", "Freq": "230"})
    a.schedblock_name = "X_TM1_C-3"
    a.spatial_resolution = 0.5
    a.obs_type = "S"
    a.obs_release_date = datetime(2020, 1, 1)
    b = cache.add("uid://A/b", antenna_arrays="A001:PM01", **{"FOV": "60", "Freq": "345"})
    b.schedblock_name = "X_TP"
    b.spatial_resolution = 5.0
    b.obs_type = "T"
    b.obs_release_date = datetime(2023, 1, 1)
    cache.session.commit()
    repository = importlib.import_module("app.db.repositories").ObservationRepository(cache.session)

    def uids(**filters):
        exclude = filters.pop("exclude", None)
        return _uids(repository.query_observations(InclusionFilters(**filters), exclude))

    assert uids(array_type=["7m"]) == ["uid://A/a"]
    assert uids(array_type=["TP"], array_configuration=["TP"]) == ["uid://A/b"]
    assert uids(fov_range=(10, 30), frequency_range=(200, None)) == ["uid://A/a"]
    assert uids(angular_resolution_range=(None, 1.0)) == ["uid://A/a"]
    assert uids(observation_date_range=("2022-01-01", "2024-01-01")) == ["uid://A/b"]
    assert uids(obs_type=["T"], antenna_arrays="PM01") == ["uid://A/b"]
    assert uids(exclude=ExclusionFilters(obs_type=["S"])) == ["uid://A/b"]
    assert uids(exclude=ExclusionFilters(scientific_category=["Active"])) == []
    assert len(repository.query_observations(limit=1, offset=1)) == 1

    assert repository.get_observation_by_member_uid("uid://A/b").id == b.id
    assert repository.count_observations_with_keywords() == 2
    assert repository.get_all_science_keywords() == ["Galaxies"]
