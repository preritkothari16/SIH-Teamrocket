"""Integration tests against the real Postgres + PostGIS instance.

Skipped entirely when ``DATABASE_URL`` isn't configured — everything else in
this suite stays offline per this project's convention (see CLAUDE.md's
Conventions section); these are the deliberate, documented exception.
GeoAlchemy2's ``Geometry`` columns have no SQLite equivalent, so there is no
way to exercise src/db.py's actual table without a real Postgres+PostGIS
server — mocking the ORM/SQL layer here would just test the mock, not
whether this code actually talks to the real schema correctly (which is the
whole point, after the report-endpoint and status-word bugs found earlier
this session by trusting an untested translation path). Every test cleans
up its own row via a unique per-test spill_id, regardless of pass/fail.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
from shapely.geometry import Polygon

import src.config  # noqa: F401 — importing loads .env via load_dotenv() first
from src.alerts.registry import RegistryError, SpillRegistry
from src.api.registry import get_run, list_runs

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set - skipping live Postgres integration tests",
)

GEOM = Polygon([(10.0, 55.0), (10.1, 55.0), (10.1, 55.1), (10.0, 55.1)])


@pytest.fixture()
def spill_id():
    sid = f"test_pg_{uuid.uuid4().hex[:12]}"
    yield sid
    from sqlalchemy.orm import Session

    from src.db import SpillRow, get_engine

    with Session(get_engine()) as session:
        session.query(SpillRow).filter(SpillRow.spill_id == sid).delete()
        session.commit()


class TestSpillRegistryPostgres:
    def test_register_requires_confidence(self, spill_id: str) -> None:
        with SpillRegistry() as reg:
            with pytest.raises(RegistryError, match="confidence"):
                reg.register(
                    spill_id, geometry=GEOM, centroid_lon=10.05, centroid_lat=55.05,
                    area_km2=1.0, seen_at=datetime.now(timezone.utc), status="possible",
                )

    def test_register_get_roundtrip(self, spill_id: str) -> None:
        now = datetime.now(timezone.utc)
        with SpillRegistry() as reg:
            reg.register(
                spill_id, geometry=GEOM, centroid_lon=10.05, centroid_lat=55.05,
                area_km2=1.23, seen_at=now, status="possible",
                scene_id="pg_test_scene", confidence=0.81,
            )
            fetched = reg.get(spill_id)
        assert fetched is not None
        assert fetched.status == "possible"
        assert fetched.area_km2 == 1.23
        assert fetched.centroid_lon == 10.05
        assert fetched.centroid_lat == 55.05

    def test_duplicate_register_raises(self, spill_id: str) -> None:
        now = datetime.now(timezone.utc)
        with SpillRegistry() as reg:
            reg.register(
                spill_id, geometry=GEOM, centroid_lon=10.05, centroid_lat=55.05,
                area_km2=1.0, seen_at=now, status="possible", confidence=0.7,
            )
            with pytest.raises(RegistryError):
                reg.register(
                    spill_id, geometry=GEOM, centroid_lon=10.05, centroid_lat=55.05,
                    area_km2=1.0, seen_at=now, confidence=0.7,
                )

    def test_update_preserves_first_seen_changes_status(self, spill_id: str) -> None:
        t0 = datetime.now(timezone.utc)
        with SpillRegistry() as reg:
            first = reg.register(
                spill_id, geometry=GEOM, centroid_lon=10.0, centroid_lat=55.0,
                area_km2=1.0, seen_at=t0, status="possible", confidence=0.7,
            )
            updated = reg.update(
                spill_id, geometry=GEOM, centroid_lon=10.01, centroid_lat=55.01,
                area_km2=2.0, seen_at=t0, status="active",
            )
        assert updated.first_seen == first.first_seen
        assert updated.status == "active"
        assert updated.area_km2 == 2.0

    def test_recent_and_all_include_registered(self, spill_id: str) -> None:
        now = datetime.now(timezone.utc)
        with SpillRegistry() as reg:
            reg.register(
                spill_id, geometry=GEOM, centroid_lon=10.0, centroid_lat=55.0,
                area_km2=1.0, seen_at=now, status="active", confidence=0.9,
            )
            assert spill_id in [r.spill_id for r in reg.recent(now)]
            assert spill_id in [r.spill_id for r in reg.all()]


class TestApiRegistryPostgres:
    def test_list_and_get_run_agree(self, spill_id: str) -> None:
        now = datetime.now(timezone.utc)
        scene_id = f"scene_{spill_id}"
        with SpillRegistry() as reg:
            reg.register(
                spill_id, geometry=GEOM, centroid_lon=10.0, centroid_lat=55.0,
                area_km2=3.5, seen_at=now, status="possible", scene_id=scene_id,
                confidence=0.77, major_axis_bearing=12.0, elongation=2.0,
                rules_fired=["wind"],
            )

        runs = {r["scene_id"]: r for r in list_runs()}
        assert scene_id in runs
        assert runs[scene_id]["alert_status"] == "possible"

        run = get_run(scene_id)
        assert run is not None
        assert run["alert"]["status"] == "possible"
        assert run["alert"]["spill_id"] == spill_id
        assert run["alert"]["rules_fired"] == ["wind"]
        assert run["spill"]["area_km2"] == 3.5
        assert run["spill"]["major_axis_bearing"] == 12.0
        assert run["vessels"] == []
        assert run["drift"] == {"forecast": [], "hindcast": []}

    def test_vessels_and_drift_default_to_empty_then_reflect_set_vessels_and_drift(
        self, spill_id: str,
    ) -> None:
        """Step 10.2: vessels_json/drift_json start out NULL ("not computed
        yet") and get_run() reports the same empty defaults as before this
        step existed - then set_vessels_and_drift() populates them and
        get_run() reflects exactly what was stored, not a copy of it."""
        now = datetime.now(timezone.utc)
        scene_id = f"scene_{spill_id}"
        vessels = [{
            "mmsi": "123456789", "name": "TEST VESSEL", "vessel_type": "tanker",
            "score": 0.9, "explanation": "test", "cpa_distance_km": 1.1,
            "cpa_time": now.isoformat(),
            "track": {"type": "LineString", "coordinates": [[10.0, 55.0], [10.05, 55.02]]},
        }]
        drift = {"forecast": [{"hours": 6, "time": now.isoformat(), "polygon": {"type": "Polygon", "coordinates": []}}], "hindcast": []}

        with SpillRegistry() as reg:
            reg.register(
                spill_id, geometry=GEOM, centroid_lon=10.0, centroid_lat=55.0,
                area_km2=1.0, seen_at=now, status="possible", scene_id=scene_id,
                confidence=0.8,
            )

        run_before = get_run(scene_id)
        assert run_before is not None
        assert run_before["vessels"] == []
        assert run_before["drift"] == {"forecast": [], "hindcast": []}

        with SpillRegistry() as reg:
            reg.set_vessels_and_drift(spill_id, vessels, drift)

        run_after = get_run(scene_id)
        assert run_after is not None
        assert run_after["vessels"] == vessels
        assert run_after["drift"] == drift

    def test_set_vessels_and_drift_raises_for_an_unregistered_spill(self) -> None:
        with SpillRegistry() as reg:
            with pytest.raises(RegistryError, match="no registered spill"):
                reg.set_vessels_and_drift("no-such-spill-id", [], {"forecast": [], "hindcast": []})
