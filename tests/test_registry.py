"""Direct unit tests for :mod:`src.alerts.registry` - SpillRegistry CRUD."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

from src.alerts.registry import RegistryError, SpillRegistry

_EARLY = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
_LATE = datetime(2025, 6, 2, 12, 0, tzinfo=timezone.utc)
_GEO = Polygon([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
_CENTROID = (0.0, 0.0)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "registry.sqlite3"


class TestRegisterAndGet:
    def test_round_trip(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            rec = reg.register(
                "SPILL-1", _GEO, _CENTROID[0], _CENTROID[1], 1.5, _EARLY, status="active",
                scene_id="s1",
            )
            assert rec.spill_id == "SPILL-1"
            assert rec.status == "active"
            assert rec.scene_id == "s1"

            got = reg.get("SPILL-1")
            assert got is not None
            assert got.spill_id == "SPILL-1"
            assert got.area_km2 == 1.5
            assert got.geometry.equals(_GEO)

    def test_get_missing_returns_none(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            assert reg.get("NOPE") is None

    def test_duplicate_raises(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            reg.register("X", _GEO, 0, 0, 1.0, _EARLY)
            with pytest.raises(RegistryError, match="already registered"):
                reg.register("X", _GEO, 0, 0, 1.0, _LATE)


class TestUpdate:
    def test_update_preserves_first_seen(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            reg.register("SPILL-2", _GEO, 0, 0, 1.0, _EARLY)
            updated = reg.update(
                "SPILL-2", _GEO, 0.1, 0.1, 2.0, _LATE, status="possible",
            )
            assert updated.first_seen == _EARLY
            assert updated.last_updated == _LATE
            assert updated.area_km2 == 2.0
            assert updated.status == "possible"
            assert updated.centroid_lon == 0.1

    def test_update_missing_raises(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            with pytest.raises(RegistryError, match="no registered spill"):
                reg.update("NOPE", _GEO, 0, 0, 1.0, _LATE)


class TestRecent:
    def test_recent_window(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            reg.register("OLD", _GEO, 0, 0, 1.0, _EARLY)
            reg.register("NEW", _GEO, 0, 0, 1.0, _LATE)

            recent = reg.recent(_LATE - timedelta(hours=1))
            ids = [r.spill_id for r in recent]
            assert "NEW" in ids
            assert "OLD" not in ids

    def test_recent_empty(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            assert reg.recent(_LATE) == []


class TestAll:
    def test_all_returns_desc(self, db: Path) -> None:
        with SpillRegistry(path=db) as reg:
            reg.register("A", _GEO, 0, 0, 1.0, _EARLY)
            reg.register("B", _GEO, 0, 0, 1.0, _LATE)
            all_recs = reg.all()
            assert len(all_recs) == 2
            assert all_recs[0].spill_id == "B"  # most recent first


class TestContextManager:
    def test_close_drops_connection(self, db: Path) -> None:
        reg = SpillRegistry(path=db)
        reg.register("X", _GEO, 0, 0, 1.0, _EARLY)
        reg.close()
        with pytest.raises(sqlite3.ProgrammingError):
            reg.get("X")
