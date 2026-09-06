"""Durable store of registered spill events, across pipeline runs.

Sqlite3 (stdlib, no new dependency) rather than a parquet/JSON file: the
alert manager needs to **update an existing row in place** when a later
detection matches an already-registered event (drift, a refined area, an
escalated status - see ``dedup_rule`` in :mod:`src.alerts.manager`), which a
flat file would need rewriting wholesale to do safely, and a database can just
do with an ``UPDATE``.

One row per persistent **event**, not one per scene detection - the same
physical slick, re-detected in a later scene, updates its existing row rather
than adding a new one. ``spill_id`` is the row's identity across that history;
the first detection's value is kept even as later detections update the rest.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings
from src.ingestion.types import as_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS spills (
    spill_id      TEXT PRIMARY KEY,
    scene_id      TEXT,
    geometry      TEXT NOT NULL,
    centroid_lon  REAL NOT NULL,
    centroid_lat  REAL NOT NULL,
    area_km2      REAL NOT NULL,
    first_seen    TEXT NOT NULL,
    last_updated  TEXT NOT NULL,
    status        TEXT NOT NULL
)
"""


class RegistryError(RuntimeError):
    """The spill registry could not be read or written."""


@dataclass
class SpillRecord:
    """One registered event, as stored - see :mod:`src.alerts.manager` for
    how it gets created/updated."""

    spill_id: str
    scene_id: Optional[str]
    geometry: BaseGeometry
    centroid_lon: float
    centroid_lat: float
    area_km2: float
    first_seen: datetime
    last_updated: datetime
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spill_id": self.spill_id,
            "scene_id": self.scene_id,
            "geometry": mapping(self.geometry),
            "centroid_lon": self.centroid_lon,
            "centroid_lat": self.centroid_lat,
            "area_km2": self.area_km2,
            "first_seen": self.first_seen.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "status": self.status,
        }


def _row_to_record(row: sqlite3.Row) -> SpillRecord:
    return SpillRecord(
        spill_id=row["spill_id"],
        scene_id=row["scene_id"],
        geometry=shape(json.loads(row["geometry"])),
        centroid_lon=row["centroid_lon"],
        centroid_lat=row["centroid_lat"],
        area_km2=row["area_km2"],
        first_seen=as_utc(datetime.fromisoformat(row["first_seen"])),
        last_updated=as_utc(datetime.fromisoformat(row["last_updated"])),
        status=row["status"],
    )


class SpillRegistry:
    """Durable store of registered spill events, backed by a small sqlite3
    database (default: ``<paths.alerts_dir>/registry.sqlite3``).
    """

    def __init__(
        self, path: Optional[Path] = None, settings: Optional[Settings] = None
    ) -> None:
        settings = settings or get_settings()
        if path is None:
            path = settings.paths.resolve(settings.paths.alerts_dir) / "registry.sqlite3"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SpillRegistry":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def register(
        self,
        spill_id: str,
        geometry: BaseGeometry,
        centroid_lon: float,
        centroid_lat: float,
        area_km2: float,
        seen_at: datetime,
        status: str = "active",
        scene_id: Optional[str] = None,
    ) -> SpillRecord:
        """Insert a brand-new event. Raises if ``spill_id`` already exists -
        callers that mean "update if seen before" should check
        :meth:`get`/:meth:`recent` first (see ``evaluate_alert``'s dedup step).
        """
        seen_at = as_utc(seen_at)
        record = SpillRecord(
            spill_id=spill_id, scene_id=scene_id, geometry=geometry,
            centroid_lon=centroid_lon, centroid_lat=centroid_lat, area_km2=area_km2,
            first_seen=seen_at, last_updated=seen_at, status=status,
        )
        try:
            self._conn.execute(
                "INSERT INTO spills (spill_id, scene_id, geometry, centroid_lon, "
                "centroid_lat, area_km2, first_seen, last_updated, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.spill_id, record.scene_id, json.dumps(mapping(record.geometry)),
                    record.centroid_lon, record.centroid_lat, record.area_km2,
                    record.first_seen.isoformat(), record.last_updated.isoformat(),
                    record.status,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RegistryError(f"spill {spill_id!r} is already registered") from exc
        return record

    def update(
        self,
        spill_id: str,
        geometry: BaseGeometry,
        centroid_lon: float,
        centroid_lat: float,
        area_km2: float,
        seen_at: datetime,
        status: Optional[str] = None,
        scene_id: Optional[str] = None,
    ) -> SpillRecord:
        """Update an already-registered event with a newer detection -
        geometry/centroid/area track the slick's latest known extent,
        ``last_updated`` moves forward, but ``spill_id`` and ``first_seen``
        never change.
        """
        existing = self.get(spill_id)
        if existing is None:
            raise RegistryError(f"no registered spill {spill_id!r} to update")

        seen_at = as_utc(seen_at)
        record = SpillRecord(
            spill_id=spill_id,
            scene_id=scene_id if scene_id is not None else existing.scene_id,
            geometry=geometry, centroid_lon=centroid_lon, centroid_lat=centroid_lat,
            area_km2=area_km2, first_seen=existing.first_seen, last_updated=seen_at,
            status=status or existing.status,
        )
        self._conn.execute(
            "UPDATE spills SET scene_id=?, geometry=?, centroid_lon=?, centroid_lat=?, "
            "area_km2=?, last_updated=?, status=? WHERE spill_id=?",
            (
                record.scene_id, json.dumps(mapping(record.geometry)),
                record.centroid_lon, record.centroid_lat, record.area_km2,
                record.last_updated.isoformat(), record.status, spill_id,
            ),
        )
        return record

    def get(self, spill_id: str) -> Optional[SpillRecord]:
        row = self._conn.execute(
            "SELECT * FROM spills WHERE spill_id = ?", (spill_id,)
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def recent(self, since: datetime) -> List[SpillRecord]:
        """Every event last updated at or after ``since`` - the dedup
        lookback window (``alerts.dedup_window_days``).
        """
        rows = self._conn.execute(
            "SELECT * FROM spills WHERE last_updated >= ? ORDER BY last_updated DESC",
            (as_utc(since).isoformat(),),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def all(self) -> List[SpillRecord]:
        rows = self._conn.execute(
            "SELECT * FROM spills ORDER BY last_updated DESC"
        ).fetchall()
        return [_row_to_record(r) for r in rows]


__all__ = ["SpillRegistry", "SpillRecord", "RegistryError"]
