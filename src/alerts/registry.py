"""Durable store of registered spill events, across pipeline runs.

Two backends, chosen automatically:

- **Postgres + PostGIS** (SQLAlchemy + GeoAlchemy2, ``src/db.py``'s
  pre-existing ``spills`` table) whenever ``DATABASE_URL`` is configured
  (see :func:`src.db.database_url`) and no explicit ``path=`` is given —
  this is what a deployed backend uses, and what survives across container
  restarts.
- **sqlite3** (stdlib, no dependency) otherwise — an explicit ``path=``
  always forces this backend regardless of ``DATABASE_URL``, which is how
  every existing test gets deterministic, network-free, per-test isolation
  (each passes its own ``tmp_path / "registry.sqlite3"``).

Either way, the alert manager needs to **update an existing row in place**
when a later detection matches an already-registered event (drift, a refined
area, an escalated status - see ``dedup_rule`` in :mod:`src.alerts.manager`),
which a flat file would need rewriting wholesale to do safely, and a database
can just do with an ``UPDATE``.

One row per persistent **event**, not one per scene detection - the same
physical slick, re-detected in a later scene, updates its existing row rather
than adding a new one. ``spill_id`` is the row's identity across that history;
the first detection's value is kept even as later detections update the rest.

The Postgres ``spills`` table's columns were named to match the frontend
contract (``src/api/models.py``), not this module's narrower dedup identity
- it needs ``confidence``/``bbox``/``major_axis_bearing``/``elongation``/
``rules_fired`` that :class:`SpillRecord` never tracked. :meth:`register`
and :meth:`update` take these as new **optional** keyword arguments so the
sqlite-backed call sites (and every existing test) don't have to change;
writing to Postgres without ``confidence`` (the one NOT NULL column with no
sensible default) raises :class:`RegistryError` naming it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings
from src.db import database_url, get_sessionmaker, map_alert_status
from src.ingestion.types import as_utc

#: Reverse of src.db.STATUS_MAP — safe because this table only ever stores
#: an *alerted* spill's status, so the stored word is always "new" or
#: "possible", never "none" (that would mean "rejected", which is never
#: persisted at all - see src/alerts/manager.py::evaluate_alert).
_UNMAP_STATUS = {"new": "active", "possible": "possible"}

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


def _to_geoalchemy_point(lon: float, lat: float):
    from geoalchemy2.shape import from_shape

    return from_shape(ShapelyPoint(lon, lat), srid=4326)


def _to_geoalchemy_geometry(geometry: BaseGeometry):
    from geoalchemy2.shape import from_shape

    return from_shape(geometry, srid=4326)


def _bbox_dict(bbox: Optional[Sequence[float]], geometry: BaseGeometry) -> Dict[str, float]:
    """``{minLon, minLat, maxLon, maxLat}`` — the frontend's ``BBox`` shape
    (``src/api/models.py``), so the API's Postgres reader doesn't need a
    separate translation step. Falls back to the geometry's own bounds when
    the caller doesn't supply one explicitly.
    """
    min_lon, min_lat, max_lon, max_lat = bbox if bbox is not None else geometry.bounds
    return {"minLon": min_lon, "minLat": min_lat, "maxLon": max_lon, "maxLat": max_lat}


def _pg_row_to_record(row: Any) -> SpillRecord:
    from geoalchemy2.shape import to_shape

    centroid = to_shape(row.centroid)
    return SpillRecord(
        spill_id=row.spill_id,
        scene_id=row.scene_id,
        geometry=to_shape(row.polygon),
        centroid_lon=centroid.x,
        centroid_lat=centroid.y,
        area_km2=row.area_km2,
        first_seen=as_utc(row.first_seen),
        last_updated=as_utc(row.last_updated),
        status=_UNMAP_STATUS.get(row.status, row.status),
    )


class SpillRegistry:
    """Durable store of registered spill events, backed by a small sqlite3
    database (default: ``<paths.alerts_dir>/registry.sqlite3``).
    """

    def __init__(
        self, path: Optional[Path] = None, settings: Optional[Settings] = None
    ) -> None:
        settings = settings or get_settings()
        url = database_url(settings) if path is None else None

        if url:
            self._backend = "postgres"
            self._Session = get_sessionmaker(settings)
            return

        self._backend = "sqlite"
        if path is None:
            path = settings.paths.resolve(settings.paths.alerts_dir) / "registry.sqlite3"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)

    def close(self) -> None:
        # Postgres: sessions are opened/closed per call already (a `with
        # self._Session() as session` block each time) - the underlying
        # engine is a process-wide cached pool (src/db.py), not owned by any
        # one SpillRegistry instance, so there's nothing to close here.
        if self._backend == "sqlite":
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
        confidence: Optional[float] = None,
        bbox: Optional[Sequence[float]] = None,
        major_axis_bearing: Optional[float] = None,
        elongation: Optional[float] = None,
        rules_fired: Optional[List[str]] = None,
    ) -> SpillRecord:
        """Insert a brand-new event. Raises if ``spill_id`` already exists -
        callers that mean "update if seen before" should check
        :meth:`get`/:meth:`recent` first (see ``evaluate_alert``'s dedup step).

        ``confidence``/``bbox``/``major_axis_bearing``/``elongation``/
        ``rules_fired`` are ignored on the sqlite backend (no such columns);
        the Postgres backend requires ``confidence`` (its one NOT NULL
        column with no sensible default) and derives ``bbox`` from
        ``geometry.bounds`` when not given.
        """
        seen_at = as_utc(seen_at)
        record = SpillRecord(
            spill_id=spill_id, scene_id=scene_id, geometry=geometry,
            centroid_lon=centroid_lon, centroid_lat=centroid_lat, area_km2=area_km2,
            first_seen=seen_at, last_updated=seen_at, status=status,
        )

        if self._backend == "postgres":
            self._register_postgres(record, confidence, bbox, major_axis_bearing, elongation, rules_fired)
            return record

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
        confidence: Optional[float] = None,
        bbox: Optional[Sequence[float]] = None,
        major_axis_bearing: Optional[float] = None,
        elongation: Optional[float] = None,
        rules_fired: Optional[List[str]] = None,
    ) -> SpillRecord:
        """Update an already-registered event with a newer detection -
        geometry/centroid/area track the slick's latest known extent,
        ``last_updated`` moves forward, but ``spill_id`` and ``first_seen``
        never change. See :meth:`register` for the Postgres-only kwargs.
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

        if self._backend == "postgres":
            self._update_postgres(record, confidence, bbox, major_axis_bearing, elongation, rules_fired)
            return record

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
        if self._backend == "postgres":
            return self._get_postgres(spill_id)
        row = self._conn.execute(
            "SELECT * FROM spills WHERE spill_id = ?", (spill_id,)
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def recent(self, since: datetime) -> List[SpillRecord]:
        """Every event last updated at or after ``since`` - the dedup
        lookback window (``alerts.dedup_window_days``).
        """
        if self._backend == "postgres":
            return self._query_postgres(since=as_utc(since))
        rows = self._conn.execute(
            "SELECT * FROM spills WHERE last_updated >= ? ORDER BY last_updated DESC",
            (as_utc(since).isoformat(),),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def all(self) -> List[SpillRecord]:
        if self._backend == "postgres":
            return self._query_postgres(since=None)
        rows = self._conn.execute(
            "SELECT * FROM spills ORDER BY last_updated DESC"
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    # ----------------------------------------------------------------- #
    # Postgres backend
    # ----------------------------------------------------------------- #
    def _register_postgres(
        self,
        record: SpillRecord,
        confidence: Optional[float],
        bbox: Optional[Sequence[float]],
        major_axis_bearing: Optional[float],
        elongation: Optional[float],
        rules_fired: Optional[List[str]],
    ) -> None:
        from sqlalchemy.exc import IntegrityError

        from src.db import SpillRow

        if confidence is None:
            raise RegistryError(
                f"confidence is required to register {record.spill_id!r} in Postgres "
                "(spills.confidence is NOT NULL)"
            )

        row = SpillRow(
            spill_id=record.spill_id,
            scene_id=record.scene_id or "",
            acquisition_timestamp=record.first_seen,
            confidence=confidence,
            area_km2=record.area_km2,
            centroid=_to_geoalchemy_point(record.centroid_lon, record.centroid_lat),
            bbox=_bbox_dict(bbox, record.geometry),
            polygon=_to_geoalchemy_geometry(record.geometry),
            major_axis_bearing=major_axis_bearing,
            elongation=elongation,
            status=map_alert_status(record.status),
            rules_fired=rules_fired or [],
            first_seen=record.first_seen,
            last_updated=record.last_updated,
        )
        try:
            with self._Session() as session:
                session.add(row)
                session.commit()
        except IntegrityError as exc:
            raise RegistryError(f"spill {record.spill_id!r} is already registered") from exc

    def _update_postgres(
        self,
        record: SpillRecord,
        confidence: Optional[float],
        bbox: Optional[Sequence[float]],
        major_axis_bearing: Optional[float],
        elongation: Optional[float],
        rules_fired: Optional[List[str]],
    ) -> None:
        from sqlalchemy import select

        from src.db import SpillRow

        with self._Session() as session:
            row = (
                session.execute(select(SpillRow).where(SpillRow.spill_id == record.spill_id))
                .scalars()
                .first()
            )
            if row is None:
                raise RegistryError(f"no registered spill {record.spill_id!r} to update")

            row.scene_id = record.scene_id or row.scene_id
            row.area_km2 = record.area_km2
            row.centroid = _to_geoalchemy_point(record.centroid_lon, record.centroid_lat)
            row.polygon = _to_geoalchemy_geometry(record.geometry)
            row.status = map_alert_status(record.status)
            row.last_updated = record.last_updated
            row.acquisition_timestamp = record.last_updated
            if confidence is not None:
                row.confidence = confidence
            if bbox is not None:
                row.bbox = _bbox_dict(bbox, record.geometry)
            if major_axis_bearing is not None:
                row.major_axis_bearing = major_axis_bearing
            if elongation is not None:
                row.elongation = elongation
            if rules_fired is not None:
                row.rules_fired = rules_fired
            session.commit()

    def _get_postgres(self, spill_id: str) -> Optional[SpillRecord]:
        from sqlalchemy import select

        from src.db import SpillRow

        with self._Session() as session:
            row = (
                session.execute(select(SpillRow).where(SpillRow.spill_id == spill_id))
                .scalars()
                .first()
            )
        return _pg_row_to_record(row) if row is not None else None

    def _query_postgres(self, since: Optional[datetime]) -> List[SpillRecord]:
        from sqlalchemy import select

        from src.db import SpillRow

        stmt = select(SpillRow).order_by(SpillRow.last_updated.desc())
        if since is not None:
            stmt = stmt.where(SpillRow.last_updated >= since)
        with self._Session() as session:
            rows = session.execute(stmt).scalars().all()
        return [_pg_row_to_record(r) for r in rows]

    # ----------------------------------------------------------------- #
    # vessels/drift (Step 10.2) — Postgres-only, same precedent as
    # confidence/bbox/major_axis_bearing/elongation/rules_fired already
    # being Postgres-only kwargs on register()/update(): the sqlite schema
    # (SCHEMA above) has no such columns, and every existing sqlite-backed
    # call site/test predates this and shouldn't have to change.
    # ----------------------------------------------------------------- #
    def set_vessels_and_drift(
        self, spill_id: str, vessels: List[Dict[str, Any]], drift: Dict[str, Any],
    ) -> None:
        """Attach the ranked vessel list and drift forecast/hindcast to an
        already-registered event, once attribution/drift have actually run
        for it (``scripts/run_pipeline.py`` — after ``register()``/
        ``update()``, which happen during alert evaluation, before either
        of those exist yet).

        A documented no-op on the sqlite backend — there is nothing there
        to write these into, and this project's whole test suite staying
        offline by default depends on that being fine, not an error.
        """
        if self._backend != "postgres":
            return

        from sqlalchemy import select

        from src.db import SpillRow

        with self._Session() as session:
            row = (
                session.execute(select(SpillRow).where(SpillRow.spill_id == spill_id))
                .scalars()
                .first()
            )
            if row is None:
                raise RegistryError(f"no registered spill {spill_id!r} to attach vessels/drift to")
            row.vessels_json = vessels
            row.drift_json = drift
            session.commit()


__all__ = ["SpillRegistry", "SpillRecord", "RegistryError"]
