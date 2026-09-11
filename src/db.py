"""Shared Postgres + PostGIS access: one engine, one table model.

``src/alerts/registry.py`` (writes, dedup) and ``src/api/registry.py``
(reads, serves the frontend) both talk to the same pre-existing ``spills``
table — this module is the one place that knows its shape, so the two never
drift into disagreeing column definitions.

The table already exists (created outside this repo, see the ``sar-oilspill``
Supabase project) — nothing here issues ``CREATE TABLE``. ``configs/
config.yaml`` never stores the connection string itself, only the name of
the env var that holds it (``credentials.database_url_env``, default
``DATABASE_URL``) — see :class:`src.config.CredentialsConfig`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from geoalchemy2 import Geometry
from sqlalchemy import Column, DateTime, Float, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import Settings, get_settings


class DatabaseError(RuntimeError):
    """No Postgres connection is configured, or it could not be reached."""


class Base(DeclarativeBase):
    pass


class SpillRow(Base):
    """One row per persistent alerted-spill event — mirrors
    :class:`src.alerts.registry.SpillRecord` (the dedup identity: spill_id,
    geometry, area, status, first_seen/last_updated) plus the fields only
    the frontend contract needs (confidence, bbox, major_axis_bearing,
    elongation, rules_fired, acquisition_timestamp) — see
    ``src/api/models.py``'s ``SpillObject``/``Alert``, which this table's
    columns were named to match directly, field for field.
    """

    __tablename__ = "spills"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    spill_id = Column(Text, unique=True, nullable=False)
    scene_id = Column(Text, nullable=False)
    acquisition_timestamp = Column(DateTime(timezone=True), nullable=False)
    confidence = Column(Float, nullable=False)
    area_km2 = Column(Float, nullable=False)
    centroid = Column(Geometry("POINT", srid=4326), nullable=False)
    bbox = Column(JSONB, nullable=False)
    polygon = Column(Geometry(srid=4326), nullable=False)  # Polygon or MultiPolygon
    major_axis_bearing = Column(Float, nullable=True)
    elongation = Column(Float, nullable=True)
    status = Column(Text, nullable=False)
    rules_fired = Column(JSONB, nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=False)
    last_updated = Column(DateTime(timezone=True), nullable=False)
    # Step 10.2 (migrations/002): NULL means "not computed yet", distinct
    # from an empty list/dict ("computed, none found") — see
    # src/api/registry.py::_get_run_postgres() for the read-side of that
    # distinction.
    vessels_json = Column(JSONB, nullable=True)
    drift_json = Column(JSONB, nullable=True)


#: The alert manager's raw status word (``src/alerts/manager.py::AlertDecision
#: .status``, always "active"/"possible"/"rejected") is not the frontend
#: contract's word ("new"/"update"/"possible"/"none" — "update" reserved for
#: a future re-detection case). Every writer and reader of the ``spills``
#: table's ``status`` column (its own CHECK constraint only allows the
#: frontend vocabulary) must go through this one mapping — two independent
#: copies of this dict already drifted apart once (src/api/registry.py's
#: list vs. detail endpoints, before both were made to call the same
#: function); don't reintroduce a third copy.
STATUS_MAP = {
    "active": "new",
    "possible": "possible",
    "rejected": "none",
}


def map_alert_status(backend_status: Optional[str]) -> str:
    """Alert manager's raw status word -> the spills table's/frontend's word."""
    return STATUS_MAP.get(backend_status, "none")


def database_url(settings: Optional[Settings] = None) -> Optional[str]:
    """The configured Postgres URL, or None if unset (SQLite/local-only)."""
    settings = settings or get_settings()
    return settings.credentials.resolve().get("database_url")


@lru_cache(maxsize=1)
def _cached_engine(url: str) -> Engine:
    # pool_pre_ping: Supabase (and most managed Postgres) close idle
    # connections server-side; without this a long-idle worker's first query
    # after a gap gets a stale-connection error instead of a transparent
    # reconnect.
    return create_engine(url, pool_pre_ping=True, future=True)


def get_engine(settings: Optional[Settings] = None) -> Engine:
    """The shared SQLAlchemy engine for the configured Postgres database.

    Raises :class:`DatabaseError` if no ``DATABASE_URL`` (or configured
    equivalent) is set — callers that have a non-Postgres fallback (e.g.
    :class:`src.alerts.registry.SpillRegistry`'s SQLite path) should check
    :func:`database_url` first rather than catching this.
    """
    url = database_url(settings)
    if not url:
        raise DatabaseError(
            "No database configured — set DATABASE_URL "
            "(see .env.example / credentials.database_url_env)"
        )
    return _cached_engine(url)


def get_sessionmaker(settings: Optional[Settings] = None) -> sessionmaker:
    return sessionmaker(bind=get_engine(settings), future=True, expire_on_commit=False)


__all__ = [
    "Base",
    "SpillRow",
    "DatabaseError",
    "database_url",
    "get_engine",
    "get_sessionmaker",
]
