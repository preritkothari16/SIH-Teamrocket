-- RECONSTRUCTED, not retrieved verbatim: this predates this session and no
-- original source is accessible - rebuilt from its self-describing name
-- and the live schema already verified via list_tables (id, spill_id,
-- scene_id, acquisition_timestamp, confidence, area_km2, centroid, bbox,
-- polygon, major_axis_bearing, elongation, status + CHECK, rules_fired,
-- first_seen, last_updated - everything except vessels_json/drift_json,
-- which this session's own migrations added afterward).

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS spills (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spill_id              TEXT NOT NULL UNIQUE,
    scene_id              TEXT NOT NULL,
    acquisition_timestamp TIMESTAMPTZ NOT NULL,
    confidence            DOUBLE PRECISION NOT NULL,
    area_km2              DOUBLE PRECISION NOT NULL,
    centroid              GEOMETRY(POINT, 4326) NOT NULL,
    bbox                  JSONB NOT NULL,
    polygon               GEOMETRY(GEOMETRY, 4326) NOT NULL,
    major_axis_bearing    DOUBLE PRECISION,
    elongation            DOUBLE PRECISION,
    status                TEXT NOT NULL CHECK (status IN ('new', 'update', 'possible', 'none')),
    rules_fired           JSONB NOT NULL,
    first_seen            TIMESTAMPTZ NOT NULL,
    last_updated          TIMESTAMPTZ NOT NULL
);
