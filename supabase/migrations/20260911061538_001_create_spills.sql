-- Idempotent, run earlier this session via the Supabase MCP tools
-- (apply_migration) - a no-op against the table above since it already
-- existed, confirmed via list_tables. Column-for-column match to
-- src/db.py::SpillRow.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS spills (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    spill_id             TEXT NOT NULL UNIQUE,
    scene_id             TEXT NOT NULL,
    acquisition_timestamp TIMESTAMPTZ NOT NULL,
    confidence           DOUBLE PRECISION NOT NULL,
    area_km2             DOUBLE PRECISION NOT NULL,
    centroid             GEOMETRY(POINT, 4326) NOT NULL,
    bbox                 JSONB NOT NULL,
    polygon              GEOMETRY(GEOMETRY, 4326) NOT NULL,
    major_axis_bearing   DOUBLE PRECISION,
    elongation           DOUBLE PRECISION,
    status               TEXT NOT NULL,
    rules_fired          JSONB NOT NULL,
    first_seen           TIMESTAMPTZ NOT NULL,
    last_updated          TIMESTAMPTZ NOT NULL
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'spills_status_check'
    ) THEN
        ALTER TABLE spills
            ADD CONSTRAINT spills_status_check
            CHECK (status IN ('new', 'update', 'possible', 'none'));
    END IF;
END $$;
