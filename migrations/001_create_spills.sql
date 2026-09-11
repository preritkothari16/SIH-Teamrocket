-- Step 10.1: idempotent record of the spills table's intended shape.
--
-- Safe to run whether or not the table already exists — src/db.py's own
-- docstring says it was "created outside this repo," unverified from the
-- repo alone; this file is what makes that verifiable and reproducible.
-- Columns match src/db.py::SpillRow column-for-column.

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

-- src/db.py::STATUS_MAP is the frontend vocabulary this table's status
-- column stores — "new"/"possible" (an alerted spill's mapped status);
-- "update" is reserved for a future re-detection case, not produced yet;
-- "none" would mean "rejected", which src/alerts/manager.py never persists
-- at all. Add the constraint only if it isn't already there — a table that
-- really was created outside this repo may already have it (or a stricter/
-- looser one); don't fight an existing constraint by trying to add a
-- same-named one twice.
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
