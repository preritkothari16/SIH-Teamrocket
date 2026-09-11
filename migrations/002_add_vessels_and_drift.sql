-- Step 10.2's storage: vessels/drift as plain JSONB columns on the
-- existing spills table, not new normalized tables — this table already
-- stores bbox/rules_fired the same way, and a full normalized vessels/
-- drift schema is more migration than a hackathon needs right now.
--
-- NULL (the default) means "not computed yet for this spill" — a real,
-- honest state distinct from an empty array/object ("computed, none
-- found"). src/api/registry.py::_get_run_postgres() only falls back to
-- the empty-array/object defaults when the column is actually NULL.

ALTER TABLE spills ADD COLUMN IF NOT EXISTS vessels_json JSONB;
ALTER TABLE spills ADD COLUMN IF NOT EXISTS drift_json JSONB;
