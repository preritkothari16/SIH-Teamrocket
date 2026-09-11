-- Applied earlier this session via the Supabase MCP tools. NULL = "not
-- computed yet", distinct from an empty array/object. See
-- src/api/registry.py::_get_run_postgres().

ALTER TABLE spills ADD COLUMN IF NOT EXISTS vessels_json JSONB;
ALTER TABLE spills ADD COLUMN IF NOT EXISTS drift_json JSONB;
