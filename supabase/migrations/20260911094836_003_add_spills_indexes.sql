-- Applied earlier this session via the Supabase MCP tools. Indexes the two
-- columns src/api/registry.py actually filters/sorts by
-- (_get_run_postgres()'s WHERE scene_id=..., both functions' ORDER BY
-- last_updated DESC).

CREATE INDEX IF NOT EXISTS spills_scene_id_idx ON spills (scene_id);
CREATE INDEX IF NOT EXISTS spills_last_updated_idx ON spills (last_updated DESC);
