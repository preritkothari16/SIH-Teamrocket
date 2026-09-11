-- Mirrors supabase/migrations/20260911094836_003_add_spills_indexes.sql —
-- see 003_enable_rls_on_spills.sql's own header for why this folder is a
-- numbered mirror rather than the source of truth.
--
-- Indexes the two columns src/api/registry.py actually filters/sorts by
-- (_get_run_postgres()'s WHERE scene_id=..., both functions' ORDER BY
-- last_updated DESC). IF NOT EXISTS makes this a no-op against the live DB,
-- which already has both.

CREATE INDEX IF NOT EXISTS spills_scene_id_idx ON spills (scene_id);
CREATE INDEX IF NOT EXISTS spills_last_updated_idx ON spills (last_updated DESC);
