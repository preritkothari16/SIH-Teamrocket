-- Mirrors supabase/migrations/20260910095143_enable_rls_on_spills.sql —
-- supabase/migrations/ is the source of truth for schema history (CLI-
-- tracked, drift-checked against the live DB via `supabase db push
-- --dry-run`); this folder is the numbered mirror scripts/apply_migrations.py
-- and the Dockerfile actually run at deploy time, since that path has no
-- supabase CLI in the image. Keep the two in sync by hand until it does.
--
-- Idempotent: re-enabling RLS on a table that already has it enabled is a
-- no-op, not an error — safe to run against the live DB, which already has
-- this applied.
--
-- RLS with no policies defined is deny-all via PostgREST/the anon key; this
-- app's own connection is always direct Postgres (DATABASE_URL), which never
-- goes through PostgREST/RLS at all — see src/db.py.

ALTER TABLE spills ENABLE ROW LEVEL SECURITY;
