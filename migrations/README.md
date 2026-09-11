# Migrations

Run these by hand against Supabase, in order, once each — no migration
runner or ORM-driven migrations here (this repo has no `alembic`
dependency and shouldn't gain one for four files), or let
`scripts/apply_migrations.py` do it (applies every `*.sql` here in filename
order — same thing this README describes by hand):

```bash
psql "$DATABASE_URL" -f migrations/001_create_spills.sql
psql "$DATABASE_URL" -f migrations/002_add_vessels_and_drift.sql
psql "$DATABASE_URL" -f migrations/003_enable_rls_on_spills.sql
psql "$DATABASE_URL" -f migrations/004_add_spills_indexes.sql
```

Use the **Session Pooler** connection string for `$DATABASE_URL`
(Supabase dashboard → Project Settings → Database → Connection pooling →
*Session* mode), not the direct `db.<ref>.supabase.co` host — see
`render.yaml`'s own note on why. Or paste each file's contents into
Supabase's SQL editor instead of using `psql` at all.

All four files are idempotent (`CREATE EXTENSION IF NOT EXISTS`,
`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`,
`ENABLE ROW LEVEL SECURITY` on a table that already has it,
`CREATE INDEX IF NOT EXISTS`) — safe to run whether or not `spills`
already exists, and safe to re-run.

**This folder is a numbered mirror, not the source of truth.**
`supabase/migrations/` (CLI-tracked, timestamp-named, checked for drift
against the live DB via `supabase db push --dry-run`) is canonical — author
new schema changes there first. This folder exists only because
`scripts/apply_migrations.py`/the Dockerfile deploy path has no `supabase`
CLI in the image and needs a plain SQL file it can glob and run in order.
When you add a migration to `supabase/migrations/`, add the matching
numbered file here by hand — that's exactly the sync step that was missed
for the indexes migration before this file existed.
