# Migrations

Run these by hand against Supabase, in order, once each — no migration
runner or ORM-driven migrations here (this repo has no `alembic`
dependency and shouldn't gain one for two files):

```bash
psql "$DATABASE_URL" -f migrations/001_create_spills.sql
psql "$DATABASE_URL" -f migrations/002_add_vessels_and_drift.sql
```

Use the **Session Pooler** connection string for `$DATABASE_URL`
(Supabase dashboard → Project Settings → Database → Connection pooling →
*Session* mode), not the direct `db.<ref>.supabase.co` host — see
`render.yaml`'s own note on why. Or paste each file's contents into
Supabase's SQL editor instead of using `psql` at all.

Both files are idempotent (`CREATE EXTENSION IF NOT EXISTS`,
`CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`) — safe to run
whether or not `spills` already exists, and safe to re-run.
