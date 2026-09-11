-- RECONSTRUCTED, not retrieved verbatim - see 20260910095046's own note.
-- Name is self-describing and matches the live advisory state confirmed
-- this session (RLS enabled on spills, no policies defined - deny-all via
-- PostgREST/anon key, which this app's direct-Postgres connection never
-- goes through anyway).

ALTER TABLE spills ENABLE ROW LEVEL SECURITY;
