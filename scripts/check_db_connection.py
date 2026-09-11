"""Step 10.3: the one thing to run first, before trusting anything else in
this phase — a tiny, standalone smoke test for the configured Postgres
connection.

Run it::

    .venv/Scripts/python.exe scripts/check_db_connection.py

Connects to the configured ``DATABASE_URL`` (a short-lived engine, not
:func:`src.db.get_engine`'s process-wide cached one — see below), runs
``SELECT 1`` and ``SELECT postgis_version()``, prints both, and exits
non-zero with a message naming the fix on any failure — it does not guess
at *which* fix; see the printed message.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/check_db_connection.py`
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine, text  # noqa: E402

from src.db import database_url  # noqa: E402

#: A short, explicit timeout — psycopg2/SQLAlchemy default to the OS's own
#: TCP timeout otherwise, which on a network that silently drops packets
#: (rather than refusing the connection) can hang for minutes instead of
#: failing with a clear message.
CONNECT_TIMEOUT_S = 10


def main() -> int:
    url = database_url()
    if not url:
        print(
            "DATABASE_URL is not set — copy .env.example to .env and fill it "
            "in, or set it directly in the environment.",
            file=sys.stderr,
        )
        return 1

    try:
        # A dedicated engine, not src.db.get_engine()'s process-wide cached
        # one — this is a one-off diagnostic connection, not something that
        # should join the shared pool other modules reuse.
        engine = create_engine(url, connect_args={"connect_timeout": CONNECT_TIMEOUT_S})
        with engine.connect() as conn:
            one = conn.execute(text("SELECT 1")).scalar()
            postgis_version = conn.execute(text("SELECT postgis_version()")).scalar()
    except Exception as exc:  # noqa: BLE001 - report the real driver error, don't hide it
        print(
            f"Could not connect: {type(exc).__name__}: {exc}\n\n"
            "If this looks like a timeout, DNS failure, or 'Network is "
            "unreachable': DATABASE_URL is probably pointing at Supabase's "
            "*direct* hostname (db.<ref>.supabase.co), which is IPv6-only. "
            "Use the Session Pooler connection string instead — Supabase "
            "dashboard -> Project Settings -> Database -> Connection "
            "pooling -> Session mode (port 5432, not Transaction mode's "
            "6543 — this project holds a persistent connection pool, not "
            "one connection per request).",
            file=sys.stderr,
        )
        return 1

    print(f"SELECT 1 -> {one}")
    print(f"postgis_version() -> {postgis_version}")
    print("Connection OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
