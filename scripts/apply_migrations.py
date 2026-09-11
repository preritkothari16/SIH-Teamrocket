"""Apply every SQL file under migrations/, in filename order, against
DATABASE_URL - a one-off, run by hand (see migrations/README.md).

Exists because Render's free plan doesn't allow one-off Jobs ("new paid
services not allowed"), so this can't run *as* a Render Job - it runs
directly against whichever DATABASE_URL is set in the environment,
typically from a developer's own machine using the target database's
external connection string.

Run it::

    $env:DATABASE_URL = "postgresql://...external connection string..."
    .venv/Scripts/python.exe scripts/apply_migrations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/apply_migrations.py`
    sys.path.insert(0, str(REPO_ROOT))

import psycopg2  # noqa: E402

from src.db import database_url  # noqa: E402

MIGRATIONS_DIR = REPO_ROOT / "migrations"


def main() -> int:
    url = database_url()
    if not url:
        print(
            "DATABASE_URL is not set — set it to the target database's "
            "connection string first.",
            file=sys.stderr,
        )
        return 1

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        print(f"No .sql files found under {MIGRATIONS_DIR}", file=sys.stderr)
        return 1

    try:
        conn = psycopg2.connect(url, connect_timeout=10)
    except Exception as exc:  # noqa: BLE001 - report the real driver error
        print(f"Could not connect: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    conn.autocommit = True
    cur = conn.cursor()
    for path in sql_files:
        print(f"applying {path.name} ...")
        cur.execute(path.read_text(encoding="utf-8"))
        print(f"  ok")

    print(f"all {len(sql_files)} migration(s) applied.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
