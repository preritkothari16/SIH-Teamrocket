"""Step 10.4: one-off backfill of the one real, "real-scene verified" demo
run (``data/processed/demo_pipeline/pipeline_result.json``) into Postgres,
so ``GET /api/runs``/``GET /api/runs/demo_pipeline`` return real data with
``DATABASE_URL`` set — the same data the file-based path already serves
today.

Not wired into the normal pipeline, and not meant to be — a one-time
backfill for a fixture that predates the Postgres registry entirely (its
own scene was never re-run through it), run by hand, once, after
migrations/001 and migrations/002 (Step 10.1) are applied.

Run it::

    .venv/Scripts/python.exe scripts/backfill_postgres.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/backfill_postgres.py`
    sys.path.insert(0, str(REPO_ROOT))

from shapely.geometry import shape  # noqa: E402

from src.alerts.registry import SpillRegistry  # noqa: E402
from src.api.registry import forecast_to_contract, vessels_to_contract  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.db import database_url  # noqa: E402

DEMO_RESULT_PATH = REPO_ROOT / "data" / "processed" / "demo_pipeline" / "pipeline_result.json"


def main() -> int:
    settings = get_settings()
    if not database_url(settings):
        print(
            "DATABASE_URL not set — this backfill only makes sense against a "
            "real Postgres instance. Set it (see .env.example) and try again.",
            file=sys.stderr,
        )
        return 1

    if not DEMO_RESULT_PATH.is_file():
        print(f"{DEMO_RESULT_PATH} not found — nothing to backfill.", file=sys.stderr)
        return 1

    data = json.loads(DEMO_RESULT_PATH.read_text(encoding="utf-8"))
    spills = data.get("spills") or []
    if not spills:
        print(f"{DEMO_RESULT_PATH} has no spills — nothing to backfill.", file=sys.stderr)
        return 1

    entry = spills[0]
    spill_feature = entry["spill"]
    props = spill_feature.get("properties") or {}
    alert = entry.get("alert") or {}
    vessels_raw = entry.get("vessels") or []
    drift_forecast = entry.get("drift_forecast") or []

    spill_id = alert.get("spill_id") or props.get("spill_id")
    if not spill_id:
        print("no alert/spill id found in the fixture — can't register.", file=sys.stderr)
        return 1

    geometry = shape(spill_feature["geometry"])
    acquisition = props.get("acquisition_timestamp")
    seen_at = datetime.fromisoformat(acquisition) if acquisition else datetime.now(timezone.utc)
    rules_fired = [r["name"] for r in (alert.get("rules") or []) if not r.get("passed", True)]

    common_kwargs = dict(
        scene_id=props.get("scene_id"),
        confidence=props.get("mean_confidence"),
        bbox=props.get("bbox"),
        major_axis_bearing=props.get("orientation_deg"),
        elongation=props.get("elongation"),
        rules_fired=rules_fired,
    )

    with SpillRegistry(settings=settings) as registry:
        existing = registry.get(spill_id)
        if existing is None:
            registry.register(
                spill_id, geometry=geometry,
                centroid_lon=props.get("centroid_lon"), centroid_lat=props.get("centroid_lat"),
                area_km2=props.get("area_km2"), seen_at=seen_at,
                status=alert.get("status", "active"),
                **common_kwargs,
            )
            print(f"registered {spill_id!r} (scene {props.get('scene_id')!r})")
        else:
            registry.update(
                spill_id, geometry=geometry,
                centroid_lon=props.get("centroid_lon"), centroid_lat=props.get("centroid_lat"),
                area_km2=props.get("area_km2"), seen_at=seen_at,
                status=alert.get("status", "active"),
                **common_kwargs,
            )
            print(f"updated {spill_id!r} (already registered, scene {props.get('scene_id')!r})")

        registry.set_vessels_and_drift(
            spill_id,
            vessels_to_contract(vessels_raw),
            {"forecast": forecast_to_contract(drift_forecast), "hindcast": []},
        )
        print(f"attached {len(vessels_raw)} vessel(s) and {len(drift_forecast)} drift horizon(s)")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
