"""One-off: finish S1D's processing from the already-computed mask.tif,
after the real run crashed on a transient Postgres timeout during
registration (inference itself - the 43-minute part - had already
succeeded and was saved to disk). Not a permanent script; deleted after use.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rasterio

from src.alerts.manager import process_spill
from src.alerts.registry import SpillRegistry
from src.api.registry import forecast_to_contract, vessels_to_contract
from src.characterization.spill_object import feature_collection, geographic_centroid, spills_from_blobs
from src.config import get_settings
from src.db import database_url
from src.detection.infer import NODATA_CLASS, load_tile_index, scene_grid_from_index, stitch_backscatter
from src.detection.lookalike_filter import filter_lookalikes
from src.ingestion.local_source import LocalSceneSource
from scripts.run_pipeline import _drift_forecast_payload

SCENE_ID = "S1D_IW_GRDH_1SDV_20260904T224720_20260904T224745_004434_008374_0887"
# From the earlier, already-validated stub-vs-TideTrace comparison run
# (scripts/compare_detectors.py) against this exact scene - same
# deterministic model + same input, confirmed matching by centroid against
# the crashed production run's own log (identical coordinates).
KNOWN_CONFIDENCE = {
    (103.68637915487204, 2.0405280208234355): 0.87,
    (103.71286934512649, 1.2620540194845842): 0.95,
}

settings = get_settings()
processed_root = settings.paths.resolve(settings.paths.processed_dir)
scene_dir = processed_root / SCENE_ID

frame = load_tile_index(scene_dir)
grid = scene_grid_from_index(frame)

mask_path = scene_dir / f"{SCENE_ID}_mask.tif"
with rasterio.open(mask_path) as ds:
    mask = ds.read(1)

backscatter = stitch_backscatter(scene_dir, frame=frame, grid=grid)
filtered = filter_lookalikes(mask, backscatter, confidence=None, settings=settings)
print(f"[1.5] {filtered.summary()}")

source = LocalSceneSource(settings=settings)
scene = source.get(SCENE_ID)

features = spills_from_blobs(
    filtered, transform=grid.transform, scene_id=SCENE_ID,
    acquisition_timestamp=scene.acquisition_time,
    crs=grid.crs.to_string() if grid.crs else "EPSG:4326",
    settings=settings,
)

for f in features:
    props = f["properties"]
    key = (round(props["centroid_lon"], 6), round(props["centroid_lat"], 6))
    matched = None
    for known_key, conf in KNOWN_CONFIDENCE.items():
        if abs(known_key[0] - key[0]) < 1e-3 and abs(known_key[1] - key[1]) < 1e-3:
            matched = conf
            break
    if matched is None:
        raise SystemExit(f"no known confidence match for spill at {key} - refusing to fabricate one")
    props["mean_confidence"] = matched
    print(f"  spill {props['spill_id']}: area={props['area_km2']}km2 confidence={matched}")

collection = feature_collection(
    features, scene_id=SCENE_ID, acquisition_timestamp=scene.acquisition_time,
    extra={
        "source": scene.source.value, "tiles": len(frame),
        "blobs_examined": len(filtered.blobs), "blobs_rejected": len(filtered.rejected),
        "stub_model": False,
        "covered_fraction": round(float((mask != NODATA_CLASS).mean()), 4),
    },
)
spills_target = processed_root / settings.characterization.spills_dirname / f"{SCENE_ID}.geojson"
spills_target.parent.mkdir(parents=True, exist_ok=True)
spills_target.write_text(json.dumps(collection, indent=2), encoding="utf-8")
print(f"[1.6] wrote {spills_target}")

registry_path = None
results = []
alerted_count = 0
for spill in features:
    with SpillRegistry(path=registry_path, settings=settings) as registry:
        decision = process_spill(spill, registry, settings=settings)
    if decision.alert:
        alerted_count += 1
    failed = [r.name for r in decision.rules if not r.passed]
    print(f"      {decision.spill_id or '(rejected)'}: {decision.status}" + (f" <- {', '.join(failed)}" if failed else ""))

    drift_forecast = _drift_forecast_payload(spill, settings)
    results.append({"spill": spill, "alert": decision.to_dict(), "vessels": [], "drift_forecast": drift_forecast})

    if decision.alert and decision.spill_id and database_url(settings):
        with SpillRegistry(path=registry_path, settings=settings) as registry:
            registry.set_vessels_and_drift(
                decision.spill_id, vessels_to_contract([]),
                {"forecast": forecast_to_contract(drift_forecast), "hindcast": []},
            )

print(f"      {alerted_count} of {len(features)} spill(s) alerted")

any_registered = any(r["alert"].get("spill_id") for r in results)
scene_centroid = None
if not any_registered:
    best = max(features, key=lambda f: (f.get("properties") or {}).get("mean_confidence") or 0.0)
    best_props = best["properties"]
    from shapely.geometry import shape as shapely_shape
    best_geometry = shapely_shape(best["geometry"])
    centroid = geographic_centroid(best_geometry)
    scene_centroid = {"lat": centroid.y, "lon": centroid.x}
    placeholder_id = best_props["spill_id"]
    if database_url(settings):
        with SpillRegistry(path=registry_path, settings=settings) as registry:
            kwargs = dict(
                geometry=best_geometry, centroid_lon=centroid.x, centroid_lat=centroid.y,
                area_km2=float(best_props["area_km2"]), seen_at=scene.acquisition_time,
                status="rejected", scene_id=SCENE_ID,
            )
            if registry.get(placeholder_id) is not None:
                registry.update(placeholder_id, **kwargs)
            else:
                registry.register(spill_id=placeholder_id, confidence=float(best_props["mean_confidence"]), rules_fired=[], **kwargs)
            print(f"      registered best-candidate placeholder {placeholder_id}")

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "scene_id": SCENE_ID,
    "stub_model": False,
    "provenance": {
        "sar_source": scene.source.value, "sar_scene_id": SCENE_ID,
        "ais_source_label": "unspecified", "wind_source": None, "current_source": None,
    },
    "spills": results,
    **({"scene_centroid": scene_centroid} if scene_centroid else {}),
}
target = scene_dir / "pipeline_result.json"
target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
print(f"[done] wrote {target}")
