"""File-based spill registry.

Scans ``data/processed/`` for pipeline outputs.  Two sources are supported:

1. **``<scene_id>/pipeline_result.json``** — the full Phase 3 combined output
   from ``scripts/run_pipeline.py``.  Contains spills, alert decisions, and
   scored vessels.
2. **``spills/<scene_id>.geojson``** — Phase 1-only FeatureCollection from
   ``scripts/run_detection.py``.  No alerts, no vessels.

Source 1 takes priority when both exist for the same scene.  The conversion
to the frontend ``PipelineRun`` contract happens here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def _processed_dir(settings: Settings) -> Path:
    return settings.paths.resolve(settings.paths.processed_dir)


def _spills_dir(settings: Settings) -> Path:
    d = _processed_dir(settings) / settings.characterization.spills_dirname
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_pipeline_results(settings: Settings) -> Dict[str, Path]:
    """Map scene_id → pipeline_result.json path."""
    root = _processed_dir(settings)
    results: Dict[str, Path] = {}
    for p in root.glob("*/pipeline_result.json"):
        scene_id = p.parent.name
        results[scene_id] = p
    return results


def _find_geojson_spills(settings: Settings) -> Dict[str, Path]:
    """Map scene_id → spills/<scene_id>.geojson path (Phase 1 only)."""
    spills_dir = _spills_dir(settings)
    results: Dict[str, Path] = {}
    for p in spills_dir.glob("*.geojson"):
        results[p.stem] = p
    return results


# --------------------------------------------------------------------------- #
# list_runs
# --------------------------------------------------------------------------- #
def list_runs(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    """Return one summary dict per processed scene."""
    settings = settings or get_settings()
    summaries: List[Dict[str, Any]] = []

    # Source 1: pipeline_result.json (Phase 3 output)
    for scene_id, path in sorted(_find_pipeline_results(settings).items()):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping corrupt file %s", path)
            continue
        summaries.append(_summary_from_pipeline(data, scene_id))

    # Source 2: spills/*.geojson (Phase 1 only, skip if pipeline exists)
    for scene_id, path in sorted(_find_geojson_spills(settings).items()):
        if scene_id in {s["scene_id"] for s in summaries}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping corrupt file %s", path)
            continue
        summaries.append(_summary_from_geojson(data, scene_id))

    return summaries


def _summary_from_pipeline(data: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    spills = data.get("spills", [])
    if spills:
        first = spills[0]
        spill_props = first.get("spill", {}).get("properties", {})
        alert = first.get("alert", {})
        total_area = sum(
            s.get("spill", {}).get("properties", {}).get("area_km2", 0)
            for s in spills
        )
        return {
            "scene_id": scene_id,
            "acquisition_timestamp": spill_props.get("acquisition_timestamp"),
            "area_km2": round(total_area, 4),
            "confidence": round(spill_props.get("mean_confidence") or 0.0, 4),
            "alert_status": alert.get("status", "none"),
        }
    return {
        "scene_id": scene_id,
        "acquisition_timestamp": data.get("generated_at"),
        "area_km2": 0.0,
        "confidence": 0.0,
        "alert_status": "none",
    }


def _summary_from_geojson(data: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    fc_props = data.get("properties", {})
    features = data.get("features", [])
    if features:
        first_props = features[0].get("properties", {})
        total_area = sum(f["properties"].get("area_km2", 0) for f in features)
        confidence = first_props.get("mean_confidence") or 0.0
    else:
        total_area = 0.0
        confidence = 0.0
    return {
        "scene_id": scene_id,
        "acquisition_timestamp": fc_props.get("acquisition_timestamp"),
        "area_km2": round(total_area, 4),
        "confidence": round(confidence, 4),
        "alert_status": "none",
    }


# --------------------------------------------------------------------------- #
# get_run
# --------------------------------------------------------------------------- #
def get_run(
    scene_id: str, settings: Optional[Settings] = None
) -> Optional[Dict[str, Any]]:
    """Return the full PipelineRun contract for one scene, or None."""
    settings = settings or get_settings()

    # Try pipeline_result.json first
    pipeline_path = _find_pipeline_results(settings).get(scene_id)
    if pipeline_path:
        try:
            data = json.loads(pipeline_path.read_text(encoding="utf-8"))
            return _pipeline_to_contract(data, scene_id)
        except Exception:
            logger.warning("Error reading %s", pipeline_path)

    # Fall back to Phase 1 GeoJSON
    geojson_path = _find_geojson_spills(settings).get(scene_id)
    if geojson_path:
        try:
            data = json.loads(geojson_path.read_text(encoding="utf-8"))
            return _geojson_to_contract(data, scene_id)
        except Exception:
            logger.warning("Error reading %s", geojson_path)

    return None


# --------------------------------------------------------------------------- #
# get_report_path
# --------------------------------------------------------------------------- #
def get_report_path(
    scene_id: str, settings: Optional[Settings] = None
) -> Optional[Path]:
    """Return path to the report file if it exists, else None."""
    settings = settings or get_settings()
    processed = _processed_dir(settings)
    candidates = [
        processed / scene_id / "pipeline_result.json",
        processed / scene_id / f"{scene_id}_report.json",
        processed / "reports" / f"{scene_id}_report.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- #
# Shape translation: pipeline_result.json -> frontend PipelineRun
# --------------------------------------------------------------------------- #
def _pipeline_to_contract(
    data: Dict[str, Any], scene_id: str
) -> Optional[Dict[str, Any]]:
    """Convert a Phase 3 pipeline_result.json to the frontend contract.

    The pipeline output has ``spills: [{spill, alert, vessels}]``.
    The frontend expects ``{spill, alert, vessels, drift}`` (one spill).
    We return the **first alerted spill** (or first spill if none alerted).
    """
    spills = data.get("spills", [])
    if not spills:
        return _empty_contract(scene_id, data.get("generated_at", ""))

    # Prefer an alerted spill
    target = spills[0]
    for s in spills:
        if s.get("alert", {}).get("alert"):
            target = s
            break

    return _spill_entry_to_contract(target, scene_id)


def _spill_entry_to_contract(
    entry: Dict[str, Any], scene_id: str
) -> Dict[str, Any]:
    """Convert one {spill, alert, vessels} entry to the frontend contract."""
    spill_feature = entry.get("spill", {})
    props = spill_feature.get("properties", {})
    geom = spill_feature.get("geometry", {})
    alert_raw = entry.get("alert", {})
    vessels_raw = entry.get("vessels", [])

    spill = {
        "scene_id": props.get("scene_id", scene_id),
        "acquisition_timestamp": props.get("acquisition_timestamp", ""),
        "confidence": props.get("mean_confidence") or 0.0,
        "area_km2": props.get("area_km2", 0.0),
        "centroid": {
            "lat": props.get("centroid_lat", 0.0),
            "lon": props.get("centroid_lon", 0.0),
        },
        "bbox": _bbox_to_object(props.get("bbox", [0, 0, 0, 0])),
        "polygon": geom,
        "major_axis_bearing": props.get("orientation_deg", 0.0),
        "elongation": props.get("elongation") or 1.0,
    }

    # Map backend alert status to frontend status
    backend_status = alert_raw.get("status", "none")
    status_map = {
        "active": "new",
        "possible": "possible",
        "rejected": "none",
    }
    frontend_status = status_map.get(backend_status, "none")

    # rules_fired = names of rules that did NOT pass
    rules_raw = alert_raw.get("rules", [])
    rules_fired = [r["name"] for r in rules_raw if not r.get("passed", True)]

    first_seen = props.get("acquisition_timestamp", "")
    last_updated = props.get("detected_at", datetime.now(timezone.utc).isoformat())
    if alert_raw.get("status"):
        last_updated = data_get(alert_raw, "last_updated", last_updated)

    alert = {
        "spill_id": alert_raw.get("spill_id") or props.get("spill_id", ""),
        "status": frontend_status,
        "rules_fired": rules_fired,
        "first_seen": first_seen,
        "last_updated": last_updated,
    }

    # Map backend vessels to frontend vessels
    vessels = []
    for v in vessels_raw:
        # Extract track coordinates as LineString
        track_coords = []
        if "track" in v and isinstance(v["track"], list):
            for point in v["track"]:
                if isinstance(point, dict) and "lon" in point and "lat" in point:
                    track_coords.append([float(point["lon"]), float(point["lat"])])
        
        vessels.append({
            "mmsi": str(v.get("mmsi", "")),
            "name": v.get("vessel_name", ""),
            "vessel_type": v.get("vessel_type", ""),
            "score": v.get("score", 0.0),
            "explanation": v.get("explanation", ""),
            "cpa_distance_km": v.get("cpa_distance_km", 0.0),
            "cpa_time": v.get("cpa_time", ""),
            "track": {"type": "LineString", "coordinates": track_coords},
        })

    return {
        "spill": spill,
        "alert": alert,
        "vessels": vessels,
        "drift": {"forecast": [], "hindcast": []},
    }


# --------------------------------------------------------------------------- #
# Shape translation: Phase 1 GeoJSON -> frontend PipelineRun
# --------------------------------------------------------------------------- #
def _geojson_to_contract(data: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    fc_props = data.get("properties", {})
    features = data.get("features", [])

    if not features:
        return _empty_contract(scene_id, fc_props.get("generated_at", ""))

    feat = features[0]
    props = feat.get("properties", {})
    geom = feat.get("geometry", {})

    spill = {
        "scene_id": props.get("scene_id", scene_id),
        "acquisition_timestamp": props.get(
            "acquisition_timestamp", fc_props.get("acquisition_timestamp", "")
        ),
        "confidence": props.get("mean_confidence") or 0.0,
        "area_km2": props.get("area_km2", 0.0),
        "centroid": {
            "lat": props.get("centroid_lat", 0.0),
            "lon": props.get("centroid_lon", 0.0),
        },
        "bbox": _bbox_to_object(props.get("bbox", [0, 0, 0, 0])),
        "polygon": geom,
        "major_axis_bearing": props.get("orientation_deg", 0.0),
        "elongation": props.get("elongation") or 1.0,
    }

    alert = {
        "spill_id": props.get("spill_id", ""),
        "status": "none",
        "rules_fired": [],
        "first_seen": props.get(
            "acquisition_timestamp", fc_props.get("acquisition_timestamp", "")
        ),
        "last_updated": props.get(
            "detected_at", datetime.now(timezone.utc).isoformat()
        ),
    }

    return {
        "spill": spill,
        "alert": alert,
        "vessels": [],
        "drift": {"forecast": [], "hindcast": []},
    }


def _empty_contract(scene_id: str, ts: str) -> Dict[str, Any]:
    return {
        "spill": {
            "scene_id": scene_id,
            "acquisition_timestamp": ts,
            "confidence": 0.0,
            "area_km2": 0.0,
            "centroid": {"lat": 0.0, "lon": 0.0},
            "bbox": {"minLon": 0.0, "minLat": 0.0, "maxLon": 0.0, "maxLat": 0.0},
            "polygon": {"type": "Polygon", "coordinates": []},
            "major_axis_bearing": 0.0,
            "elongation": 1.0,
        },
        "alert": {
            "spill_id": "",
            "status": "none",
            "rules_fired": [],
            "first_seen": ts,
            "last_updated": ts,
        },
        "vessels": [],
        "drift": {"forecast": [], "hindcast": []},
    }


def _bbox_to_object(bbox: list) -> Dict[str, float]:
    if len(bbox) >= 4:
        return {
            "minLon": bbox[0],
            "minLat": bbox[1],
            "maxLon": bbox[2],
            "maxLat": bbox[3],
        }
    return {"minLon": 0.0, "minLat": 0.0, "maxLon": 0.0, "maxLat": 0.0}


def data_get(d: dict, key: str, default: Any = None) -> Any:
    """Safe dict.get that works for nested access."""
    return d.get(key, default)
