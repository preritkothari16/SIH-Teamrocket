"""Spill registry backing the frontend API.

Three sources are supported, in priority order:

1. **Postgres ``spills`` table** (``src/db.py``) — used whenever
   ``DATABASE_URL`` is configured (see :func:`src.db.database_url`). This is
   the only source that survives a stateless deploy (Render, etc.): the
   other two read the local filesystem, which a fresh container doesn't
   have. One row per *alerted* spill (see ``src/alerts/registry.py`` — the
   only thing that ever writes to this table). Step 10.2 added
   ``vessels_json``/``drift_json`` columns (``migrations/002``); NULL there
   means "not computed yet for this spill" (falls back to an empty
   list/empty forecast+hindcast), populated once
   ``scripts/run_pipeline.py`` calls
   :meth:`src.alerts.registry.SpillRegistry.set_vessels_and_drift` after
   attribution/drift actually run for it.
2. **``<scene_id>/pipeline_result.json``** — the full Phase 3 combined
   output from ``scripts/run_pipeline.py``, read straight off
   ``data/processed/`` (local dev only). Contains spills, alert decisions,
   scored vessels, and the drift forecast.
3. **``spills/<scene_id>.geojson``** — Phase 1-only FeatureCollection from
   ``scripts/run_detection.py`` (also local-disk only). No alerts, no
   vessels, no drift.

When no ``DATABASE_URL`` is set, only 2 and 3 apply, exactly as before this
module knew about Postgres at all — nothing here changes local, offline
development. Source 2 takes priority over 3 when both exist for the same
scene. The conversion to the frontend ``PipelineRun`` contract happens here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Settings, get_settings
from src.db import database_url, map_alert_status

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

    if database_url(settings):
        return _list_runs_postgres(settings)

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


# --------------------------------------------------------------------------- #
# Postgres source (src/db.py's spills table)
# --------------------------------------------------------------------------- #
def _list_runs_postgres(settings: Settings) -> List[Dict[str, Any]]:
    from sqlalchemy import select

    from src.db import SpillRow, get_sessionmaker

    with get_sessionmaker(settings)() as session:
        rows = (
            session.execute(select(SpillRow).order_by(SpillRow.last_updated.desc()))
            .scalars()
            .all()
        )

    # One summary per scene_id — a scene with more than one alerted spill
    # keeps only its most-recently-updated one (rows already sorted above).
    by_scene: Dict[str, Any] = {}
    for row in rows:
        by_scene.setdefault(row.scene_id, row)

    return [
        {
            "scene_id": row.scene_id,
            "acquisition_timestamp": (
                row.acquisition_timestamp.isoformat() if row.acquisition_timestamp else None
            ),
            "area_km2": round(row.area_km2, 4),
            "confidence": round(row.confidence, 4),
            "alert_status": row.status,  # already the mapped word — see registry.py's writer
        }
        for row in by_scene.values()
    ]


def _get_run_postgres(scene_id: str, settings: Settings) -> Optional[Dict[str, Any]]:
    from geoalchemy2.shape import to_shape
    from shapely.geometry import mapping as shapely_mapping
    from sqlalchemy import select

    from src.db import SpillRow, get_sessionmaker

    with get_sessionmaker(settings)() as session:
        row = (
            session.execute(
                select(SpillRow)
                .where(SpillRow.scene_id == scene_id)
                .order_by(SpillRow.last_updated.desc())
            )
            .scalars()
            .first()
        )

    if row is None:
        return None

    centroid = to_shape(row.centroid)
    polygon = to_shape(row.polygon)

    return {
        "spill": {
            "scene_id": row.scene_id,
            "acquisition_timestamp": (
                row.acquisition_timestamp.isoformat() if row.acquisition_timestamp else ""
            ),
            "confidence": row.confidence,
            "area_km2": row.area_km2,
            "centroid": {"lat": centroid.y, "lon": centroid.x},
            "bbox": row.bbox,
            "polygon": shapely_mapping(polygon),
            "major_axis_bearing": row.major_axis_bearing or 0.0,
            "elongation": row.elongation or 1.0,
        },
        "alert": {
            "spill_id": row.spill_id,
            "status": row.status,
            "rules_fired": row.rules_fired or [],
            "first_seen": row.first_seen.isoformat() if row.first_seen else "",
            "last_updated": row.last_updated.isoformat() if row.last_updated else "",
        },
        # Step 10.2 (migrations/002): NULL means "not computed yet" (this
        # spill was never alerted through run_pipeline.py's full chain, or
        # predates these columns) — only then do the empty defaults apply.
        # An empty list/dict already stored (attribution/drift ran, found
        # nothing) is returned as-is, not collapsed into the same default.
        "vessels": row.vessels_json if row.vessels_json is not None else [],
        "drift": row.drift_json if row.drift_json is not None else {"forecast": [], "hindcast": []},
        "provenance": {
            "sar_source": None,
            "sar_scene_id": row.scene_id,
            "ais_source_label": None,
            "wind_source": None,
            "current_source": None,
        },
    }


#: Shared with src/alerts/registry.py's Postgres writer — see
#: src/db.py::map_alert_status for why this must be the *only* copy.
_map_alert_status = map_alert_status


def _select_target_spill(spills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the spill a run's top-level status/summary should reflect.

    Prefers the first *alerted* spill over ``spills[0]`` — an alert firing on
    any spill in the scene matters more than array order. Falls back to
    ``spills[0]`` when none are alerted. Shared by the list endpoint
    (:func:`_summary_from_pipeline`) and the detail endpoint
    (:func:`_pipeline_to_contract`) so they can't drift apart again.
    """
    target = spills[0]
    for s in spills:
        if s.get("alert", {}).get("alert"):
            return s
    return target


def _summary_from_pipeline(data: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    spills = data.get("spills", [])
    if spills:
        first = _select_target_spill(spills)
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
            "alert_status": _map_alert_status(alert.get("status")),
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

    if database_url(settings):
        return _get_run_postgres(scene_id, settings)

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
    """Return path to the Step 5.3 standalone HTML incident report
    (:mod:`src.output.report`), if ``run_pipeline.py --report`` has been run
    for this scene, else None.

    ``run_pipeline.py``'s default output name is ``report.html`` next to
    ``pipeline_result.json`` (``target.with_name("report.html")``); a custom
    ``--report-output`` can name it anything, so ``{scene_id}_report.html``
    and a shared ``reports/`` directory are also checked. This is never
    ``pipeline_result.json`` itself — that's the raw combined JSON, not a
    report, and is served by ``get_run()`` instead.
    """
    settings = settings or get_settings()
    processed = _processed_dir(settings)
    candidates = [
        processed / scene_id / "report.html",
        processed / scene_id / f"{scene_id}_report.html",
        processed / "reports" / f"{scene_id}_report.html",
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
        return _empty_contract(scene_id, data.get("generated_at", ""), data.get("scene_centroid"))

    target = _select_target_spill(spills)
    contract = _spill_entry_to_contract(target, scene_id)
    contract["provenance"] = data.get("provenance")
    return contract


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

    frontend_status = _map_alert_status(alert_raw.get("status"))

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

    return {
        "spill": spill,
        "alert": alert,
        "vessels": vessels_to_contract(vessels_raw),
        "drift": {"forecast": forecast_to_contract(entry.get("drift_forecast", [])), "hindcast": []},
    }


def vessels_to_contract(vessels_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``run_pipeline.py``'s raw candidate-vessel dicts (``vessel_name``,
    a ``track`` list of ``{lon, lat, timestamp}`` points, ...) to the
    frontend's ``Vessel`` shape (``name``, a ``track`` ``LineString``, ...).

    Used both by the file-based path above and by ``scripts/run_pipeline.py``
    itself before writing ``vessels_json`` to Postgres (Step 10.2) - one
    mapping, not two independent copies that could drift apart, same
    precedent as :func:`src.db.map_alert_status`.
    """
    vessels = []
    for v in vessels_raw:
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
    return vessels


def forecast_to_contract(drift_forecast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``run_pipeline.py``'s ``drift_forecast`` (``hours_elapsed``/``time``/``polygon``)
    to the frontend's ``ForecastEntry`` (``hours``/``time``/``polygon``)."""
    return [
        {
            "hours": round(f["hours_elapsed"]),
            "time": f["time"],
            "polygon": f["polygon"],
        }
        for f in drift_forecast
    ]


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
        # Phase 1 has no combined pipeline_result.json - no provenance object
        # was ever built for this run, only the sar_source this scene's own
        # FeatureCollection.properties["source"] already carries.
        "provenance": {
            "sar_source": fc_props.get("source"),
            "sar_scene_id": scene_id,
            "ais_source_label": None,
            "wind_source": None,
            "current_source": None,
        } if fc_props.get("source") else None,
    }


def _empty_contract(
    scene_id: str, ts: str, scene_centroid: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """A processed scene with zero characterized spills still needs to be
    shown at its own real location, not Null Island - ``scene_centroid``
    (run_pipeline.py's own real footprint centroid, set only when the spill
    list came back empty) takes priority over the (0, 0) fallback used when
    it isn't available at all (the Phase 1 GeoJSON path never carries one).
    """
    centroid = scene_centroid or {"lat": 0.0, "lon": 0.0}
    return {
        "spill": {
            "scene_id": scene_id,
            "acquisition_timestamp": ts,
            "confidence": 0.0,
            "area_km2": 0.0,
            "centroid": centroid,
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
        "provenance": None,
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


# --------------------------------------------------------------------------- #
# Report regeneration: PipelineRun contract -> run_pipeline.py "result" shape
# --------------------------------------------------------------------------- #
def contract_to_pipeline_result(contract: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    """Adapt one ``get_run()``-shaped ``PipelineRun`` contract back into the
    ``{scene_id, generated_at, spills: [...]}`` shape
    ``src/output/report.py::build_report_html()`` (and the ``build_map()``
    it calls) expect.

    This is what lets ``GET /api/runs/{id}/report`` regenerate a report on
    request instead of requiring ``scripts/run_pipeline.py --report``'s
    original file to still be sitting on local disk - the only thing that
    survives a stateless Render deploy is whatever ``get_run()`` itself can
    already produce (the Postgres ``spills`` table included).

    Lossy for a Postgres-backed contract, by that table's own design (see
    this module's docstring): ``rules_fired`` there is only the *failed*
    rule *names*, not each rule's own pass/fail reason string, and
    ``vessels``/``drift`` are always empty - the regenerated report shows
    exactly what the contract has, same as the dashboard/map already do.
    """
    spill = contract["spill"]
    alert = contract["alert"]
    vessels = contract.get("vessels") or []
    drift = contract.get("drift") or {}

    feature = {
        "type": "Feature",
        "geometry": spill.get("polygon"),
        "properties": {
            "spill_id": alert.get("spill_id") or scene_id,
            "scene_id": spill.get("scene_id", scene_id),
            "acquisition_timestamp": spill.get("acquisition_timestamp"),
            "centroid_lon": (spill.get("centroid") or {}).get("lon"),
            "centroid_lat": (spill.get("centroid") or {}).get("lat"),
            "area_km2": spill.get("area_km2"),
            "mean_confidence": spill.get("confidence"),
            "elongation": spill.get("elongation"),
            "orientation_deg": spill.get("major_axis_bearing"),
        },
    }
    alert_out = {
        "status": alert.get("status"),
        # Each failed rule's own reason didn't survive into the contract -
        # see this function's docstring - so only the name is known here.
        "rules": [{"name": name, "passed": False, "reason": ""} for name in (alert.get("rules_fired") or [])],
    }
    vessels_out = [
        {
            "mmsi": v.get("mmsi"),
            "vessel_name": v.get("name"),
            "vessel_type": v.get("vessel_type"),
            "score": v.get("score"),
            "cpa_distance_km": v.get("cpa_distance_km"),
            "explanation": v.get("explanation"),
        }
        for v in vessels
    ]
    forecast_out = [
        {"hours_elapsed": f.get("hours"), "time": f.get("time")}
        for f in (drift.get("forecast") or [])
    ]

    return {
        "scene_id": scene_id,
        "generated_at": alert.get("last_updated") or spill.get("acquisition_timestamp"),
        "spills": [{
            "spill": feature,
            "alert": alert_out,
            "vessels": vessels_out,
            "drift_forecast": forecast_out,
        }],
    }
