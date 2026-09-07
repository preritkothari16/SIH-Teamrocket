"""Tests for src.api (FastAPI backend).

Uses synthetic GeoJSON fixtures in a temporary spills directory rather than
real pipeline output, so the tests run fully offline in <2 s.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.registry import _geojson_to_contract
from src.config import Settings

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
SCENE_ID = "test_scene_001"
ACQUISITION_TS = "2024-04-10T14:20:00+00:00"

SAMPLE_FEATURE: Dict[str, Any] = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            [[-90.2, 23.8], [-89.7, 24.2], [-89.3, 24.0], [-89.8, 23.5], [-90.2, 23.8]]
        ],
    },
    "properties": {
        "spill_id": "test_scene_001_spill_001",
        "scene_id": SCENE_ID,
        "acquisition_timestamp": ACQUISITION_TS,
        "detected_at": "2024-04-10T14:25:00+00:00",
        "area_km2": 8.3,
        "perimeter_km": 15.2,
        "centroid_lon": -89.9999,
        "centroid_lat": 24.1111,
        "bbox": [-90.5, 23.5, -89.5, 24.5],
        "orientation_deg": 90.0,
        "elongation": 2.1,
        "major_axis_km": 12.5,
        "minor_axis_km": 5.95,
        "mean_confidence": 0.72,
        "estimated_volume_m3": 8300.0,
        "crs": "EPSG:4326",
    },
}


SAMPLE_COLLECTION: Dict[str, Any] = {
    "type": "FeatureCollection",
    "properties": {
        "scene_id": SCENE_ID,
        "acquisition_timestamp": ACQUISITION_TS,
        "spill_count": 1,
        "generated_at": "2024-04-10T14:30:00+00:00",
    },
    "features": [SAMPLE_FEATURE],
}

EMPTY_COLLECTION: Dict[str, Any] = {
    "type": "FeatureCollection",
    "properties": {
        "scene_id": "empty_scene",
        "acquisition_timestamp": ACQUISITION_TS,
        "spill_count": 0,
        "generated_at": "2024-04-10T14:30:00+00:00",
    },
    "features": [],
}


@pytest.fixture()
def spills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry at a temporary spills directory."""
    spills = tmp_path / "data" / "processed" / "spills"
    spills.mkdir(parents=True)
    monkeypatch.setenv("SAROIL_PATHS__PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    # Settings caches — invalidate so the new path is picked up.
    from src.api import registry
    original = registry.get_settings
    monkeypatch.setattr(registry, "get_settings", lambda: _make_settings(tmp_path))
    from src.api import main as main_mod
    monkeypatch.setattr(main_mod, "get_settings", lambda: _make_settings(tmp_path))
    return spills


def _make_settings(tmp_path: Path) -> Settings:
    """Build a Settings rooted at tmp_path."""
    return Settings(
        _env_file=None,
        paths={"processed_dir": str(tmp_path / "data" / "processed")},
    )


@pytest.fixture()
def populated_spills(spills_dir: Path) -> Path:
    """Write one sample GeoJSON into the spills dir."""
    (spills_dir / f"{SCENE_ID}.geojson").write_text(
        json.dumps(SAMPLE_COLLECTION, indent=2), encoding="utf-8"
    )
    return spills_dir


# --------------------------------------------------------------------------- #
# Shape translation tests
# --------------------------------------------------------------------------- #
class TestShapeTranslation:
    def test_bbox_converted(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        bbox = run["spill"]["bbox"]
        assert bbox == {"minLon": -90.5, "minLat": 23.5, "maxLon": -89.5, "maxLat": 24.5}

    def test_centroid_converted(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        centroid = run["spill"]["centroid"]
        assert centroid == {"lat": 24.1111, "lon": -89.9999}

    def test_orientation_maps_to_bearing(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        assert run["spill"]["major_axis_bearing"] == 90.0

    def test_confidence_from_mean_confidence(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        assert run["spill"]["confidence"] == 0.72

    def test_polygon_preserved(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        poly = run["spill"]["polygon"]
        assert poly["type"] == "Polygon"
        assert len(poly["coordinates"][0]) == 5

    def test_alert_defaults(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        alert = run["alert"]
        assert alert["status"] == "none"
        assert alert["rules_fired"] == []
        assert alert["spill_id"] == "test_scene_001_spill_001"

    def test_vessels_empty(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        assert run["vessels"] == []

    def test_drift_empty(self) -> None:
        run = _geojson_to_contract(SAMPLE_COLLECTION, SCENE_ID)
        assert run["drift"] == {"forecast": [], "hindcast": []}

    def test_empty_collection(self) -> None:
        run = _geojson_to_contract(EMPTY_COLLECTION, "empty_scene")
        assert run["spill"]["scene_id"] == "empty_scene"
        assert run["spill"]["area_km2"] == 0.0
        assert run["vessels"] == []


# --------------------------------------------------------------------------- #
# API endpoint tests
# --------------------------------------------------------------------------- #
class TestHealthEndpoint:
    def test_health(self) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestListRunsEndpoint:
    def test_empty_when_no_runs(self, spills_dir: Path) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_summary(self, populated_spills: Path) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        run = runs[0]
        assert run["scene_id"] == SCENE_ID
        assert run["area_km2"] == 8.3
        assert run["confidence"] == 0.72
        assert run["alert_status"] == "none"
        assert run["acquisition_timestamp"] == ACQUISITION_TS


class TestGetRunEndpoint:
    def test_404_when_missing(self, spills_dir: Path) -> None:
        resp = client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    def test_returns_full_contract(self, populated_spills: Path) -> None:
        resp = client.get(f"/api/runs/{SCENE_ID}")
        assert resp.status_code == 200
        run = resp.json()

        # Top-level keys
        assert set(run.keys()) == {"spill", "alert", "vessels", "drift"}

        # Spill fields
        spill = run["spill"]
        assert spill["scene_id"] == SCENE_ID
        assert spill["confidence"] == 0.72
        assert spill["area_km2"] == 8.3
        assert isinstance(spill["centroid"], dict)
        assert "lat" in spill["centroid"] and "lon" in spill["centroid"]
        assert isinstance(spill["bbox"], dict)
        assert all(
            k in spill["bbox"] for k in ("minLon", "minLat", "maxLon", "maxLat")
        )
        assert spill["polygon"]["type"] == "Polygon"
        assert isinstance(spill["major_axis_bearing"], (int, float))
        assert isinstance(spill["elongation"], (int, float))

        # Alert fields
        alert = run["alert"]
        assert alert["spill_id"] == "test_scene_001_spill_001"
        assert alert["status"] in ("new", "update", "possible", "none")
        assert isinstance(alert["rules_fired"], list)
        assert isinstance(alert["first_seen"], str)
        assert isinstance(alert["last_updated"], str)

        # Vessels — empty for Phase 1 only
        assert run["vessels"] == []

        # Drift — empty for Phase 1 only
        drift = run["drift"]
        assert "forecast" in drift
        assert "hindcast" in drift
        assert drift["forecast"] == []
        assert drift["hindcast"] == []


class TestGetReportEndpoint:
    def test_404_when_no_report(self, populated_spills: Path) -> None:
        resp = client.get(f"/api/runs/{SCENE_ID}/report")
        assert resp.status_code == 404

    def test_returns_report_when_exists(self, populated_spills: Path) -> None:
        report = {"scene_id": SCENE_ID, "summary": "test report"}
        # Write to the scene's own directory (pipeline_result.json convention)
        scene_dir = populated_spills.parent / SCENE_ID
        scene_dir.mkdir(parents=True, exist_ok=True)
        report_path = scene_dir / "pipeline_result.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")

        resp = client.get(f"/api/runs/{SCENE_ID}/report")
        assert resp.status_code == 200
        assert resp.json()["scene_id"] == SCENE_ID


class TestTriggerRunEndpoint:
    def test_400_without_scene_identifier(self) -> None:
        resp = client.post("/api/runs", json={})
        assert resp.status_code == 400
