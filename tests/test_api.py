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
from src.api.registry import _geojson_to_contract, _pipeline_to_contract, _select_target_spill
from src.config import Settings

client = TestClient(app)

# Auth (src/api/main.py::require_api_key) gates only the two POST endpoints.
# Every test in this module gets a known API_KEY via the autouse fixture
# below, so pre-existing POST calls keep working once they pass this header;
# TestApiKeyAuth is what actually exercises the gate itself.
TEST_API_KEY = "test-api-key-for-suite"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture(autouse=True)
def _api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", TEST_API_KEY)


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
    # These tests exercise the local-file source; a real DATABASE_URL in the
    # environment (a developer's own .env) would otherwise take priority —
    # see src/api/registry.py's Postgres-first branch — and these fixtures
    # would silently query an empty/unrelated real table instead.
    monkeypatch.delenv("DATABASE_URL", raising=False)
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
# Shape translation: Phase 3 pipeline_result.json -> frontend PipelineRun
# --------------------------------------------------------------------------- #
SAMPLE_PIPELINE_RESULT: Dict[str, Any] = {
    "generated_at": "2024-04-10T14:30:00+00:00",
    "scene_id": SCENE_ID,
    "spills": [
        {
            "spill": SAMPLE_FEATURE,
            "alert": {
                "spill_id": "test_scene_001_spill_001",
                "alert": True,
                "status": "active",
                "rules": [{"name": "confidence_rule", "passed": True}],
                "last_updated": "2024-04-10T14:40:00+00:00",
            },
            "vessels": [
                {
                    "mmsi": "367123452",
                    "vessel_name": "TANKER Gamma",
                    "vessel_type": "tanker",
                    "score": 0.88,
                    "explanation": "Vessel detected in spill corridor",
                    "cpa_distance_km": 1.5,
                    "cpa_time": "2024-04-10T14:35:00+00:00",
                    "track": [{"lon": -90.0, "lat": 24.0}, {"lon": -89.9, "lat": 24.1}],
                }
            ],
            "drift_forecast": [
                {"hours_elapsed": 6.0, "time": "2024-04-10T20:20:00+00:00", "polygon": {"type": "Polygon", "coordinates": []}},
                {"hours_elapsed": 24.0, "time": "2024-04-11T14:20:00+00:00", "polygon": {"type": "Polygon", "coordinates": []}},
            ],
        }
    ],
}


class TestPipelineResultTranslation:
    def test_drift_forecast_mapped(self) -> None:
        run = _pipeline_to_contract(SAMPLE_PIPELINE_RESULT, SCENE_ID)
        forecast = run["drift"]["forecast"]
        assert [f["hours"] for f in forecast] == [6, 24]
        assert forecast[0]["time"] == "2024-04-10T20:20:00+00:00"
        assert run["drift"]["hindcast"] == []

    def test_vessels_mapped(self) -> None:
        run = _pipeline_to_contract(SAMPLE_PIPELINE_RESULT, SCENE_ID)
        vessel = run["vessels"][0]
        assert vessel["mmsi"] == "367123452"
        assert vessel["name"] == "TANKER Gamma"
        assert vessel["track"]["coordinates"] == [[-90.0, 24.0], [-89.9, 24.1]]

    def test_alert_status_mapped(self) -> None:
        run = _pipeline_to_contract(SAMPLE_PIPELINE_RESULT, SCENE_ID)
        assert run["alert"]["status"] == "new"


# --------------------------------------------------------------------------- #
# Regression: list endpoint and detail endpoint must agree on a run's status
# --------------------------------------------------------------------------- #
MULTI_SPILL_SCENE_ID = "multi_spill_scene"


def _spill_entry(spill_id: str, alerted: bool, status: str) -> Dict[str, Any]:
    feature = json.loads(json.dumps(SAMPLE_FEATURE))
    feature["properties"]["spill_id"] = spill_id
    return {
        "spill": feature,
        "alert": {
            "spill_id": spill_id if alerted else None,
            "alert": alerted,
            "status": status,
            "rules": [],
            "last_updated": "2024-04-10T14:40:00+00:00",
        },
        "vessels": [],
        "drift_forecast": [],
    }


# spills[0] is NOT alerted; spills[1] is — the exact shape that let the list
# endpoint (spills[0]-only) and detail endpoint (first-alerted) disagree.
SAMPLE_MULTI_SPILL_RESULT: Dict[str, Any] = {
    "generated_at": "2024-04-10T14:30:00+00:00",
    "scene_id": MULTI_SPILL_SCENE_ID,
    "spills": [
        _spill_entry(f"{MULTI_SPILL_SCENE_ID}_spill_001", alerted=False, status="rejected"),
        _spill_entry(f"{MULTI_SPILL_SCENE_ID}_spill_002", alerted=True, status="possible"),
    ],
}


@pytest.fixture()
def populated_multi_spill_pipeline(spills_dir: Path, tmp_path: Path) -> Path:
    """Write a pipeline_result.json for MULTI_SPILL_SCENE_ID under the same
    tmp_path the ``spills_dir`` fixture already pointed settings at."""
    run_dir = tmp_path / "data" / "processed" / MULTI_SPILL_SCENE_ID
    run_dir.mkdir(parents=True)
    result_path = run_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(SAMPLE_MULTI_SPILL_RESULT), encoding="utf-8")
    return result_path


class TestStatusSelectionConsistency:
    def test_select_target_spill_prefers_alerted(self) -> None:
        target = _select_target_spill(SAMPLE_MULTI_SPILL_RESULT["spills"])
        assert target["alert"]["status"] == "possible"

    def test_list_and_detail_agree_on_status(
        self, populated_multi_spill_pipeline: Path
    ) -> None:
        list_resp = client.get("/api/runs")
        assert list_resp.status_code == 200
        runs = {r["scene_id"]: r for r in list_resp.json()}
        assert MULTI_SPILL_SCENE_ID in runs
        list_status = runs[MULTI_SPILL_SCENE_ID]["alert_status"]

        detail_resp = client.get(f"/api/runs/{MULTI_SPILL_SCENE_ID}")
        assert detail_resp.status_code == 200
        detail_status = detail_resp.json()["alert"]["status"]

        assert list_status == detail_status == "possible"


# --------------------------------------------------------------------------- #
# Regression: list and detail must agree for every raw status word, not just
# "possible" (which happened to match by coincidence — "possible" maps to
# itself). Mirrors src/alerts/manager.py's real (alert, status) pairings:
# "rejected" only ever pairs with alert=False, "active"/"possible" with True.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scene_id, alerted, raw_status, expected_mapped",
    [
        ("status_scene_active", True, "active", "new"),
        ("status_scene_possible", True, "possible", "possible"),
        ("status_scene_rejected", False, "rejected", "none"),
    ],
)
class TestStatusWordConsistency:
    def test_list_and_detail_agree(
        self,
        spills_dir: Path,
        tmp_path: Path,
        scene_id: str,
        alerted: bool,
        raw_status: str,
        expected_mapped: str,
    ) -> None:
        result = {
            "generated_at": "2024-04-10T14:30:00+00:00",
            "scene_id": scene_id,
            "spills": [_spill_entry(f"{scene_id}_spill_001", alerted=alerted, status=raw_status)],
        }
        run_dir = tmp_path / "data" / "processed" / scene_id
        run_dir.mkdir(parents=True)
        (run_dir / "pipeline_result.json").write_text(json.dumps(result), encoding="utf-8")

        list_resp = client.get("/api/runs")
        assert list_resp.status_code == 200
        runs = {r["scene_id"]: r for r in list_resp.json()}
        assert scene_id in runs
        list_status = runs[scene_id]["alert_status"]

        detail_resp = client.get(f"/api/runs/{scene_id}")
        assert detail_resp.status_code == 200
        detail_status = detail_resp.json()["alert"]["status"]

        assert list_status == detail_status == expected_mapped


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
        assert set(run.keys()) == {"spill", "alert", "vessels", "drift", "provenance"}

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
    def test_regenerates_a_report_when_no_file_exists(self, populated_spills: Path) -> None:
        """No report.html on disk, but populated_spills gives GET /api/runs/:id
        real spill data via the Phase-1 geojson path - the endpoint should
        build one on the fly rather than 404, so a stateless deploy (no
        local disk to have ever written report.html on) still works."""
        resp = client.get(f"/api/runs/{SCENE_ID}/report")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert SCENE_ID.encode() in resp.content

    def test_returns_report_when_exists(self, populated_spills: Path) -> None:
        # run_pipeline.py --report's default output: report.html next to
        # pipeline_result.json — NOT pipeline_result.json itself (that's the
        # raw combined JSON, already served by GET /api/runs/:id). A local
        # file, when present, wins over regenerating one.
        scene_dir = populated_spills.parent / SCENE_ID
        scene_dir.mkdir(parents=True, exist_ok=True)
        report_path = scene_dir / "report.html"
        report_path.write_text("<html><body>test report</body></html>", encoding="utf-8")

        resp = client.get(f"/api/runs/{SCENE_ID}/report")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert b"test report" in resp.content

    def test_does_not_serve_pipeline_result_as_report(self, populated_spills: Path) -> None:
        # A spills-less pipeline_result.json takes priority over the good
        # geojson fallback (get_run()'s own, pre-existing priority order) -
        # regeneration then has no real geometry to build a map from, so
        # this degrades to 404, never to serving the raw JSON as if it were
        # the report.
        scene_dir = populated_spills.parent / SCENE_ID
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "pipeline_result.json").write_text(
            json.dumps({"scene_id": SCENE_ID}), encoding="utf-8"
        )

        resp = client.get(f"/api/runs/{SCENE_ID}/report")
        assert resp.status_code == 404
        assert b'"scene_id"' not in resp.content

    def test_404_when_no_data_exists_anywhere(self, spills_dir: Path) -> None:
        resp = client.get("/api/runs/no-such-scene-at-all/report")
        assert resp.status_code == 404


class TestContractToPipelineResult:
    """contract_to_pipeline_result() is what lets the report endpoint
    regenerate from a Postgres-backed contract - that table never has a
    report.html file, or vessels/drift, sitting anywhere. Exercised here
    directly (no live DB needed) against exactly the shape
    src/api/registry.py::_get_run_postgres() produces."""

    POSTGRES_SHAPED_CONTRACT: Dict[str, Any] = {
        "spill": {
            "scene_id": "postgres_scene",
            "acquisition_timestamp": "2024-04-10T14:20:00+00:00",
            "confidence": 0.9,
            "area_km2": 5.0,
            "centroid": {"lat": 24.1, "lon": -90.0},
            "bbox": {"minLon": -90.5, "minLat": 23.5, "maxLon": -89.5, "maxLat": 24.5},
            "polygon": {
                "type": "Polygon",
                "coordinates": [[[-90.2, 23.8], [-89.7, 24.2], [-89.3, 24.0], [-89.8, 23.5], [-90.2, 23.8]]],
            },
            "major_axis_bearing": 45.0,
            "elongation": 2.0,
        },
        "alert": {
            "spill_id": "evt_test",
            "status": "new",
            "rules_fired": ["wind"],
            "first_seen": "2024-04-10T14:20:00+00:00",
            "last_updated": "2024-04-10T14:25:00+00:00",
        },
        # Never populated for a Postgres-backed run — see src/api/registry.py's
        # own module docstring on why.
        "vessels": [],
        "drift": {"forecast": [], "hindcast": []},
    }

    def test_produces_a_buildable_report_from_a_postgres_shaped_contract(self) -> None:
        from src.api.registry import contract_to_pipeline_result
        from src.output.report import build_report_html

        result = contract_to_pipeline_result(self.POSTGRES_SHAPED_CONTRACT, "postgres_scene")
        html = build_report_html(result)

        assert "postgres_scene" in html
        assert "<html" in html.lower()
        assert "wind" in html  # the surviving failed-rule name

    def test_regenerated_report_is_served_end_to_end(
        self, spills_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same contract, driven through the real endpoint with get_run()
        mocked to return it — stands in for a Postgres-backed run without
        needing a live database connection in this test."""
        from src.api import main as main_mod

        monkeypatch.setattr(
            main_mod, "get_run",
            lambda scene_id: dict(self.POSTGRES_SHAPED_CONTRACT) if scene_id == "postgres_scene" else None,
        )

        resp = client.get("/api/runs/postgres_scene/report")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert b"postgres_scene" in resp.content


class TestTriggerRunEndpoint:
    def test_400_without_scene_identifier(self) -> None:
        resp = client.post("/api/runs", json={}, headers=AUTH_HEADERS)
        assert resp.status_code == 400


class TestAskEndpoint:
    """Step 8.1 — POST /api/runs/{id}/ask. The LLM call itself is exercised
    in tests/test_attribution_qa.py; here we only check the endpoint's own
    wiring (404 for an unknown run, the answer_question() call, error
    mapping) against a real pipeline_result.json-backed run."""

    def _write_scene_result(self, spills_dir: Path) -> None:
        run_dir = spills_dir.parent / SCENE_ID
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "pipeline_result.json").write_text(
            json.dumps(SAMPLE_PIPELINE_RESULT), encoding="utf-8"
        )

    def test_404_for_unknown_run(self, populated_spills: Path) -> None:
        resp = client.post(
            "/api/runs/no-such-scene/ask", json={"question": "who?"}, headers=AUTH_HEADERS
        )
        assert resp.status_code == 404

    def test_answers_using_the_runs_own_data(
        self, populated_spills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_scene_result(populated_spills)

        captured: Dict[str, Any] = {}

        def fake_answer_question(run: Dict[str, Any], question: str, **kwargs: Any):
            from src.attribution.qa import QAAnswer

            captured["run"] = run
            captured["question"] = question
            return QAAnswer(answer="TANKER Gamma is the top match.", cited_vessels=["367123452"])

        from src.api import main as main_mod

        monkeypatch.setattr(main_mod, "answer_question", fake_answer_question)

        resp = client.post(
            f"/api/runs/{SCENE_ID}/ask",
            json={"question": "which vessel is most likely responsible?"},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "TANKER Gamma is the top match."
        assert body["cited_vessels"] == ["367123452"]
        assert captured["run"]["vessels"][0]["mmsi"] == "367123452"

    def test_503_when_llm_unreachable(
        self, populated_spills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._write_scene_result(populated_spills)

        from src.api import main as main_mod
        from src.attribution.qa import QAError

        def raise_qa_error(run: Dict[str, Any], question: str, **kwargs: Any):
            raise QAError("ANTHROPIC_API_KEY not set - copy .env.example to .env and fill it in")

        monkeypatch.setattr(main_mod, "answer_question", raise_qa_error)

        resp = client.post(
            f"/api/runs/{SCENE_ID}/ask", json={"question": "who?"}, headers=AUTH_HEADERS
        )
        assert resp.status_code == 503
        assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


class TestApiKeyAuth:
    """src/api/main.py::require_api_key — gates only the two POST endpoints.

    GETs must stay public no matter what; that's the regression that matters
    most here, since the live frontend never sends this header.
    """

    # --- POST /api/runs -------------------------------------------------- #
    def test_post_runs_rejects_missing_key(self) -> None:
        resp = client.post("/api/runs", json={})
        assert resp.status_code == 401

    def test_post_runs_rejects_wrong_key(self) -> None:
        resp = client.post("/api/runs", json={}, headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_post_runs_accepts_right_key(self) -> None:
        # Right key clears auth; 400 comes from the handler's own payload
        # validation (no scene_path/scene_id) — proof the request reached it.
        resp = client.post("/api/runs", json={}, headers=AUTH_HEADERS)
        assert resp.status_code == 400

    # --- POST /api/runs/{id}/ask ------------------------------------------ #
    def test_post_ask_rejects_missing_key(self, populated_spills: Path) -> None:
        resp = client.post(f"/api/runs/{SCENE_ID}/ask", json={"question": "who?"})
        assert resp.status_code == 401

    def test_post_ask_rejects_wrong_key(self, populated_spills: Path) -> None:
        resp = client.post(
            f"/api/runs/{SCENE_ID}/ask",
            json={"question": "who?"},
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 401

    def test_post_ask_accepts_right_key(
        self, populated_spills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.api import main as main_mod
        from src.attribution.qa import QAAnswer

        monkeypatch.setattr(
            main_mod, "answer_question",
            lambda run, question, **kwargs: QAAnswer(answer="ok", cited_vessels=[]),
        )
        resp = client.post(
            f"/api/runs/{SCENE_ID}/ask", json={"question": "who?"}, headers=AUTH_HEADERS
        )
        assert resp.status_code == 200

    # --- unset API_KEY must fail closed, never open ----------------------- #
    def test_unset_api_key_rejects_even_with_a_header_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("API_KEY", raising=False)
        resp = client.post("/api/runs", json={}, headers=AUTH_HEADERS)
        assert resp.status_code == 503

    # --- question length cap ---------------------------------------------- #
    def test_ask_rejects_an_overlong_question(self, populated_spills: Path) -> None:
        resp = client.post(
            f"/api/runs/{SCENE_ID}/ask",
            json={"question": "x" * 2001},
            headers=AUTH_HEADERS,
        )
        assert resp.status_code == 422

    # --- GETs stay public, with or without the header ---------------------- #
    def test_get_endpoints_work_with_no_key_at_all(self, populated_spills: Path) -> None:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/runs").status_code == 200
        assert client.get(f"/api/runs/{SCENE_ID}").status_code == 200
        assert client.get(f"/api/runs/{SCENE_ID}/report").status_code == 200
        assert client.get("/api/regions").status_code == 200
        assert client.get("/api/model/info").status_code == 200


class TestRegionsEndpoint:
    """Step 8.4 — GET /api/regions. Reads the real committed
    configs/demo_regions.yaml, so this also doubles as a check that the
    seeded file itself has the shape the acceptance criteria describe: one
    real region (a non-null scene_id) plus pending ones."""

    def test_returns_the_configured_list(self) -> None:
        resp = client.get("/api/regions")
        assert resp.status_code == 200
        regions = resp.json()
        assert len(regions) >= 1
        for r in regions:
            assert set(r.keys()) == {"id", "label", "bbox", "scene_id"}

    def test_exactly_one_real_region_backed_by_a_committed_run(self) -> None:
        resp = client.get("/api/regions")
        regions = resp.json()
        real = [r for r in regions if r["scene_id"]]
        pending = [r for r in regions if not r["scene_id"]]
        assert len(real) == 1
        assert real[0]["scene_id"] == "demo_pipeline"
        assert len(pending) >= 2


class TestRunsRegionFilter:
    """Step 8.4 — GET /api/runs?region=... . _load_regions() is monkeypatched
    here rather than relying on the real demo_regions.yaml, so this test
    doesn't drift if that file's contents ever change."""

    def _write_two_scenes(self, tmp_path: Path) -> None:
        for scene_id in ("region_scene_a", "region_scene_b"):
            run_dir = tmp_path / "data" / "processed" / scene_id
            run_dir.mkdir(parents=True)
            result = {
                "generated_at": "2024-04-10T14:30:00+00:00",
                "scene_id": scene_id,
                "spills": [_spill_entry(f"{scene_id}_spill_001", alerted=False, status="rejected")],
            }
            (run_dir / "pipeline_result.json").write_text(json.dumps(result), encoding="utf-8")

    def test_region_with_a_scene_id_filters_the_list(
        self, spills_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._write_two_scenes(tmp_path)
        from src.api import main as main_mod

        monkeypatch.setattr(
            main_mod, "_load_regions",
            lambda: [{"id": "region_a", "label": "Region A", "bbox": [0, 0, 1, 1],
                      "scene_id": "region_scene_a"}],
        )

        resp = client.get("/api/runs", params={"region": "region_a"})
        assert resp.status_code == 200
        assert {r["scene_id"] for r in resp.json()} == {"region_scene_a"}

    def test_pending_region_leaves_the_list_unfiltered(
        self, spills_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._write_two_scenes(tmp_path)
        from src.api import main as main_mod

        monkeypatch.setattr(
            main_mod, "_load_regions",
            lambda: [{"id": "pending", "label": "Pending", "bbox": [0, 0, 1, 1], "scene_id": None}],
        )

        resp = client.get("/api/runs", params={"region": "pending"})
        assert resp.status_code == 200
        assert {r["scene_id"] for r in resp.json()} == {"region_scene_a", "region_scene_b"}

    def test_unknown_region_leaves_the_list_unfiltered(
        self, spills_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._write_two_scenes(tmp_path)
        from src.api import main as main_mod

        monkeypatch.setattr(main_mod, "_load_regions", lambda: [])

        resp = client.get("/api/runs", params={"region": "no-such-region"})
        assert resp.status_code == 200
        assert {r["scene_id"] for r in resp.json()} == {"region_scene_a", "region_scene_b"}

    def test_no_region_param_leaves_the_list_unfiltered(
        self, spills_dir: Path, tmp_path: Path,
    ) -> None:
        self._write_two_scenes(tmp_path)
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert {r["scene_id"] for r in resp.json()} == {"region_scene_a", "region_scene_b"}
