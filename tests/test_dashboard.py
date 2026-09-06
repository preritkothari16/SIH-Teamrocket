"""src/output/dashboard.py tests: the FastAPI app must serve a real pipeline
run's output as one renderable HTML page, with the hindcast-corridor layer
actually populated for an alerted spill.

Reuses test_pipeline_integration.py's scene fixtures, same pattern as
test_report.py.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import load_settings
from src.drift.hindcast import OriginSnapshot
from src.output.dashboard import create_app, hindcast_corridors_for, render_dashboard_html
from scripts.run_pipeline import run
from tests.test_pipeline_integration import (
    ACQUIRED,
    ORIGIN_LAT,
    ORIGIN_LON,
    PIXEL_DEG,
    painted_streak,
    write_scene_tif,
)


@pytest.fixture
def scene_path(tmp_path: Path) -> Path:
    shape = (128, 128)
    rng = np.random.default_rng(0)
    data = (-10.0 + rng.normal(0.0, 0.3, size=shape)).astype(np.float32)
    data[painted_streak(shape, (64, 64), semi_major=65.0, semi_minor=16.0)] = -19.0
    return write_scene_tif(tmp_path / "scene.tif", data)


@pytest.fixture
def settings(tmp_path: Path):
    base = load_settings()
    return base.model_copy(update={
        "paths": base.paths.model_copy(update={
            "processed_dir": tmp_path / "processed",
            "raw_dir": tmp_path / "raw",
        }),
    })


@pytest.fixture
def ais_path(tmp_path: Path) -> Path:
    t0 = ACQUIRED - timedelta(hours=5)
    rows = []
    for i in range(5):
        f = i / 4
        rows.append({
            "mmsi": 900001, "timestamp": (t0 + timedelta(minutes=20 * i)).isoformat(),
            "lat": ORIGIN_LAT - 64 * PIXEL_DEG, "lon": ORIGIN_LON + (63 + 0.2 * f) * PIXEL_DEG,
            "sog": 10.0, "cog": 90.0, "heading": 90.0,
            "vessel_name": "MV TEST", "vessel_type": "Tanker",
        })
    path = tmp_path / "ais.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def result_path(tmp_path: Path, scene_path: Path, ais_path: Path, settings) -> Path:
    output = tmp_path / "result.json"
    run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        ais_path=ais_path, output=output,
        registry_path=tmp_path / "registry.sqlite3", settings=settings,
    )
    return output


@pytest.fixture
def pipeline_result(result_path: Path) -> dict:
    return json.loads(result_path.read_text(encoding="utf-8"))


def test_hindcast_corridors_for_is_populated_for_alerted_spills(pipeline_result, settings) -> None:
    corridors = hindcast_corridors_for(pipeline_result, settings)
    alerted_ids = {
        e["alert"]["spill_id"] for e in pipeline_result["spills"] if e["alert"]["alert"]
    }
    assert alerted_ids  # the painted patch must clear confidence/area and alert
    assert set(corridors.keys()) == alerted_ids
    for corridor in corridors.values():
        assert corridor
        assert all(isinstance(s, OriginSnapshot) for s in corridor)


def test_hindcast_corridors_for_skips_unalerted_spills(pipeline_result, settings) -> None:
    corridors = hindcast_corridors_for(pipeline_result, settings)
    unalerted_ids = {
        e["alert"]["spill_id"] for e in pipeline_result["spills"]
        if e["alert"].get("spill_id") and not e["alert"]["alert"]
    }
    assert not (unalerted_ids & corridors.keys())


def test_render_dashboard_html_includes_layer_names_and_metadata(pipeline_result, settings) -> None:
    doc = render_dashboard_html(pipeline_result, settings=settings)
    assert "<html" in doc.lower()
    for layer_name in ("Detected slicks", "Drift forecast", "Hindcast corridor", "Vessel tracks"):
        assert layer_name in doc
    assert pipeline_result["scene_id"] in doc


def test_render_dashboard_html_includes_the_ranked_vessel_table(pipeline_result, settings) -> None:
    doc = render_dashboard_html(pipeline_result, settings=settings)
    alerted = [e for e in pipeline_result["spills"] if e["alert"]["alert"]]
    assert alerted
    for vessel in alerted[0]["vessels"]:
        assert str(vessel["mmsi"]) in doc


def test_dashboard_app_serves_the_page_over_http(result_path: Path, settings) -> None:
    """The actual acceptance criterion: the FastAPI app must actually run and
    render a real pipeline run's output when hit over HTTP."""
    app = create_app(result_path, settings=settings)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SAR oil-spill dashboard" in response.text
    assert "Hindcast corridor" in response.text
