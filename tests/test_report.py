"""src/output/report.py smoke test: report generation must run without error
on a real (synthetic-scene) combined pipeline result and produce a
non-empty, openable HTML file.

Reuses test_pipeline_integration.py's scene-fixture helpers (tests/ is a
package, same pattern test_env_service.py already uses to reuse
test_env_currents.py's fixtures) rather than duplicating the painted-streak
recipe.
"""

from __future__ import annotations

import html
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_settings
from src.output.report import ReportBuildError, build_report_html, save_report
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
def pipeline_result(tmp_path: Path, scene_path: Path, ais_path: Path, settings):
    return run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        ais_path=ais_path, output=tmp_path / "result.json",
        registry_path=tmp_path / "registry.sqlite3", settings=settings,
    )


def test_build_report_html_runs_without_error_and_is_substantial(pipeline_result) -> None:
    doc = build_report_html(pipeline_result)
    assert doc.lower().startswith("<!doctype html>")
    assert "<html" in doc.lower()
    assert len(doc) > 1000  # a real map embed alone is several KB


def test_build_report_html_includes_the_alerted_spills_details(pipeline_result) -> None:
    doc = build_report_html(pipeline_result)
    alerted = [e for e in pipeline_result["spills"] if e["alert"]["alert"]]
    assert alerted, "the painted patch must clear confidence/area and alert"

    spill_id = alerted[0]["alert"]["spill_id"]
    assert spill_id in doc

    for vessel in alerted[0]["vessels"]:
        assert str(vessel["mmsi"]) in doc
        assert vessel["explanation"] in doc or html.escape(vessel["explanation"]) in doc


def test_build_report_html_includes_every_drift_forecast_horizon(pipeline_result) -> None:
    doc = build_report_html(pipeline_result)
    for entry in pipeline_result["spills"]:
        for forecast in entry["drift_forecast"]:
            assert f"+{forecast['hours_elapsed']:.0f}h" in doc


def test_build_report_html_raises_on_a_result_with_no_spills() -> None:
    with pytest.raises(ReportBuildError):
        build_report_html({"spills": []})


def test_save_report_writes_a_non_empty_openable_file(tmp_path: Path, pipeline_result) -> None:
    output = tmp_path / "report.html"
    result_path = save_report(pipeline_result, output)
    assert result_path == output
    assert output.is_file()
    on_disk = output.read_text(encoding="utf-8")
    assert len(on_disk) > 1000
    assert "<html" in on_disk.lower()


def test_run_pipeline_report_flag_produces_an_openable_report(
    tmp_path: Path, scene_path: Path, ais_path: Path, settings,
) -> None:
    """The actual acceptance criterion: --report on scripts/run_pipeline.py
    must produce a report file, end to end, no separate wiring needed."""
    report_output = tmp_path / "report.html"
    run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        ais_path=ais_path, output=tmp_path / "result2.json",
        registry_path=tmp_path / "registry2.sqlite3",
        write_report=True, report_output=report_output, settings=settings,
    )
    assert report_output.is_file()
    doc = report_output.read_text(encoding="utf-8")
    assert "<html" in doc.lower()
    assert len(doc) > 1000


def test_run_pipeline_skips_the_report_by_default(
    tmp_path: Path, scene_path: Path, settings,
) -> None:
    run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=tmp_path / "result3.json", registry_path=tmp_path / "registry3.sqlite3",
        settings=settings,
    )
    assert not (tmp_path / "report.html").exists()
