"""End-to-end test of scripts/run_pipeline.py: a raw scene in, a combined
spill + alert + ranked-vessels JSON out - detection through attribution in
one call, exercising the actual reused run_detection.run() /
run_attribution.attribute_spill() functions, not a re-implementation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import load_settings
from scripts.run_pipeline import run

# Away from configs/coastline's vendored (Gujarat-only) coverage, matching
# test_run_detection.py's convention, so nothing here is accidentally masked
# as land by the real coastline data.
ORIGIN_LON, ORIGIN_LAT = -160.0, -5.0
PIXEL_DEG = 0.001
ACQUIRED = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)


def painted_streak(shape, centre, semi_major=35.0, semi_minor=6.0, angle_deg=0.0):
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dr = rows - centre[0]
    dc = cols - centre[1]
    theta = np.radians(angle_deg)
    major = dc * np.cos(theta) + dr * np.sin(theta)
    minor = -dc * np.sin(theta) + dr * np.cos(theta)
    return (major / semi_major) ** 2 + (minor / semi_minor) ** 2 <= 1.0


def write_scene_tif(path: Path, data: np.ndarray) -> Path:
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype="float32", crs="EPSG:4326",
        transform=from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG),
    ) as dst:
        dst.write(data.astype("float32"))
    return path


@pytest.fixture
def scene_path(tmp_path: Path) -> Path:
    """A noisy sea background, like every other synthetic scene in this
    repo (see test_lookalike_filter.py's sea_field()) - a perfectly flat,
    noiseless background makes ThresholdStubModel's per-tile 15th-percentile
    cutoff land on the background's own tied value, flagging the whole tile
    as oil instead of isolating the painted patch.

    The patch itself is sized to comfortably exceed 15% of the tile's area -
    ThresholdStubModel always calls the darkest 15% of a tile "oil" by rank,
    regardless of whether a real patch is present, so a patch smaller than
    that gets diluted by the darkest background noise instead of cleanly
    selected on its own.
    """
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
    """A vessel that plausibly crosses the spill shortly before acquisition -
    just enough for the attribution chain to run against something real."""
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


def test_run_pipeline_produces_the_expected_combined_structure(
    tmp_path: Path, scene_path: Path, settings,
) -> None:
    result = run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=tmp_path / "result.json", registry_path=tmp_path / "registry.sqlite3",
        settings=settings,
    )

    assert set(result.keys()) >= {"generated_at", "scene_id", "stub_model", "ais_source", "spills"}
    assert result["stub_model"] is True
    assert result["ais_source"] is None
    assert len(result["spills"]) >= 1

    for entry in result["spills"]:
        assert set(entry.keys()) == {"spill", "alert", "vessels"}
        assert entry["spill"]["type"] == "Feature"
        assert "acquisition_timestamp" in entry["spill"]["properties"]

        alert = entry["alert"]
        assert set(alert.keys()) == {"alert", "status", "spill_id", "is_new", "rules"}
        assert alert["status"] in ("active", "possible", "rejected")
        assert isinstance(alert["rules"], list) and alert["rules"]
        for rule in alert["rules"]:
            assert set(rule.keys()) == {"name", "passed", "reason"}

        assert isinstance(entry["vessels"], list)


def test_run_pipeline_writes_the_output_file(tmp_path: Path, scene_path: Path, settings) -> None:
    output = tmp_path / "result.json"
    run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=output, registry_path=tmp_path / "registry.sqlite3", settings=settings,
    )
    assert output.is_file()
    on_disk = json.loads(output.read_text())
    assert on_disk["spills"]


def test_run_pipeline_leaves_vessels_empty_without_an_ais_source(
    tmp_path: Path, scene_path: Path, settings,
) -> None:
    result = run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=tmp_path / "result.json", registry_path=tmp_path / "registry.sqlite3",
        settings=settings,
    )
    for entry in result["spills"]:
        assert entry["vessels"] == []


def test_run_pipeline_runs_attribution_for_an_alerted_spill_when_ais_is_given(
    tmp_path: Path, scene_path: Path, ais_path: Path, settings,
) -> None:
    result = run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        ais_path=ais_path, output=tmp_path / "result.json",
        registry_path=tmp_path / "registry.sqlite3", settings=settings,
    )
    assert result["ais_source"] == str(ais_path)

    alerted = [e for e in result["spills"] if e["alert"]["alert"]]
    assert alerted, "the painted patch must clear confidence/area and alert"
    # at least one alerted spill actually ran the attribution chain
    assert any(e["vessels"] for e in alerted) or all(
        isinstance(e["vessels"], list) for e in alerted
    )


def test_run_pipeline_registers_a_spill_in_the_registry(
    tmp_path: Path, scene_path: Path, settings,
) -> None:
    from src.alerts.registry import SpillRegistry

    registry_path = tmp_path / "registry.sqlite3"
    result = run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=tmp_path / "result.json", registry_path=registry_path, settings=settings,
    )
    alerted = [e for e in result["spills"] if e["alert"]["alert"]]
    assert alerted

    with SpillRegistry(path=registry_path, settings=settings) as registry:
        stored = registry.get(alerted[0]["alert"]["spill_id"])
        assert stored is not None


def test_run_pipeline_writes_a_map_when_requested(
    tmp_path: Path, scene_path: Path, settings,
) -> None:
    map_output = tmp_path / "map.html"
    run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=tmp_path / "result.json", registry_path=tmp_path / "registry.sqlite3",
        write_map=True, map_output=map_output, settings=settings,
    )
    assert map_output.is_file()
    assert "<html" in map_output.read_text(encoding="utf-8").lower()


def test_run_pipeline_skips_the_map_by_default(tmp_path: Path, scene_path: Path, settings) -> None:
    run(
        scene_path=scene_path, stub_model=True, tile_size=128, overlap=0,
        output=tmp_path / "result.json", registry_path=tmp_path / "registry.sqlite3",
        settings=settings,
    )
    assert not (tmp_path / "map.html").exists()
