"""End-to-end test of scripts/run_attribution.py: query -> tracks -> filter ->
scoring, on a synthetic spill with one obviously guilty vessel and a few
vessels that must not be blamed for it.

The spill polygon is an elongated east-west box near the equator, so its
major axis has a compass bearing of ~90 deg (verified via
src.characterization.spill_object.orientation_and_elongation's own
convention - see src/attribution/scoring.py's slick_axis_bearing) and 1
degree of lon/lat is ~111.32 km, keeping every distance in this fixture
hand-checkable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import box

from src.characterization.spill_object import build_spill_object
from src.config import load_settings
from scripts.run_attribution import load_spills, run

ACQUIRED = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
SPILL_POLYGON = box(70.0, 22.0, 70.06, 22.01)  # ~6.7 km x 1.1 km, long axis east-west
SCENE_ID = "validation_scene"


def ais_row(mmsi, when, lat, lon, sog, heading, name, vtype):
    return {
        "mmsi": mmsi, "timestamp": when.isoformat(), "lat": lat, "lon": lon,
        "sog": sog, "cog": heading, "heading": heading,
        "vessel_name": name, "vessel_type": vtype,
    }


def straight_line(mmsi, lon0, lat0, lon1, lat1, sog, heading, name, vtype,
                   t0, n=5, step_minutes=20):
    rows = []
    for i in range(n):
        f = i / (n - 1)
        rows.append(ais_row(
            mmsi, t0 + timedelta(minutes=step_minutes * i),
            lat0 + (lat1 - lat0) * f, lon0 + (lon1 - lon0) * f,
            sog, heading, name, vtype,
        ))
    return rows


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def spill_path(tmp_path: Path) -> Path:
    feature = build_spill_object(SPILL_POLYGON, SCENE_ID, ACQUIRED, spill_id="spill_001")
    path = tmp_path / "spill.geojson"
    path.write_text(json.dumps(feature), encoding="utf-8")
    return path


@pytest.fixture
def ais_path(tmp_path: Path) -> Path:
    rows = []

    # GUILTY: crosses right through the polygon 5h before acquisition,
    # heading due east (~90 deg) - aligned with the slick's own east-west
    # axis - at a plausible tanker speed.
    guilty_t0 = ACQUIRED - timedelta(hours=5, minutes=40)
    rows += straight_line(
        900001, 69.9, 22.005, 70.16, 22.005, sog=12.0, heading=90.0,
        name="MV GUILTY", vtype="Tanker", t0=guilty_t0,
    )

    # FAR: nowhere near the spill the entire window - dropped by CPA.
    far_t0 = ACQUIRED - timedelta(hours=10)
    rows += straight_line(
        900002, 100.0, 40.0, 100.02, 40.02, sog=10.0, heading=45.0,
        name="MV FAR", vtype="Cargo", t0=far_t0,
    )

    # MOORED: sits right next to the spill, but never moves - dropped by the
    # stationarity check despite a near-zero CPA.
    moored_t0 = ACQUIRED - timedelta(hours=8)
    rows += straight_line(
        900003, 70.03, 22.005, 70.03, 22.005, sog=0.05, heading=0.0,
        name="MV MOORED", vtype="Fishing", t0=moored_t0,
    )

    # WEAK: passes near the spill and is moving, so it is not filtered out,
    # but scores far below GUILTY - wrong timing (minutes before acquisition,
    # not enough time for a slick to form), heading perpendicular to the
    # slick's axis, and a low-prior vessel type.
    weak_t0 = ACQUIRED - timedelta(minutes=30)
    rows += straight_line(
        900004, 70.03, 21.95, 70.03, 22.05, sog=8.0, heading=0.0,
        name="MV WEAK", vtype="Leisure", t0=weak_t0, n=3, step_minutes=10,
    )

    # AFTER: would cross right through the spill, but only after the SAR
    # image was acquired - the one-sided search window must exclude it
    # entirely (it must not even appear as a candidate, guilty or not).
    after_t0 = ACQUIRED + timedelta(hours=1)
    rows += straight_line(
        900005, 69.9, 22.005, 70.16, 22.005, sog=12.0, heading=90.0,
        name="MV AFTER", vtype="Tanker", t0=after_t0,
    )

    path = tmp_path / "ais.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_spills_reads_a_single_feature(spill_path: Path) -> None:
    spills = load_spills(spill_path)
    assert len(spills) == 1
    assert spills[0]["properties"]["spill_id"] == "spill_001"


def test_run_attribution_end_to_end(tmp_path: Path, spill_path: Path, ais_path: Path, settings) -> None:
    output = tmp_path / "result.json"
    result = run(spill_path=spill_path, ais_path=ais_path, output=output, settings=settings)

    assert output.is_file()
    on_disk = json.loads(output.read_text())
    assert on_disk["spills"][0]["spill_id"] == "spill_001"
    assert result == on_disk


def test_the_guilty_vessel_ranks_first(spill_path: Path, ais_path: Path, settings) -> None:
    result = run(spill_path=spill_path, ais_path=ais_path,
                 output=spill_path.with_suffix(".attribution.json"), settings=settings)

    candidates = result["spills"][0]["candidates"]
    mmsi_ranked = [c["mmsi"] for c in candidates]

    assert mmsi_ranked[0] == 900001, f"expected MV GUILTY first, got order {mmsi_ranked}"
    assert 900002 not in mmsi_ranked, "MV FAR must be dropped by the CPA/search-radius cutoff"
    assert 900003 not in mmsi_ranked, "MV MOORED must be dropped by the stationarity check"
    assert 900005 not in mmsi_ranked, "MV AFTER must be excluded by the one-sided time window"

    guilty = candidates[0]
    assert guilty["score"] > 0.7
    assert "tanker" in guilty["explanation"]
    assert guilty["track"], "the candidate must carry its track segment"

    if 900004 in mmsi_ranked:
        weak = next(c for c in candidates if c["mmsi"] == 900004)
        assert weak["score"] < guilty["score"]


def test_load_spills_reads_a_feature_collection(tmp_path: Path) -> None:
    from src.characterization.spill_object import feature_collection

    feature = build_spill_object(SPILL_POLYGON, SCENE_ID, ACQUIRED, spill_id="a")
    collection = feature_collection([feature], SCENE_ID, ACQUIRED)
    path = tmp_path / "collection.geojson"
    path.write_text(json.dumps(collection), encoding="utf-8")

    spills = load_spills(path)
    assert len(spills) == 1
    assert spills[0]["properties"]["spill_id"] == "a"


def test_load_spills_on_an_empty_feature_collection_yields_nothing(tmp_path: Path) -> None:
    from src.characterization.spill_object import feature_collection

    path = tmp_path / "empty.geojson"
    path.write_text(json.dumps(feature_collection([], SCENE_ID, ACQUIRED)), encoding="utf-8")
    assert load_spills(path) == []
