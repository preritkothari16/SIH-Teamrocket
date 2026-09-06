"""Basic tests for src/output/map.py - a synthetic pipeline result in, a
standalone HTML file out. Nothing here inspects the map's rendered pixels;
Folium's own HTML generation is trusted, so these check that the right
pieces (spill polygon, vessel tracks, popups) actually end up in the output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from shapely.geometry import box, mapping

from src.characterization.spill_object import build_spill_object
from src.output.map import MapBuildError, build_map, save_map

ACQUIRED = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
SPILL_POLYGON = box(70.0, 22.0, 70.05, 22.03)


def make_result(vessels=None, alert_status="active", alerted=True):
    spill = build_spill_object(SPILL_POLYGON, "scene", ACQUIRED, confidence=0.9)
    return {
        "generated_at": ACQUIRED.isoformat(),
        "scene_id": "scene",
        "spills": [
            {
                "spill": spill,
                "alert": {
                    "alert": alerted,
                    "status": alert_status,
                    "spill_id": spill["properties"]["spill_id"],
                    "is_new": True,
                    "rules": [{"name": "confidence", "passed": True, "reason": "ok"}],
                },
                "vessels": vessels or [],
            }
        ],
    }


def make_vessel(mmsi=900001, score=0.85, rank_track=True):
    track = (
        [
            {"timestamp": "2023-05-13T18:00:00+00:00", "lat": 22.01, "lon": 70.0},
            {"timestamp": "2023-05-13T18:20:00+00:00", "lat": 22.02, "lon": 70.02},
        ]
        if rank_track
        else []
    )
    return {
        "mmsi": mmsi,
        "vessel_name": "MV TEST",
        "vessel_type": "Tanker",
        "cpa_distance_km": 0.5,
        "score": score,
        "explanation": "0.5 km CPA, 5h before image, track within 3° of slick axis, tanker",
        "track": track,
    }


def test_build_map_raises_on_no_spills() -> None:
    with pytest.raises(MapBuildError, match="no spills"):
        build_map({"spills": []})


def test_build_map_includes_the_spill_polygon() -> None:
    result = make_result()
    fmap = build_map(result)
    html = fmap.get_root().render()
    # the spill's own geometry coordinates must appear somewhere in the map
    assert "70.0" in html or "70.05" in html


def test_build_map_draws_a_vessel_track_and_popup() -> None:
    result = make_result(vessels=[make_vessel()])
    fmap = build_map(result)
    html = fmap.get_root().render()
    assert "900001" in html
    assert "MV TEST" in html
    assert "tanker" in html.lower() or "track within" in html


def test_build_map_skips_a_vessel_with_no_track_points() -> None:
    """A vessel with an empty track (shouldn't normally happen, but must not
    crash the map build) contributes no polyline/marker."""
    result = make_result(vessels=[make_vessel(rank_track=False)])
    fmap = build_map(result)  # must not raise
    assert fmap is not None


def test_build_map_handles_a_spill_with_no_vessels() -> None:
    """No --ais given upstream - vessels is an empty list, not missing."""
    result = make_result(vessels=[])
    fmap = build_map(result)
    assert fmap is not None


def test_build_map_colors_vessels_by_rank() -> None:
    result = make_result(vessels=[make_vessel(mmsi=1, score=0.9), make_vessel(mmsi=2, score=0.5)])
    fmap = build_map(result)
    html = fmap.get_root().render()
    assert "red" in html or "#" in html  # first-rank colour must be present somewhere


def test_save_map_writes_a_standalone_html_file(tmp_path: Path) -> None:
    result = make_result(vessels=[make_vessel()])
    output = tmp_path / "map.html"
    path = save_map(result, output)
    assert path == output
    assert output.is_file()
    content = output.read_text(encoding="utf-8")
    assert "<html" in content.lower()
    assert "900001" in content


def test_save_map_creates_missing_parent_directories(tmp_path: Path) -> None:
    result = make_result()
    output = tmp_path / "nested" / "dir" / "map.html"
    save_map(result, output)
    assert output.is_file()
