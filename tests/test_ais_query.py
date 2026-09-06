"""query.py tests: the space-time box must keep exactly the right vessels.

A synthetic AIS fixture with a few MMSIs, deliberately placed inside and
outside the window / area, so each rejection reason is isolated.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.characterization.spill_object import build_spill_object
from src.config import load_settings
from src.ais.query import (
    AISQueryError,
    query_ais_for_spill,
    search_area,
    search_window,
    spill_geometry_and_time,
)

ACQUIRED = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
SPILL_POLYGON = box(70.0, 22.0, 70.05, 22.03)  # ~5km x 3km near 22N


def ais_point(mmsi: int, when: datetime, lat: float, lon: float) -> dict:
    return {
        "mmsi": mmsi, "timestamp": when, "lat": lat, "lon": lon,
        "sog": 8.0, "cog": 90.0, "heading": 90.0,
        "vessel_name": f"V{mmsi}", "vessel_type": "Cargo",
    }


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def ais_fixture() -> pd.DataFrame:
    """Six MMSIs, each isolating one reason to keep or drop it.

    100: inside the window, inside the area -> kept, both its pings.
    200: inside the window, far outside the area -> dropped.
    300: inside the area, but 4 days before acquisition (outside the 48h
         window) -> dropped.
    400: one ping inside the area, one far outside, same window -> kept,
         BOTH pings returned (qualifies on the one hit, not filtered per-ping).
    500: exactly at the acquisition time, right in the area -> kept (the
         window's closed at acquisition_time, not open).
    600: one second after acquisition_time -> dropped (the window looks
         backward only, never forward).
    """
    rows = [
        ais_point(100, ACQUIRED - timedelta(hours=6), 22.01, 70.02),
        ais_point(100, ACQUIRED - timedelta(hours=5), 22.015, 70.025),
        ais_point(200, ACQUIRED - timedelta(hours=6), 40.0, 100.0),
        ais_point(300, ACQUIRED - timedelta(days=4), 22.01, 70.02),
        ais_point(400, ACQUIRED - timedelta(hours=10), 22.01, 70.02),
        ais_point(400, ACQUIRED - timedelta(hours=8), 40.0, 100.0),
        ais_point(500, ACQUIRED, 22.01, 70.02),
        ais_point(600, ACQUIRED + timedelta(seconds=1), 22.01, 70.02),
    ]
    return pd.DataFrame(rows)


def test_query_keeps_exactly_the_qualifying_mmsi(ais_fixture, settings) -> None:
    result = query_ais_for_spill((SPILL_POLYGON, ACQUIRED), ais_fixture, settings=settings)
    assert set(result["mmsi"].unique()) == {100, 400, 500}


def test_query_returns_every_point_for_a_qualifying_vessel_not_just_the_hit(
    ais_fixture, settings
) -> None:
    """400 has one ping in the area and one far away - both must come back."""
    result = query_ais_for_spill((SPILL_POLYGON, ACQUIRED), ais_fixture, settings=settings)
    assert (result["mmsi"] == 400).sum() == 2


def test_query_window_is_closed_at_acquisition_time_not_open(ais_fixture, settings) -> None:
    result = query_ais_for_spill((SPILL_POLYGON, ACQUIRED), ais_fixture, settings=settings)
    assert 500 in set(result["mmsi"])
    assert 600 not in set(result["mmsi"])


def test_query_window_looks_backward_only(ais_fixture, settings) -> None:
    result = query_ais_for_spill((SPILL_POLYGON, ACQUIRED), ais_fixture, settings=settings)
    assert 300 not in set(result["mmsi"])  # inside the area, but 4 days too early


def test_query_on_an_empty_ais_frame_yields_no_rows(settings) -> None:
    empty = pd.DataFrame(
        columns=["mmsi", "timestamp", "lat", "lon", "sog", "cog", "heading",
                 "vessel_name", "vessel_type"]
    )
    result = query_ais_for_spill((SPILL_POLYGON, ACQUIRED), empty, settings=settings)
    assert result.empty


def test_query_result_is_sorted_by_mmsi_then_time(ais_fixture, settings) -> None:
    result = query_ais_for_spill((SPILL_POLYGON, ACQUIRED), ais_fixture, settings=settings)
    assert result["mmsi"].tolist() == sorted(result["mmsi"].tolist())
    for _, group in result.groupby("mmsi"):
        assert group["timestamp"].is_monotonic_increasing


def test_query_accepts_a_geojson_spill_feature(ais_fixture, settings) -> None:
    """The real call shape: what spills_from_blobs actually hands downstream."""
    feature = build_spill_object(SPILL_POLYGON, "scene", ACQUIRED)
    result = query_ais_for_spill(feature, ais_fixture, settings=settings)
    assert set(result["mmsi"].unique()) == {100, 400, 500}


def test_query_rejects_a_spill_with_no_timestamp() -> None:
    with pytest.raises(AISQueryError, match="acquisition_timestamp"):
        query_ais_for_spill(
            {"type": "Feature", "geometry": SPILL_POLYGON.__geo_interface__, "properties": {}},
            pd.DataFrame(columns=["mmsi", "timestamp", "lat", "lon"]),
        )


# --------------------------------------------------------------------------- #
# the building blocks, in isolation
# --------------------------------------------------------------------------- #
def test_search_window_is_one_sided_and_configurable(settings) -> None:
    start, end = search_window(ACQUIRED, window_hours=48, settings=settings)
    assert end == ACQUIRED
    assert start == ACQUIRED - timedelta(hours=48)

    narrow_start, narrow_end = search_window(ACQUIRED, window_hours=6, settings=settings)
    assert narrow_end == ACQUIRED
    assert narrow_start == ACQUIRED - timedelta(hours=6)


def test_search_area_buffer_is_measured_in_real_km_not_degrees(settings) -> None:
    """A point just inside a 30 km buffer is kept; just outside, it is not.

    Distances are computed independently here (equirectangular approximation
    at this latitude) so the assertion does not just re-implement search_area.
    """
    area_30km = search_area(SPILL_POLYGON, buffer_km=30.0, settings=settings)
    west, south, east, north = SPILL_POLYGON.bounds

    close_point = Point(east + 0.05, (south + north) / 2)  # a few km outside the box
    assert area_30km.contains(close_point)

    far_point = Point(east + 1.0, (south + north) / 2)  # ~100 km east, well past 30 km
    assert not area_30km.contains(far_point)


def test_spill_geometry_and_time_from_a_geojson_feature() -> None:
    feature = build_spill_object(SPILL_POLYGON, "scene", ACQUIRED)
    geometry, timestamp = spill_geometry_and_time(feature)
    assert geometry.equals(SPILL_POLYGON)
    assert timestamp == ACQUIRED


def test_spill_geometry_and_time_from_a_bare_tuple() -> None:
    geometry, timestamp = spill_geometry_and_time((SPILL_POLYGON, ACQUIRED))
    assert geometry.equals(SPILL_POLYGON)
    assert timestamp == ACQUIRED


def test_spill_geometry_and_time_accepts_wgs84_written_as_epsg_4326() -> None:
    """The default - and the only CRS any spill Phase 1 actually produces."""
    feature = build_spill_object(SPILL_POLYGON, "scene", ACQUIRED, crs="EPSG:4326")
    geometry, timestamp = spill_geometry_and_time(feature)
    assert geometry.equals(SPILL_POLYGON)
    assert timestamp == ACQUIRED


def test_spill_geometry_and_time_accepts_a_spill_with_no_crs_recorded() -> None:
    feature = build_spill_object(SPILL_POLYGON, "scene", ACQUIRED)
    del feature["properties"]["crs"]
    geometry, timestamp = spill_geometry_and_time(feature)
    assert geometry.equals(SPILL_POLYGON)


def test_spill_geometry_and_time_rejects_a_non_wgs84_spill() -> None:
    """The regression case: a projected spill must never be silently treated
    as lon/lat degrees - AIS coordinates are WGS84 and would be compared
    against the wrong numbers entirely.
    """
    metric_polygon = box(0.0, 0.0, 2000.0, 3000.0)
    feature = build_spill_object(metric_polygon, "scene", ACQUIRED, crs="EPSG:32643")
    with pytest.raises(AISQueryError, match="not WGS84"):
        spill_geometry_and_time(feature)
