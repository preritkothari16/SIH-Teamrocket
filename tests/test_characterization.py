"""Characterization tests.

The one that matters is area: a polygon of known real-world size must measure
correctly, and measuring it in lon/lat degrees must be visibly wrong - that is
the mistake this module exists to prevent.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape

from src.characterization.spill_object import (
    area_km2,
    build_spill_object,
    equal_area_crs_for,
    feature_collection,
    mask_to_polygon,
    orientation_and_elongation,
    perimeter_km,
    spills_from_blobs,
    spills_from_mask,
    to_equal_area,
)
from src.config import load_settings
from src.detection.dataset import OIL_CLASS
from src.detection.lookalike_filter import filter_lookalikes

SCENE_ID = "test_scene"
ACQUIRED = datetime(2023, 5, 14, 0, 33, 32, tzinfo=timezone.utc)

# 0.001 deg pixels near 22 N: one degree of latitude is ~110.6 km there.
ORIGIN_LON, ORIGIN_LAT = 70.0, 22.0
PIXEL_DEG = 0.001


def scene_transform():
    return from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG)


# --------------------------------------------------------------------------- #
# area - the CRS-correctness of this is the whole point
# --------------------------------------------------------------------------- #
def test_area_of_a_polygon_of_known_size() -> None:
    """A 0.1 deg by 0.1 deg box at 22 N is ~11.5 x 10.3 km, so ~118 km².

    Cross-checked against the geodesic: 0.1 deg of latitude is 11.06 km, and
    0.1 deg of longitude at 22 N is 11.06 * cos(22) = 10.26 km.
    """
    polygon = box(ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 0.1, ORIGIN_LAT + 0.1)
    expected = (0.1 * 110.57) * (0.1 * 111.32 * np.cos(np.radians(22.05)))

    measured = area_km2(polygon)

    assert measured == pytest.approx(expected, rel=0.02)
    assert 110.0 < measured < 125.0


def test_area_in_degrees_would_have_been_wrong() -> None:
    """Guards the actual bug: shapely's raw .area is in square degrees."""
    polygon = box(ORIGIN_LON, ORIGIN_LAT, ORIGIN_LON + 0.1, ORIGIN_LAT + 0.1)
    naive = polygon.area  # 0.01 square degrees, a meaningless number
    assert naive == pytest.approx(0.01)
    assert area_km2(polygon) / naive > 10_000  # nowhere near each other


def test_area_shrinks_with_latitude_for_the_same_degree_box() -> None:
    """A degree of longitude narrows towards the poles; the area must follow."""
    equator = box(0.0, 0.0, 0.1, 0.1)
    high = box(0.0, 60.0, 0.1, 60.1)
    assert area_km2(high) < 0.55 * area_km2(equator)


def test_area_matches_a_projected_polygon_of_exact_size() -> None:
    """A 2 km x 3 km rectangle in a metric CRS is exactly 6 km²."""
    metric = box(0.0, 0.0, 2000.0, 3000.0)
    assert area_km2(metric, source_crs="EPSG:32643") == pytest.approx(6.0, rel=1e-9)


def test_area_of_a_polygon_with_a_hole_excludes_the_hole() -> None:
    outer = box(70.0, 22.0, 70.1, 22.1)
    inner = box(70.04, 22.04, 70.06, 22.06)
    holed = outer.difference(inner)
    assert area_km2(holed) == pytest.approx(area_km2(outer) - area_km2(inner), rel=1e-6)


def test_area_of_an_empty_geometry_is_zero() -> None:
    assert area_km2(box(0, 0, 1, 1).intersection(box(5, 5, 6, 6))) == 0.0


def test_equal_area_crs_is_centred_on_the_geometry() -> None:
    polygon = box(70.0, 22.0, 70.1, 22.1)
    projected, crs = to_equal_area(polygon)
    assert crs.is_projected
    assert "Azimuthal Equal Area" in crs.coordinate_operation.method_name
    # the centroid maps to the projection origin
    assert projected.centroid.x == pytest.approx(0.0, abs=1.0)
    assert projected.centroid.y == pytest.approx(0.0, abs=1.0)
    assert equal_area_crs_for(polygon).is_projected


def test_perimeter_is_in_kilometres() -> None:
    """A 2 km x 3 km rectangle has a 10 km perimeter."""
    metric = box(0.0, 0.0, 2000.0, 3000.0)
    assert perimeter_km(metric, source_crs="EPSG:32643") == pytest.approx(10.0, rel=1e-9)

    geographic = box(70.0, 22.0, 70.1, 22.1)
    assert perimeter_km(geographic) == pytest.approx(2 * (11.06 + 10.26), rel=0.03)


# --------------------------------------------------------------------------- #
# mask -> polygon
# --------------------------------------------------------------------------- #
def test_mask_to_polygon_round_trips_a_rectangle() -> None:
    """A 40x20 px block must come back as the right box in map coordinates."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 20:60] = True  # rows 10-30, cols 20-60

    polygons = mask_to_polygon(mask, scene_transform(), simplify_tolerance_m=0.0,
                               min_area_km2=0.0)

    assert len(polygons) == 1
    west, south, east, north = polygons[0].bounds
    assert west == pytest.approx(ORIGIN_LON + 20 * PIXEL_DEG)
    assert east == pytest.approx(ORIGIN_LON + 60 * PIXEL_DEG)
    assert north == pytest.approx(ORIGIN_LAT - 10 * PIXEL_DEG)
    assert south == pytest.approx(ORIGIN_LAT - 30 * PIXEL_DEG)


def test_mask_to_polygon_area_matches_the_pixel_count() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 20:60] = True  # 800 px

    polygon = mask_to_polygon(mask, scene_transform(), simplify_tolerance_m=0.0,
                              min_area_km2=0.0)[0]

    pixel_km2 = area_km2(box(70.0, 22.0, 70.0 + PIXEL_DEG, 22.0 + PIXEL_DEG))
    assert area_km2(polygon) == pytest.approx(800 * pixel_km2, rel=0.01)


def test_mask_to_polygon_finds_every_separate_blob() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[60:90, 60:95] = True

    polygons = mask_to_polygon(mask, scene_transform(), simplify_tolerance_m=0.0,
                               min_area_km2=0.0)
    assert len(polygons) == 2
    # sorted largest first
    assert area_km2(polygons[0]) > area_km2(polygons[1])


def test_mask_to_polygon_drops_slivers_below_the_minimum() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:40] = True  # 900 px -> ~10 km²
    mask[80, 80] = True  # a single pixel, ~0.011 km²

    polygons = mask_to_polygon(mask, scene_transform(), min_area_km2=0.05)
    assert len(polygons) == 1
    assert area_km2(polygons[0]) > 5.0


def test_mask_to_polygon_preserves_a_hole() -> None:
    """A clear patch inside a slick makes it genuinely smaller."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:60, 10:60] = True
    mask[30:40, 30:40] = False

    polygon = mask_to_polygon(mask, scene_transform(), simplify_tolerance_m=0.0,
                              min_area_km2=0.0)[0]
    assert len(polygon.interiors) == 1


def test_mask_to_polygon_on_an_empty_mask() -> None:
    assert mask_to_polygon(np.zeros((50, 50), dtype=bool), scene_transform()) == []


def test_simplification_reduces_vertices_without_moving_the_polygon() -> None:
    rng = np.random.default_rng(0)
    mask = np.zeros((120, 120), dtype=bool)
    mask[20:100, 20:100] = True
    mask &= rng.random((120, 120)) > 0.02  # ragged edges

    detailed = mask_to_polygon(mask, scene_transform(), simplify_tolerance_m=0.0,
                               min_area_km2=0.0)[0]
    simplified = mask_to_polygon(mask, scene_transform(), simplify_tolerance_m=200.0,
                                 min_area_km2=0.0)[0]

    assert len(simplified.exterior.coords) < len(detailed.exterior.coords)
    assert area_km2(simplified) == pytest.approx(area_km2(detailed), rel=0.35)


# --------------------------------------------------------------------------- #
# shape descriptors
# --------------------------------------------------------------------------- #
def test_orientation_and_elongation_of_a_known_rectangle() -> None:
    """A 4 km x 1 km east-west rectangle: elongation 4, orientation ~0 deg."""
    metric = box(0.0, 0.0, 4000.0, 1000.0)
    orientation, elongation, major, minor = orientation_and_elongation(
        metric, source_crs="EPSG:32643"
    )
    assert elongation == pytest.approx(4.0, rel=1e-6)
    assert major == pytest.approx(4.0, rel=1e-6)
    assert minor == pytest.approx(1.0, rel=1e-6)
    assert orientation % 180 == pytest.approx(0.0, abs=1e-6)


def test_a_square_is_not_elongated() -> None:
    assert orientation_and_elongation(box(0, 0, 1000, 1000), "EPSG:32643")[1] == (
        pytest.approx(1.0, rel=1e-6)
    )


def test_axes_are_real_distances_for_a_geographic_polygon() -> None:
    polygon = box(70.0, 22.0, 70.1, 22.02)  # ~10.3 km by ~2.2 km
    _, elongation, major, minor = orientation_and_elongation(polygon)
    assert major == pytest.approx(10.3, rel=0.05)
    assert minor == pytest.approx(2.2, rel=0.05)
    assert elongation == pytest.approx(major / minor, rel=1e-6)


# --------------------------------------------------------------------------- #
# spill objects
# --------------------------------------------------------------------------- #
def test_build_spill_object_is_valid_geojson() -> None:
    polygon = box(70.0, 22.0, 70.05, 22.03)
    feature = build_spill_object(polygon, SCENE_ID, ACQUIRED)

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert shape(feature["geometry"]).equals(polygon)
    json.dumps(feature)  # must survive serialisation


def test_spill_object_carries_scene_id_and_timestamp_through() -> None:
    """AIS attribution matches on this exact timestamp, so it must survive."""
    feature = build_spill_object(box(70.0, 22.0, 70.05, 22.03), SCENE_ID, ACQUIRED)
    properties = feature["properties"]
    assert properties["scene_id"] == SCENE_ID
    assert properties["acquisition_timestamp"] == ACQUIRED.isoformat()


def test_naive_timestamps_are_normalised_to_utc() -> None:
    naive = datetime(2023, 5, 14, 0, 33, 32)
    feature = build_spill_object(box(70.0, 22.0, 70.01, 22.01), SCENE_ID, naive)
    assert feature["properties"]["acquisition_timestamp"].endswith("+00:00")


def test_spill_object_properties_are_complete() -> None:
    feature = build_spill_object(
        box(70.0, 22.0, 70.05, 22.03), SCENE_ID, ACQUIRED, confidence=0.87
    )
    properties = feature["properties"]
    for key in (
        "spill_id", "scene_id", "acquisition_timestamp", "area_km2", "perimeter_km",
        "centroid_lon", "centroid_lat", "bbox", "orientation_deg", "elongation",
        "major_axis_km", "minor_axis_km", "mean_confidence", "estimated_volume_m3",
    ):
        assert key in properties, key

    assert properties["centroid_lon"] == pytest.approx(70.025)
    assert properties["centroid_lat"] == pytest.approx(22.015)
    assert properties["mean_confidence"] == pytest.approx(0.87)
    assert properties["area_km2"] > 0


def test_estimated_volume_follows_area_times_thickness() -> None:
    """1 um spread over 1 km² is 1e6 m² x 1e-6 m = 1 m³ of oil."""
    feature = build_spill_object(box(0.0, 0.0, 1000.0, 1000.0), SCENE_ID, ACQUIRED,
                                 crs="EPSG:32643")
    assert feature["properties"]["area_km2"] == pytest.approx(1.0)
    assert feature["properties"]["estimated_volume_m3"] == pytest.approx(1.0)


def test_spills_from_mask_describes_every_blob() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:60] = True
    mask[60:90, 60:95] = True

    features = spills_from_mask(mask, scene_transform(), SCENE_ID, ACQUIRED)

    assert len(features) == 2
    assert {f["properties"]["spill_id"] for f in features} == {
        f"{SCENE_ID}_spill_001", f"{SCENE_ID}_spill_002"
    }
    assert all(f["properties"]["scene_id"] == SCENE_ID for f in features)


def test_spills_from_mask_averages_the_confidence() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:40, 10:60] = True
    confidence = np.full((100, 100), 0.3, dtype=np.float32)
    confidence[mask] = 0.8

    features = spills_from_mask(mask, scene_transform(), SCENE_ID, ACQUIRED,
                                confidence=confidence)
    assert features[0]["properties"]["mean_confidence"] == pytest.approx(0.8, abs=0.01)


def _painted_oil_streak(shape, centre, semi_major=40.0, semi_minor=6.0, angle_deg=0.0):
    """A long, thin, hard-edged dark streak - shaped so it clears every
    look-alike-filter test (contrast, elongation, edge gradient) on its own.
    """
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dr = rows - centre[0]
    dc = cols - centre[1]
    theta = np.radians(angle_deg)
    major = dc * np.cos(theta) + dr * np.sin(theta)
    minor = -dc * np.sin(theta) + dr * np.cos(theta)
    return (major / semi_major) ** 2 + (minor / semi_minor) ** 2 <= 1.0


def test_spills_from_blobs_keeps_each_blobs_own_confidence() -> None:
    """The regression case: two spills in one scene must not collapse onto the
    same (wrongly re-averaged) confidence value.

    Both blobs are painted identically so they both clear the look-alike
    filter, and differ only in the confidence the (stand-in) model reported
    for each - which is exactly the case ``spills_from_mask``'s whole-mask
    average could not tell apart.
    """
    shape = (220, 220)
    sea_db = -10.0
    rng = np.random.default_rng(0)
    image = (sea_db + rng.normal(0.0, 0.2, size=shape)).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    blob_a = _painted_oil_streak(shape, (50, 60))
    blob_b = _painted_oil_streak(shape, (160, 150))
    image[blob_a] = sea_db - 9.0
    image[blob_b] = sea_db - 9.0
    mask[blob_a] = OIL_CLASS
    mask[blob_b] = OIL_CLASS

    confidence = np.full(shape, 0.3, dtype=np.float32)
    confidence[blob_a] = 0.95
    confidence[blob_b] = 0.55

    settings = load_settings()
    result = filter_lookalikes(mask, image, confidence=confidence, settings=settings)
    assert len(result.kept) == 2, "both painted streaks must pass the filter"

    features = spills_from_blobs(result, scene_transform(), SCENE_ID, ACQUIRED)

    assert len(features) == 2
    confidences = {round(f["properties"]["mean_confidence"], 2) for f in features}
    assert confidences == {0.95, 0.55}, (
        "each spill must carry its own blob's confidence, not the scene-wide average"
    )
    # sanity: this is not a coincidence of averaging over the whole mask
    whole_mask_average = float(confidence[mask == OIL_CLASS].mean())
    assert 0.55 < whole_mask_average < 0.95


def test_spills_from_blobs_matches_single_blob_behaviour_of_spills_from_mask() -> None:
    """A single-blob scene must still get the correct confidence either way."""
    shape = (160, 160)
    sea_db = -10.0
    image = np.full(shape, sea_db, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    blob = _painted_oil_streak(shape, (80, 80))
    image[blob] = sea_db - 9.0
    mask[blob] = OIL_CLASS

    confidence = np.full(shape, 0.3, dtype=np.float32)
    confidence[blob] = 0.8

    settings = load_settings()
    result = filter_lookalikes(mask, image, confidence=confidence, settings=settings)
    assert len(result.kept) == 1

    from_blobs = spills_from_blobs(result, scene_transform(), SCENE_ID, ACQUIRED)
    from_mask = spills_from_mask(
        result.mask, scene_transform(), SCENE_ID, ACQUIRED, confidence=confidence
    )

    assert len(from_blobs) == len(from_mask) == 1
    assert from_blobs[0]["properties"]["mean_confidence"] == pytest.approx(
        from_mask[0]["properties"]["mean_confidence"], abs=0.01
    )
    assert from_blobs[0]["properties"]["mean_confidence"] == pytest.approx(0.8, abs=0.01)


def test_spills_from_blobs_on_a_clean_scene_yields_nothing() -> None:
    settings = load_settings()
    result = filter_lookalikes(
        np.zeros((60, 60), dtype=np.uint8), np.full((60, 60), -10.0, dtype=np.float32),
        settings=settings,
    )
    assert spills_from_blobs(result, scene_transform(), SCENE_ID, ACQUIRED) == []


def test_a_clean_scene_produces_an_empty_collection() -> None:
    """Checked-and-clean must be distinguishable from never-processed."""
    features = spills_from_mask(np.zeros((50, 50), dtype=bool), scene_transform(),
                                SCENE_ID, ACQUIRED)
    collection = feature_collection(features, SCENE_ID, ACQUIRED)

    assert collection["type"] == "FeatureCollection"
    assert collection["features"] == []
    assert collection["properties"]["spill_count"] == 0
    assert collection["properties"]["scene_id"] == SCENE_ID
    json.dumps(collection)


def test_feature_collection_counts_and_serialises() -> None:
    features = [build_spill_object(box(70.0, 22.0, 70.01, 22.01), SCENE_ID, ACQUIRED)]
    collection = feature_collection(features, SCENE_ID, ACQUIRED, extra={"tiles": 25})
    assert collection["properties"]["spill_count"] == 1
    assert collection["properties"]["tiles"] == 25
    encoded = json.dumps(collection)
    assert json.loads(encoded)["properties"] == collection["properties"]
    assert json.dumps(json.loads(encoded)) == encoded
