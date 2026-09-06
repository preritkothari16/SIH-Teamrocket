"""Alert manager tests: each rule in isolation, then the combined decision.

Rule functions take plain synthetic inputs (confidence floats, wind speeds,
geometries) - none of them need a database or a wind dataset, only
:func:`evaluate_alert`'s registry-backed dedup step and
:func:`process_spill`'s wind lookup do.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from shapely.geometry import Point, box, mapping

from src.alerts.manager import (
    AlertsError,
    area_rule,
    confidence_rule,
    dedup_rule,
    evaluate_alert,
    exclusion_zone_rule,
    load_exclusion_zones,
    process_spill,
    wind_rule,
)
from src.alerts.registry import SpillRegistry
from src.characterization.spill_object import build_spill_object
from src.config import load_settings

ACQUIRED = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)
SPILL_POLYGON = box(70.0, 22.0, 70.05, 22.03)  # ~5km x 3km near 22N


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def registry(tmp_path: Path, settings) -> SpillRegistry:
    with SpillRegistry(path=tmp_path / "registry.sqlite3", settings=settings) as reg:
        yield reg


def make_spill(area_km2=None, confidence=0.9, spill_id=None, polygon=SPILL_POLYGON,
                when=ACQUIRED, scene_id="scene"):
    feature = build_spill_object(polygon, scene_id, when, spill_id=spill_id, confidence=confidence)
    if area_km2 is not None:
        feature["properties"]["area_km2"] = area_km2
    return feature


# --------------------------------------------------------------------------- #
# 1. confidence_rule
# --------------------------------------------------------------------------- #
def test_confidence_rule_passes_above_threshold() -> None:
    result = confidence_rule(0.8, min_confidence=0.7)
    assert result.passed
    assert result.name == "confidence"


def test_confidence_rule_fails_below_threshold() -> None:
    result = confidence_rule(0.5, min_confidence=0.7)
    assert not result.passed


def test_confidence_rule_passes_exactly_at_threshold() -> None:
    assert confidence_rule(0.7, min_confidence=0.7).passed


def test_confidence_rule_treats_missing_confidence_as_zero() -> None:
    """No confidence recorded must not accidentally pass a threshold check."""
    result = confidence_rule(None, min_confidence=0.7)
    assert not result.passed
    assert "0.00" in result.reason


# --------------------------------------------------------------------------- #
# 2. area_rule
# --------------------------------------------------------------------------- #
def test_area_rule_passes_above_minimum() -> None:
    assert area_rule(1.0, min_area_km2=0.1).passed


def test_area_rule_fails_below_minimum() -> None:
    assert not area_rule(0.05, min_area_km2=0.1).passed


def test_area_rule_passes_exactly_at_minimum() -> None:
    assert area_rule(0.1, min_area_km2=0.1).passed


# --------------------------------------------------------------------------- #
# 3. wind_rule - does not reject, only distinguishes active vs possible
# --------------------------------------------------------------------------- #
def test_wind_rule_passes_inside_the_detectable_band() -> None:
    assert wind_rule(6.0, min_speed_ms=2.0, max_speed_ms=12.0).passed


def test_wind_rule_fails_when_too_calm() -> None:
    result = wind_rule(0.5, min_speed_ms=2.0, max_speed_ms=12.0)
    assert not result.passed
    assert "outside" in result.reason


def test_wind_rule_fails_when_too_rough() -> None:
    result = wind_rule(20.0, min_speed_ms=2.0, max_speed_ms=12.0)
    assert not result.passed


def test_wind_rule_passes_at_the_band_edges() -> None:
    assert wind_rule(2.0, min_speed_ms=2.0, max_speed_ms=12.0).passed
    assert wind_rule(12.0, min_speed_ms=2.0, max_speed_ms=12.0).passed


def test_wind_rule_fails_when_wind_is_unknown() -> None:
    """Unavailable wind data must not be treated as a free pass."""
    result = wind_rule(None, min_speed_ms=2.0, max_speed_ms=12.0)
    assert not result.passed
    assert "unavailable" in result.reason


# --------------------------------------------------------------------------- #
# 4. exclusion_zone_rule
# --------------------------------------------------------------------------- #
def test_exclusion_zone_rule_passes_outside_all_zones() -> None:
    result = exclusion_zone_rule(Point(70.0, 22.0), exclusion_zones=[box(0.0, 0.0, 1.0, 1.0)])
    assert result.passed


def test_exclusion_zone_rule_fails_inside_a_zone() -> None:
    zone = box(69.9, 21.9, 70.1, 22.1)
    result = exclusion_zone_rule(Point(70.0, 22.0), exclusion_zones=[zone])
    assert not result.passed
    assert "exclusion zone" in result.reason


def test_exclusion_zone_rule_passes_with_no_zones_configured() -> None:
    assert exclusion_zone_rule(Point(70.0, 22.0), exclusion_zones=[]).passed


def test_exclusion_zone_rule_checks_every_zone_not_just_the_first() -> None:
    far_zone = box(0.0, 0.0, 1.0, 1.0)
    hit_zone = box(69.9, 21.9, 70.1, 22.1)
    result = exclusion_zone_rule(Point(70.0, 22.0), exclusion_zones=[far_zone, hit_zone])
    assert not result.passed


def test_load_exclusion_zones_with_no_path_configured_yields_nothing(settings) -> None:
    assert load_exclusion_zones(settings=settings) == []


def test_load_exclusion_zones_reads_a_feature_collection(tmp_path: Path, settings) -> None:
    zone = box(69.9, 21.9, 70.1, 22.1)
    path = tmp_path / "zones.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": mapping(zone), "properties": {}}],
    }))
    zones = load_exclusion_zones(path=path, settings=settings)
    assert len(zones) == 1
    assert zones[0].equals(zone)


def test_load_exclusion_zones_reports_a_missing_file(tmp_path: Path, settings) -> None:
    with pytest.raises(AlertsError, match="not found"):
        load_exclusion_zones(path=tmp_path / "missing.geojson", settings=settings)


# --------------------------------------------------------------------------- #
# 5. dedup_rule
# --------------------------------------------------------------------------- #
def _record(spill_id, polygon, when):
    from src.alerts.registry import SpillRecord
    return SpillRecord(
        spill_id=spill_id, scene_id="s0", geometry=polygon,
        centroid_lon=polygon.centroid.x, centroid_lat=polygon.centroid.y,
        area_km2=1.0, first_seen=when, last_updated=when, status="active",
    )


def test_dedup_rule_matches_an_overlapping_polygon() -> None:
    existing = _record("spill_a", box(70.0, 22.0, 70.05, 22.03), ACQUIRED)
    overlapping = box(70.02, 22.01, 70.07, 22.05)
    match = dedup_rule(overlapping, recent_events=[existing], buffer_km=5.0)
    assert match is not None and match.spill_id == "spill_a"


def test_dedup_rule_matches_a_nearby_non_overlapping_polygon_within_the_buffer() -> None:
    """A slick drifts between detections - dedup must catch that, not just
    an exact overlap."""
    existing = _record("spill_a", box(70.0, 22.0, 70.01, 22.01), ACQUIRED)
    # ~a few km east, not touching the original box
    drifted = box(70.05, 22.0, 70.06, 22.01)
    match = dedup_rule(drifted, recent_events=[existing], buffer_km=10.0)
    assert match is not None and match.spill_id == "spill_a"


def test_dedup_rule_does_not_match_beyond_the_buffer() -> None:
    existing = _record("spill_a", box(70.0, 22.0, 70.01, 22.01), ACQUIRED)
    far_away = box(80.0, 10.0, 80.01, 10.01)
    assert dedup_rule(far_away, recent_events=[existing], buffer_km=5.0) is None


def test_dedup_rule_returns_none_with_no_recent_events() -> None:
    assert dedup_rule(SPILL_POLYGON, recent_events=[], buffer_km=5.0) is None


def test_dedup_rule_picks_the_first_match_when_several_exist() -> None:
    a = _record("spill_a", box(70.0, 22.0, 70.05, 22.03), ACQUIRED)
    b = _record("spill_b", box(70.0, 22.0, 70.05, 22.03), ACQUIRED)
    match = dedup_rule(box(70.01, 22.01, 70.04, 22.02), recent_events=[a, b], buffer_km=5.0)
    assert match.spill_id == "spill_a"


# --------------------------------------------------------------------------- #
# combined decision - evaluate_alert
# --------------------------------------------------------------------------- #
def test_evaluate_alert_registers_a_new_active_spill(registry, settings) -> None:
    spill = make_spill()
    decision = evaluate_alert(spill, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)

    assert decision.alert is True
    assert decision.status == "active"
    assert decision.is_new is True
    assert decision.spill_id is not None
    assert [r.name for r in decision.rules] == ["confidence", "area", "wind", "exclusion_zone", "dedup"]
    assert registry.get(decision.spill_id) is not None


def test_evaluate_alert_rejects_low_confidence_without_registering(registry, settings) -> None:
    spill = make_spill(confidence=0.3)
    decision = evaluate_alert(spill, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)

    assert decision.alert is False
    assert decision.status == "rejected"
    assert decision.spill_id is None
    assert [r.name for r in decision.rules] == ["confidence"]
    assert registry.all() == []


def test_evaluate_alert_rejects_a_too_small_spill(registry, settings) -> None:
    spill = make_spill(area_km2=0.001)
    decision = evaluate_alert(spill, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)

    assert decision.alert is False
    assert decision.status == "rejected"
    assert [r.name for r in decision.rules] == ["confidence", "area"]


def test_evaluate_alert_marks_possible_when_wind_is_out_of_range(registry, settings) -> None:
    spill = make_spill()
    decision = evaluate_alert(spill, registry, wind_speed_ms=0.5, exclusion_zones=[], settings=settings)

    assert decision.alert is True
    assert decision.status == "possible"
    assert registry.get(decision.spill_id).status == "possible"


def test_evaluate_alert_rejects_a_spill_in_an_exclusion_zone(registry, settings) -> None:
    zone = box(69.9, 21.9, 70.1, 22.1)  # covers SPILL_POLYGON's centroid
    spill = make_spill()
    decision = evaluate_alert(spill, registry, wind_speed_ms=6.0, exclusion_zones=[zone], settings=settings)

    assert decision.alert is False
    assert decision.status == "rejected"
    assert registry.all() == []


def test_evaluate_alert_updates_a_matching_recent_event_instead_of_creating_new(
    registry, settings
) -> None:
    first = make_spill(spill_id="scene1_spill_001", when=ACQUIRED, scene_id="scene1")
    d1 = evaluate_alert(first, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)

    later = make_spill(
        spill_id="scene2_spill_001",
        polygon=box(70.01, 22.01, 70.06, 22.04),  # overlaps SPILL_POLYGON
        when=ACQUIRED + timedelta(hours=2),
        scene_id="scene2",
    )
    d2 = evaluate_alert(later, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)

    assert d2.is_new is False
    assert d2.spill_id == d1.spill_id
    assert len(registry.all()) == 1  # updated, not a second row
    assert registry.get(d1.spill_id).scene_id == "scene2"  # latest detection wins


def test_evaluate_alert_creates_a_new_spill_when_outside_the_dedup_window(
    registry, settings
) -> None:
    """An event older than dedup_window_days must not suppress a new one."""
    old = make_spill(spill_id="scene1_spill_001", when=ACQUIRED, scene_id="scene1")
    evaluate_alert(old, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)

    much_later = make_spill(
        spill_id="scene2_spill_001",
        when=ACQUIRED + timedelta(days=settings.alerts.dedup_window_days + 1),
        scene_id="scene2",
    )
    decision = evaluate_alert(
        much_later, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings
    )
    assert decision.is_new is True
    assert len(registry.all()) == 2


def test_evaluate_alert_to_dict_is_json_serialisable(registry, settings) -> None:
    import json as json_mod

    spill = make_spill()
    decision = evaluate_alert(spill, registry, wind_speed_ms=6.0, exclusion_zones=[], settings=settings)
    encoded = json_mod.dumps(decision.to_dict())
    assert json_mod.loads(encoded)["alert"] is True


# --------------------------------------------------------------------------- #
# process_spill - the wind.py integration wrapper
# --------------------------------------------------------------------------- #
def test_process_spill_degrades_gracefully_with_no_wind_source(registry, settings) -> None:
    """No wind dataset configured anywhere in this environment - process_spill
    must still decide (as "possible", since wind is then unknown), not raise.
    """
    spill = make_spill()
    decision = process_spill(spill, registry, exclusion_zones=[], settings=settings)
    assert decision.alert is True
    assert decision.status == "possible"
    wind_result = next(r for r in decision.rules if r.name == "wind")
    assert "unavailable" in wind_result.reason
