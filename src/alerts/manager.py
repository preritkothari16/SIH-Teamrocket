"""Rule-based alert decision for one spill.

Each rule below is its own small, independently testable function. The
combined decision - and, crucially, **which rule(s) drove it** (the whole
point of a demo where you have to explain *why* something did or didn't
alert) - is assembled by :func:`evaluate_alert`.

Evaluation order:

1. :func:`confidence_rule` - below ``alerts.min_confidence`` rejects outright.
2. :func:`area_rule` - below ``alerts.min_area_km2`` rejects outright.
3. :func:`wind_rule` - outside the detectable wind band (or unknown) does
   **not** reject; it registers the spill with status ``"possible"`` instead
   of ``"active"``, since the slick is real either way, just less certain to
   be visible/stable in this image.
4. :func:`exclusion_zone_rule` - inside a known natural seep or platform
   rejects outright; a detector cannot tell those apart from a real
   discharge by pixels alone.
5. :func:`dedup_rule` - a recent event (``alerts.dedup_window_days``) whose
   geometry overlaps or drifts within ``alerts.dedup_buffer_km`` of this one
   is updated in the registry instead of minting a new spill ID.

:func:`evaluate_alert` is the pure decision (wind speed is a plain
``Optional[float]`` argument, so it needs no wind dataset to test).
:func:`process_spill` is the convenience entry point that actually calls
:func:`src.env_data.service.get_environment` for the spill's centroid/time
first, tolerating the lookup being unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.alerts.registry import SpillRecord, SpillRegistry
from src.characterization.spill_object import WGS84, to_equal_area
from src.config import Settings, get_settings
from src.env_data.service import get_environment
from src.ingestion.types import as_utc


class AlertsError(RuntimeError):
    """A spill could not be evaluated, or exclusion zones could not be read."""


@dataclass
class RuleResult:
    """One rule's verdict. Kept even when it did not reject anything -
    "which rule(s) drove it" needs the full trail, not just the failures."""

    name: str
    passed: bool
    reason: str


@dataclass
class AlertDecision:
    """The alert manager's full decision for one spill."""

    alert: bool
    status: str  # "active" | "possible" | "rejected"
    spill_id: Optional[str]
    is_new: bool
    rules: List[RuleResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert": self.alert,
            "status": self.status,
            "spill_id": self.spill_id,
            "is_new": self.is_new,
            "rules": [{"name": r.name, "passed": r.passed, "reason": r.reason} for r in self.rules],
        }


# --------------------------------------------------------------------------- #
# individual rules
# --------------------------------------------------------------------------- #
def confidence_rule(confidence: Optional[float], min_confidence: float) -> RuleResult:
    """Reject a spill the detector itself was not confident about."""
    value = confidence if confidence is not None else 0.0
    passed = value >= min_confidence
    return RuleResult(
        name="confidence",
        passed=passed,
        reason=f"confidence {value:.2f} {'>=' if passed else '<'} {min_confidence:.2f}",
    )


def area_rule(area_km2: float, min_area_km2: float) -> RuleResult:
    """Reject a speck too small to be a reportable spill."""
    passed = area_km2 >= min_area_km2
    return RuleResult(
        name="area",
        passed=passed,
        reason=f"area {area_km2:.3f} km² {'>=' if passed else '<'} {min_area_km2:.3f} km²",
    )


def wind_rule(
    wind_speed_ms: Optional[float], min_speed_ms: float, max_speed_ms: float
) -> RuleResult:
    """Wind outside the range a dark-patch detector can trust - too calm and
    low-wind cells look identical to oil, too rough and a real slick gets
    mixed away - does not reject the spill. It only downgrades the caller's
    status to "possible"; the caller decides that from ``passed``, this rule
    does not touch the registry itself.
    """
    if wind_speed_ms is None:
        return RuleResult(name="wind", passed=False, reason="wind data unavailable")
    passed = min_speed_ms <= wind_speed_ms <= max_speed_ms
    band = f"[{min_speed_ms:.1f}, {max_speed_ms:.1f}] m/s"
    return RuleResult(
        name="wind",
        passed=passed,
        reason=f"wind {wind_speed_ms:.1f} m/s {'within' if passed else 'outside'} {band}",
    )


def exclusion_zone_rule(centroid: Point, exclusion_zones: List[BaseGeometry]) -> RuleResult:
    """Reject a spill sitting on a known natural seep or platform - a
    detector cannot tell those apart from a real discharge by pixels alone.
    """
    hit = next((zone for zone in exclusion_zones if zone.contains(centroid)), None)
    return RuleResult(
        name="exclusion_zone",
        passed=hit is None,
        reason="inside a known exclusion zone" if hit is not None else "outside all exclusion zones",
    )


def _distance_km(a: BaseGeometry, b: BaseGeometry) -> float:
    """Real (equal-area) distance in km between two WGS84 geometries."""
    if a.intersects(b):
        return 0.0
    projected_a, equal_area = to_equal_area(a, WGS84)
    transformer = Transformer.from_crs(WGS84, equal_area, always_xy=True)
    projected_b = shapely_transform(transformer.transform, b)
    return projected_a.distance(projected_b) / 1000.0


def dedup_rule(
    new_geometry: BaseGeometry, recent_events: List[SpillRecord], buffer_km: float
) -> Optional[SpillRecord]:
    """The first recent event this spill's geometry overlaps, or lies within
    ``buffer_km`` of - a slick drifts between successive detections, so a
    plausible drift buffer (not just an exact-overlap test) is what actually
    recognises the same event reappearing in a later scene.
    """
    for event in recent_events:
        if new_geometry.intersects(event.geometry) or _distance_km(
            new_geometry, event.geometry
        ) <= buffer_km:
            return event
    return None


# --------------------------------------------------------------------------- #
# exclusion zones
# --------------------------------------------------------------------------- #
def load_exclusion_zones(
    path: Optional[Path] = None, settings: Optional[Settings] = None
) -> List[BaseGeometry]:
    """Known natural-seep/platform polygons from a GeoJSON Feature or
    FeatureCollection. No configured path -> no exclusion zones (an empty
    list, not an error - most demo scenes will have none).
    """
    settings = settings or get_settings()
    path = path if path is not None else settings.alerts.exclusion_zones_path
    if path is None:
        return []

    resolved = settings.paths.resolve(Path(path))
    if not resolved.is_file():
        raise AlertsError(f"exclusion zones file not found at {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))

    if data.get("type") == "FeatureCollection":
        return [shape(f["geometry"]) for f in data.get("features", [])]
    if data.get("type") == "Feature":
        return [shape(data["geometry"])]
    return [shape(data)]


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #
def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return as_utc(value)
    return as_utc(datetime.fromisoformat(str(value)))


def evaluate_alert(
    spill: Dict[str, Any],
    registry: SpillRegistry,
    wind_speed_ms: Optional[float],
    exclusion_zones: Optional[List[BaseGeometry]] = None,
    settings: Optional[Settings] = None,
) -> AlertDecision:
    """Decide whether ``spill`` (a GeoJSON Feature, as
    :func:`src.characterization.spill_object.build_spill_object` produces)
    should raise an alert, and register/update it in ``registry`` if so.

    Pure with respect to wind: pass ``wind_speed_ms`` directly rather than
    having this function call :mod:`src.env_data.wind` itself, so every rule
    here is testable with plain synthetic inputs. :func:`process_spill` is
    the convenience wrapper that does the wind lookup.
    """
    settings = settings or get_settings()
    cfg = settings.alerts
    properties = spill.get("properties") or {}
    geometry = shape(spill["geometry"])
    centroid = geometry.centroid

    area_km2 = float(properties.get("area_km2", 0.0))
    confidence = properties.get("mean_confidence")
    acquisition_time = _parse_timestamp(properties.get("acquisition_timestamp"))
    scene_id = properties.get("scene_id")
    candidate_spill_id = properties.get("spill_id") or f"spill_{uuid4().hex[:12]}"

    if exclusion_zones is None:
        exclusion_zones = load_exclusion_zones(settings=settings)

    rules: List[RuleResult] = []

    conf = confidence_rule(confidence, cfg.min_confidence)
    rules.append(conf)
    if not conf.passed:
        return AlertDecision(alert=False, status="rejected", spill_id=None, is_new=False, rules=rules)

    area = area_rule(area_km2, cfg.min_area_km2)
    rules.append(area)
    if not area.passed:
        return AlertDecision(alert=False, status="rejected", spill_id=None, is_new=False, rules=rules)

    wind = wind_rule(wind_speed_ms, cfg.wind_min_speed_ms, cfg.wind_max_speed_ms)
    rules.append(wind)
    status = "active" if wind.passed else "possible"

    exclusion = exclusion_zone_rule(centroid, exclusion_zones)
    rules.append(exclusion)
    if not exclusion.passed:
        return AlertDecision(alert=False, status="rejected", spill_id=None, is_new=False, rules=rules)

    since = acquisition_time - timedelta(days=cfg.dedup_window_days)
    recent_events = registry.recent(since)
    match = dedup_rule(geometry, recent_events, cfg.dedup_buffer_km)
    rules.append(
        RuleResult(
            name="dedup",
            passed=match is None,
            reason=(
                f"matches existing event {match.spill_id}"
                if match is not None
                else "no matching recent event"
            ),
        )
    )

    if match is not None:
        registry.update(
            match.spill_id, geometry=geometry, centroid_lon=centroid.x, centroid_lat=centroid.y,
            area_km2=area_km2, seen_at=acquisition_time, status=status, scene_id=scene_id,
        )
        return AlertDecision(alert=True, status=status, spill_id=match.spill_id, is_new=False, rules=rules)

    registry.register(
        candidate_spill_id, geometry=geometry, centroid_lon=centroid.x, centroid_lat=centroid.y,
        area_km2=area_km2, seen_at=acquisition_time, status=status, scene_id=scene_id,
    )
    return AlertDecision(alert=True, status=status, spill_id=candidate_spill_id, is_new=True, rules=rules)


def process_spill(
    spill: Dict[str, Any],
    registry: SpillRegistry,
    exclusion_zones: Optional[List[BaseGeometry]] = None,
    settings: Optional[Settings] = None,
) -> AlertDecision:
    """Convenience entry point: fetches wind for the spill's centroid/time via
    :func:`src.env_data.service.get_environment`, then calls
    :func:`evaluate_alert`.

    A failed wind lookup (no dataset configured, point outside the grid, ...)
    is not fatal - ``get_environment()`` already degrades it to ``None``, and
    :func:`wind_rule` treats an unknown speed the same as an out-of-range one.
    """
    settings = settings or get_settings()
    properties = spill.get("properties") or {}
    geometry = shape(spill["geometry"])
    centroid = geometry.centroid
    acquisition_time = _parse_timestamp(properties.get("acquisition_timestamp"))

    environment = get_environment(centroid.y, centroid.x, acquisition_time, settings=settings)
    wind = environment["wind"]
    wind_speed_ms = wind.vector.speed_ms if wind is not None else None

    return evaluate_alert(
        spill, registry, wind_speed_ms=wind_speed_ms,
        exclusion_zones=exclusion_zones, settings=settings,
    )


__all__ = [
    "AlertsError",
    "RuleResult",
    "AlertDecision",
    "confidence_rule",
    "area_rule",
    "wind_rule",
    "exclusion_zone_rule",
    "dedup_rule",
    "load_exclusion_zones",
    "evaluate_alert",
    "process_spill",
]
