"""Weighted attribution score per AIS candidate vessel.

Combines four independent pieces of evidence into one 0-1 score:

1. :func:`spatial_score`   - how close the candidate's CPA was to the slick
2. :func:`temporal_score`  - how plausible the timing of that CPA was
3. :func:`alignment_score` - how well the vessel's heading matches the
   slick's own axis
4. :func:`type_prior`      - how likely this class of vessel is to cause a
   spill of this kind at all

``alignment_score`` is the **strongest single discriminator** of the four,
per this project's design notes: a slick is laid down along whatever path the
vessel took, so a heading that runs right along the slick's long axis is
close to definitive on its own, while the other three are corroborating
rather than conclusive - many vessels pass close to a spill, many transits
happen at a plausible time, and many vessels in a shipping lane are tankers.
``config.yaml``'s ``attribution.weight_alignment`` reflects that (it is the
largest of the four weights by default), but nothing here special-cases it
beyond the weight - the combination is a plain weighted sum.

Each candidate also gets a short, human-readable explanation built from the
same four numbers, e.g. ``"1.2 km CPA, 5h before image, track within 8° of
slick axis, tanker"`` - this is required output, not a debugging aid: the
whole point of scoring vessels instead of just ranking by distance is to be
able to say *why* one was ranked above another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from pyproj import Transformer
from shapely.geometry import Point

from src.ais.filter import CandidateVessel
from src.characterization.spill_object import WGS84, to_equal_area
from src.config import AttributionConfig, Settings, get_settings
from src.drift.hindcast import OriginSnapshot, nearest_snapshot

#: src.characterization.spill_object.orientation_and_elongation() reports its
#: angle counter-clockwise from east (standard math convention), mod 180 -
#: verified empirically, since its own docstring says "clockwise" but the
#: underlying atan2(dy, dx) call is CCW. Converting to a compass bearing
#: (clockwise from north) needs (90 - angle), not a straight pass-through.


def slick_axis_bearing(orientation_deg: Optional[float]) -> Optional[float]:
    """A slick's ``orientation_deg`` (math angle, CCW from east) as a compass
    bearing (clockwise from north), both mod 180 since an axis is a line, not
    a direction - a slick has no "forward"."""
    if orientation_deg is None:
        return None
    return (90.0 - orientation_deg) % 180.0


def _angle_to_axis(heading_deg: float, axis_bearing_deg: float) -> float:
    """Smallest angle in [0, 90] between a directional heading and an
    undirected axis bearing - a vessel heading straight along a slick's axis
    scores the same whichever of the two ways along it it was going."""
    diff = abs((heading_deg % 180.0) - (axis_bearing_deg % 180.0))
    return min(diff, 180.0 - diff)


def spatial_score(cpa_distance_km: float, scale_km: float) -> float:
    """Closer is higher: exponential decay with CPA distance.

    1.0 at zero distance, ~0.37 at ``scale_km``, ~0.05 by 3x ``scale_km``.
    Smooth rather than a hard cutoff - the hard cutoff already happened in
    :func:`src.ais.filter.filter_candidates` (candidates outside the search
    buffer never reach here); this only has to rank what is already inside.
    """
    return math.exp(-max(cpa_distance_km, 0.0) / scale_km)


def temporal_score(
    cpa_time: datetime, acquisition_time: datetime,
    optimal_lag_hours: float, scale_hours: float,
) -> float:
    """Recent-before-acquisition is better, but very recent is penalised.

    An explicit peaked curve (a Gaussian bump centred on
    ``optimal_lag_hours``), not a straight line, because a straight line can
    only prefer "more recent" or "less recent" - it cannot fall off on *both*
    sides of an optimum. Oil takes time to spread into a slick large enough
    for SAR to resolve, so a CPA seconds before the image is a *weaker* match
    than one a few hours before it; a CPA from days earlier is weaker again,
    since the vessel (and the slick) could be anywhere by then.

    A CPA at or after the acquisition time cannot explain a slick that was
    already there when the image was taken, so it scores 0.
    """
    lag_hours = (acquisition_time - cpa_time).total_seconds() / 3600.0
    if lag_hours < 0:
        return 0.0
    return math.exp(-0.5 * ((lag_hours - optimal_lag_hours) / scale_hours) ** 2)


def alignment_score(
    vessel_heading_at_cpa: Optional[float], slick_major_axis_bearing: Optional[float],
) -> float:
    """1.0 when the vessel's heading at CPA runs exactly along the slick's
    major axis (either direction), 0.0 when it crosses it at a right angle.

    Both angles are compass bearings in degrees, clockwise from north; the
    axis is undirected (mod 180) since a slick has no "forward". Neutral
    (0.5) when either angle is unknown, rather than penalising a vessel for
    AIS not reporting its heading.
    """
    if vessel_heading_at_cpa is None or slick_major_axis_bearing is None:
        return 0.5
    diff = _angle_to_axis(vessel_heading_at_cpa, slick_major_axis_bearing)
    return 1.0 - diff / 90.0


def _distance_to_polygon_km(lon: float, lat: float, polygon) -> float:
    """Great-enough-circle distance from ``(lon, lat)`` to ``polygon``, in an
    equal-area CRS centred on the polygon itself - the same one-transformer
    pattern :func:`src.ais.filter._project` uses, needed here because
    ``use_hindcasting`` measures against a *different* polygon (the nearest
    hindcast snapshot) per candidate rather than the spill's fixed one.
    """
    projected_polygon, equal_area = to_equal_area(polygon, WGS84)
    transformer = Transformer.from_crs(WGS84, equal_area, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return projected_polygon.distance(Point(x, y)) / 1000.0


def _effective_cpa_distance_km(
    candidate: CandidateVessel,
    config: AttributionConfig,
    hindcast_corridor: Optional[Sequence[OriginSnapshot]],
) -> float:
    """The CPA distance :func:`spatial_score` actually scores against.

    With ``use_hindcasting`` off (the default) - or when no corridor was
    supplied, or a candidate carries no CPA position - this is just
    ``candidate.cpa_distance_km``, the static distance to the spill's fixed,
    final polygon (step 2.3's original path, unchanged). With it on, distance
    is instead measured from the candidate's actual position at CPA to the
    hindcast polygon nearest that CPA time - the oil's estimated footprint
    back when the vessel was actually there, not where it drifted to by
    acquisition time.
    """
    if not config.use_hindcasting or not hindcast_corridor or candidate.position_at_cpa is None:
        return candidate.cpa_distance_km
    snapshot = nearest_snapshot(hindcast_corridor, candidate.cpa_time)
    lon, lat = candidate.position_at_cpa
    return _distance_to_polygon_km(lon, lat, snapshot.polygon)


def type_prior(vessel_type: Optional[str], config: AttributionConfig) -> float:
    """Configurable per-type weight: tankers/cargo score higher than fishing/
    leisure by default, since the former carry the bulk oil that causes a
    spill of this scale and the latter rarely do. ``config.type_priors`` is
    the single place that table lives - see ``attribution.type_priors`` in
    ``config.yaml``.
    """
    if not vessel_type:
        return config.default_type_prior
    return config.type_priors.get(vessel_type.strip().lower(), config.default_type_prior)


@dataclass
class ScoredCandidate:
    """One candidate vessel, its combined score, and why it got it."""

    candidate: CandidateVessel
    score: float
    spatial: float
    temporal: float
    alignment: float
    type_prior: float
    explanation: str
    #: distance actually scored by `spatial` - equal to
    #: `candidate.cpa_distance_km` unless `use_hindcasting` swapped in the
    #: nearest hindcast polygon (see `_effective_cpa_distance_km`).
    cpa_distance_km: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.candidate.mmsi,
            "vessel_name": self.candidate.vessel_name,
            "vessel_type": self.candidate.vessel_type,
            "cpa_distance_km": round(self.cpa_distance_km, 3),
            "cpa_time": self.candidate.cpa_time.isoformat(),
            "score": round(self.score, 4),
            "spatial_score": round(self.spatial, 4),
            "temporal_score": round(self.temporal, 4),
            "alignment_score": round(self.alignment, 4),
            "type_prior": round(self.type_prior, 4),
            "explanation": self.explanation,
        }


def build_explanation(
    candidate: CandidateVessel, acquisition_time: datetime,
    slick_bearing: Optional[float],
    cpa_distance_km: Optional[float] = None,
) -> str:
    """A short, human-readable summary of the same four factors that scored
    this candidate, e.g. ``"1.2 km CPA, 5h before image, track within 8° of
    slick axis, tanker"``.

    ``cpa_distance_km`` defaults to ``candidate.cpa_distance_km`` but is
    overridden by :func:`score_candidate` when ``use_hindcasting`` swapped in
    a different distance, so the explanation names the number that was
    actually scored, not the static one.
    """
    if cpa_distance_km is None:
        cpa_distance_km = candidate.cpa_distance_km
    lag_hours = (acquisition_time - candidate.cpa_time).total_seconds() / 3600.0
    parts = [f"{cpa_distance_km:.1f} km CPA"]
    parts.append(
        f"{lag_hours:.0f}h before image" if lag_hours >= 0
        else f"{abs(lag_hours):.0f}h after image"
    )

    if candidate.heading_at_cpa is not None and slick_bearing is not None:
        diff = _angle_to_axis(candidate.heading_at_cpa, slick_bearing)
        parts.append(f"track within {diff:.0f}° of slick axis")
    else:
        parts.append("heading unknown")

    parts.append((candidate.vessel_type or "unknown type").lower())
    return ", ".join(parts)


def score_candidate(
    candidate: CandidateVessel,
    acquisition_time: datetime,
    slick_orientation_deg: Optional[float],
    settings: Optional[Settings] = None,
    hindcast_corridor: Optional[Sequence[OriginSnapshot]] = None,
) -> ScoredCandidate:
    """Score one candidate against a spill's acquisition time and orientation.

    ``hindcast_corridor`` (see :func:`src.drift.hindcast.hindcast_origin`) is
    only consulted when ``settings.attribution.use_hindcasting`` is true;
    otherwise scoring is exactly the step 2.3 static-buffer path.
    """
    settings = settings or get_settings()
    cfg = settings.attribution
    bearing = slick_axis_bearing(slick_orientation_deg)
    cpa_distance_km = _effective_cpa_distance_km(candidate, cfg, hindcast_corridor)

    spatial = spatial_score(cpa_distance_km, cfg.spatial_scale_km)
    temporal = temporal_score(
        candidate.cpa_time, acquisition_time,
        cfg.temporal_optimal_lag_hours, cfg.temporal_scale_hours,
    )
    alignment = alignment_score(candidate.heading_at_cpa, bearing)
    prior = type_prior(candidate.vessel_type, cfg)

    score = (
        cfg.weight_spatial * spatial
        + cfg.weight_temporal * temporal
        + cfg.weight_alignment * alignment
        + cfg.weight_type * prior
    )

    return ScoredCandidate(
        candidate=candidate, score=score, spatial=spatial, temporal=temporal,
        alignment=alignment, type_prior=prior,
        explanation=build_explanation(candidate, acquisition_time, bearing, cpa_distance_km),
        cpa_distance_km=cpa_distance_km,
    )


def score_candidates(
    candidates: List[CandidateVessel],
    acquisition_time: datetime,
    slick_orientation_deg: Optional[float],
    settings: Optional[Settings] = None,
    hindcast_corridor: Optional[Sequence[OriginSnapshot]] = None,
) -> List[ScoredCandidate]:
    """Score every candidate and rank them highest-score first."""
    settings = settings or get_settings()
    scored = [
        score_candidate(c, acquisition_time, slick_orientation_deg, settings, hindcast_corridor)
        for c in candidates
    ]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


__all__ = [
    "ScoredCandidate",
    "slick_axis_bearing",
    "spatial_score",
    "temporal_score",
    "alignment_score",
    "type_prior",
    "build_explanation",
    "score_candidate",
    "score_candidates",
]
