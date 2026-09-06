"""Reduce reconstructed AIS tracks to a candidate list for one spill.

Every :class:`~src.ais.tracks.VesselTrack` from a spill's space-time query is
scored by how close it actually came to the spill's footprint - the closest
point of approach (CPA) - and dropped if that closest approach is still
outside the search buffer (the resampled track can pass near the spill
between two raw pings that themselves land outside the buffer, or vice versa;
CPA is the geometrically honest answer either way) or if the vessel never
appears to have been moving, which is what a moored or anchored vessel looks
like in AIS and is not a spiller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from src.ais.query import SpillLike, spill_geometry_and_time
from src.ais.tracks import VesselTrack
from src.characterization.spill_object import WGS84, to_equal_area
from src.config import Settings, get_settings

KNOTS_PER_KMH = 1.0 / 1.852

#: AIS reports 511 for heading when the sensor has none to report.
HEADING_NOT_AVAILABLE = 511.0


@dataclass
class CandidateVessel:
    """One vessel that survived the CPA and stationarity filters."""

    mmsi: int
    cpa_distance_km: float
    cpa_time: datetime
    track: pd.DataFrame
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None
    #: compass bearing (deg clockwise from north) at CPA, for alignment_score.
    heading_at_cpa: Optional[float] = None


def _project(geometry: BaseGeometry, source_crs: str = WGS84):
    """(project(lon, lat) -> (x, y), projected geometry) in an equal-area CRS
    centred on ``geometry``, so distances downstream are real metres.

    Reuses :func:`to_equal_area` for the CRS choice and the geometry's own
    projection; the transformer is rebuilt from its resulting CRS (rather than
    also returned by :func:`to_equal_area`) because callers here need it again
    afterward, to project extra points (a track's positions) into that same
    CRS - not just the one geometry.
    """
    projected_geometry, equal_area = to_equal_area(geometry, source_crs)
    transformer = Transformer.from_crs(source_crs, equal_area, always_xy=True)
    return transformer, projected_geometry


def closest_point_of_approach(
    track: VesselTrack, geometry: BaseGeometry
) -> "tuple[float, datetime]":
    """(distance_km, time) of the track's closest approach to ``geometry``.

    Measured against the track's *resampled* positions
    (:attr:`VesselTrack.resampled`), not the raw pings - the whole reason
    :mod:`src.ais.tracks` interpolates is so a closest approach that happened
    between two sparse pings is not missed. Distance is computed in an
    equal-area projection centred on the spill, never in raw degrees.
    """
    transformer, projected_geometry = _project(geometry)
    return _cpa_with_projection(track, transformer, projected_geometry)


def _cpa_with_projection(
    track: VesselTrack, transformer: Transformer, projected_geometry: BaseGeometry
) -> "tuple[float, datetime]":
    """The actual CPA computation, given an already-projected spill geometry.

    Split out so :func:`filter_candidates` can build the (spill-geometry,
    transformer) pair once and reuse it across every candidate vessel,
    instead of re-projecting the same, loop-invariant spill geometry once per
    candidate.
    """
    positions = track.resampled
    xs, ys = transformer.transform(positions["lon"].to_numpy(), positions["lat"].to_numpy())
    distances = np.array([projected_geometry.distance(Point(x, y)) for x, y in zip(xs, ys)])

    index = int(np.argmin(distances))
    cpa_time = positions["timestamp"].iloc[index]
    if isinstance(cpa_time, pd.Timestamp):
        cpa_time = cpa_time.to_pydatetime()
    return float(distances[index]) / 1000.0, cpa_time


def _speed_from_displacement_knots(track: VesselTrack) -> np.ndarray:
    """Fallback ground speed (knots) from consecutive resampled positions,
    for a track whose SOG field is entirely missing."""
    positions = track.resampled
    if len(positions) < 2:
        return np.array([])

    transformer, _ = _project(Point(positions["lon"].iloc[0], positions["lat"].iloc[0]))
    xs, ys = transformer.transform(positions["lon"].to_numpy(), positions["lat"].to_numpy())
    dist_m = np.hypot(np.diff(xs), np.diff(ys))
    dt_hours = np.diff(positions["timestamp"].astype("int64")) / 1e9 / 3600.0
    with np.errstate(invalid="ignore", divide="ignore"):
        speed_kmh = np.where(dt_hours > 0, dist_m / 1000.0 / dt_hours, 0.0)
    return speed_kmh * KNOTS_PER_KMH


def is_stationary(track: VesselTrack, min_moving_speed_knots: float) -> bool:
    """True if the vessel never appears to move faster than a moored vessel
    drifting at anchor - AIS's own SOG when it is reported, otherwise ground
    speed computed from the interpolated track.
    """
    sog = track.points["sog"].dropna()
    if not sog.empty:
        return bool((sog <= min_moving_speed_knots).all())

    speeds = _speed_from_displacement_knots(track)
    if speeds.size == 0:
        return False  # a single ping with no SOG - not enough to call it moored
    return bool((speeds <= min_moving_speed_knots).all())


def _valid_heading(value: Optional[float]) -> Optional[float]:
    """A raw AIS heading/COG value, or None for a missing/sentinel one."""
    if value is None or not math.isfinite(value):
        return None
    if 0.0 <= value < 360.0 and value != HEADING_NOT_AVAILABLE:
        return float(value)
    return None


def _movement_bearing(track: VesselTrack, index: int) -> Optional[float]:
    """Compass bearing of the resampled track's motion around ``index``.

    A fallback for when AIS itself reported neither heading nor course -
    common on smaller or older-transponder vessels. Computed from a finite
    difference in an equal-area projection centred between the two points, so
    it is a real bearing rather than a naive slope in lon/lat.
    """
    positions = track.resampled
    if len(positions) < 2:
        return None
    i0, i1 = max(index - 1, 0), min(index + 1, len(positions) - 1)
    if i0 == i1:
        return None

    p0, p1 = positions.iloc[i0], positions.iloc[i1]
    midpoint = Point((p0["lon"] + p1["lon"]) / 2, (p0["lat"] + p1["lat"]) / 2)
    transformer, _ = _project(midpoint)
    x0, y0 = transformer.transform(p0["lon"], p0["lat"])
    x1, y1 = transformer.transform(p1["lon"], p1["lat"])
    if x0 == x1 and y0 == y1:
        return None
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0


def heading_at_cpa(track: VesselTrack, cpa_time: datetime) -> Optional[float]:
    """Vessel heading (compass degrees) at its closest point of approach.

    Prefers AIS's own reported heading, then course over ground, from the raw
    ping nearest ``cpa_time``; falls back to the interpolated track's own
    direction of travel when neither was reported.
    """
    nearest = track.points.loc[
        (track.points["timestamp"] - pd.Timestamp(cpa_time)).abs().idxmin()
    ]
    heading = _valid_heading(nearest["heading"])
    if heading is not None:
        return heading
    cog = _valid_heading(nearest["cog"])
    if cog is not None:
        return cog

    resampled = track.resampled
    if resampled.empty:
        return None
    nearest_index = int((resampled["timestamp"] - pd.Timestamp(cpa_time)).abs().idxmin())
    return _movement_bearing(track, nearest_index)


def filter_candidates(
    tracks: Dict[int, VesselTrack], spill: SpillLike, settings: Optional[Settings] = None,
) -> List[CandidateVessel]:
    """Rank ``tracks`` down to a candidate list for ``spill``.

    Drops a track whose CPA distance exceeds ``ais.search_radius_km`` (the
    same buffer :func:`src.ais.query.query_ais_for_spill` searched with) or
    that never moves faster than ``ais.min_moving_speed_knots``. Survivors are
    sorted by CPA distance, nearest first, capped at
    ``ais.max_candidate_vessels``.
    """
    settings = settings or get_settings()
    geometry, _ = spill_geometry_and_time(spill)
    # Built once and reused for every candidate: the spill geometry (and so
    # its equal-area projection) does not change across the loop.
    transformer, projected_geometry = _project(geometry)

    candidates: List[CandidateVessel] = []
    for mmsi, track in tracks.items():
        cpa_distance_km, cpa_time = _cpa_with_projection(track, transformer, projected_geometry)
        if cpa_distance_km > settings.ais.search_radius_km:
            continue
        if is_stationary(track, settings.ais.min_moving_speed_knots):
            continue
        candidates.append(
            CandidateVessel(
                mmsi=mmsi,
                cpa_distance_km=cpa_distance_km,
                cpa_time=cpa_time,
                track=track.points,
                vessel_name=track.vessel_name,
                vessel_type=track.vessel_type,
                heading_at_cpa=heading_at_cpa(track, cpa_time),
            )
        )

    candidates.sort(key=lambda c: c.cpa_distance_km)
    return candidates[: settings.ais.max_candidate_vessels]


__all__ = [
    "CandidateVessel",
    "HEADING_NOT_AVAILABLE",
    "closest_point_of_approach",
    "heading_at_cpa",
    "is_stationary",
    "filter_candidates",
]
