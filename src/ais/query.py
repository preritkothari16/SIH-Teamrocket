"""Given a spill, find the AIS traffic that could plausibly explain it.

A spill (as :func:`src.characterization.spill_object.build_spill_object`
produces it) carries a polygon and an acquisition timestamp. This module
turns those two things into a space-time search: a time window looking back
from the acquisition time, and a buffered search area around the spill's
footprint, both configurable via ``ais.search_window_hours`` and
``ais.search_radius_km``.

The window is deliberately one-sided - ``[acquisition_time -
search_window_hours, acquisition_time]`` - a vessel has to have been near the
slick at or before it was seen, not after.

The search area is a **buffered bounding box computed in an equal-area
projection**, not a buffer in degrees: a degree of longitude is not a fixed
distance, so buffering in lon/lat would make the search radius depend on
latitude. This reuses the same equal-area machinery
:mod:`src.characterization.spill_object` already uses for area, centred on
the spill itself.

A vessel qualifies if **any one** of its positions falls inside the
space-time box; once it qualifies, every one of its positions within the time
window is returned (not just the ones inside the search area), so track
reconstruction downstream sees the vessel's approach and departure, not just
the moment it happened to be close.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.characterization.spill_object import WGS84, equal_area_crs_for
from src.config import Settings, get_settings

SpillLike = Union[Dict[str, Any], Tuple[BaseGeometry, Any]]


class AISQueryError(RuntimeError):
    """A spill object could not be interpreted as a space-time search."""


def _as_utc(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def spill_geometry_and_time(spill: SpillLike) -> Tuple[BaseGeometry, datetime]:
    """Pull ``(geometry, acquisition_time)`` out of a spill GeoJSON Feature.

    Also accepts a bare ``(geometry, timestamp)`` pair, which is convenient in
    tests that do not want to build a full Feature dict.
    """
    if isinstance(spill, dict):
        if "geometry" not in spill:
            raise AISQueryError("spill has no 'geometry'")
        geometry = shape(spill["geometry"])
        timestamp = (spill.get("properties") or {}).get("acquisition_timestamp")
        if timestamp is None:
            raise AISQueryError("spill has no properties.acquisition_timestamp")
        return geometry, _as_utc(timestamp)

    geometry, timestamp = spill
    return geometry, _as_utc(timestamp)


def search_window(
    acquisition_time: datetime, window_hours: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> Tuple[datetime, datetime]:
    """One-sided ``[acquisition_time - window_hours, acquisition_time]``."""
    settings = settings or get_settings()
    window_hours = window_hours if window_hours is not None else settings.ais.search_window_hours
    end = _as_utc(acquisition_time)
    start = end - timedelta(hours=window_hours)
    return start, end


def search_area(
    geometry: BaseGeometry, buffer_km: Optional[float] = None,
    source_crs: str = WGS84, settings: Optional[Settings] = None,
) -> BaseGeometry:
    """The spill's bounding box, buffered by ``buffer_km`` in an equal-area CRS.

    Returned in ``source_crs`` (WGS84 by default), ready to test AIS lon/lat
    points against directly.
    """
    settings = settings or get_settings()
    buffer_km = buffer_km if buffer_km is not None else settings.ais.search_radius_km

    bbox = box(*geometry.bounds)
    equal_area = equal_area_crs_for(bbox)
    to_equal_area = Transformer.from_crs(source_crs, equal_area, always_xy=True)
    to_source = Transformer.from_crs(equal_area, source_crs, always_xy=True)

    projected_bbox = shapely_transform(to_equal_area.transform, bbox)
    buffered = projected_bbox.buffer(buffer_km * 1000.0)
    return shapely_transform(to_source.transform, buffered)


def query_ais_for_spill(
    spill: SpillLike, ais: pd.DataFrame, settings: Optional[Settings] = None,
) -> pd.DataFrame:
    """AIS points for every vessel with at least one ping in the spill's
    space-time box.

    ``ais`` must already be in the normalized schema
    :func:`src.ais.loader.normalize_ais_frame` produces. Returns a frame with
    the same columns, sorted by ``mmsi``, ``timestamp`` - empty (not None)
    when nothing qualifies, since "no candidate vessels" is a real, reportable
    result.
    """
    settings = settings or get_settings()
    geometry, acquisition_time = spill_geometry_and_time(spill)

    start, end = search_window(acquisition_time, settings=settings)
    area = search_area(geometry, settings=settings)

    in_window = ais[(ais["timestamp"] >= start) & (ais["timestamp"] <= end)]
    if in_window.empty:
        return in_window.copy()

    inside_area = in_window[
        in_window["lon"].between(*area.bounds[0::2])
        & in_window["lat"].between(*area.bounds[1::2])
    ]
    # the bbox pre-filter above is cheap and only over-selects; the exact
    # buffered (generally non-rectangular after reprojection) area decides.
    hits = inside_area[
        inside_area.apply(lambda r: area.contains(Point(r["lon"], r["lat"])), axis=1)
    ]

    qualifying_mmsi = hits["mmsi"].unique()
    result = in_window[in_window["mmsi"].isin(qualifying_mmsi)]
    return result.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)


__all__ = [
    "AISQueryError",
    "SpillLike",
    "spill_geometry_and_time",
    "search_window",
    "search_area",
    "query_ais_for_spill",
]
