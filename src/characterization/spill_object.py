"""Turn a filtered oil mask into spill objects.

The end of Phase 1: pixels become polygons with real-world measurements, and
each one leaves as a GeoJSON Feature carrying the scene id and acquisition time
from ingestion - which is what lets AIS attribution and drift prediction pick
the spill up later without going back to the raster.

Area
----
**Areas are never computed in lon/lat degrees.** A square degree is about
12,300 km² at the equator and shrinks towards the poles, so a degree-space area
is wrong by a latitude-dependent factor - roughly 8% at 20°N and worse further
north. Every polygon is reprojected to a Lambert azimuthal equal-area
projection centred on its own centroid before its area is taken. That is exact
for equal-area purposes at the scale of a single slick and needs no choice of
UTM zone, so it works for a scene anywhere in the world.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from affine import Affine
from pyproj import CRS as ProjCRS
from pyproj import Transformer
from rasterio.features import shapes as raster_shapes
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
SQ_M_PER_SQ_KM = 1_000_000.0


def equal_area_crs_for(geometry: BaseGeometry) -> ProjCRS:
    """A Lambert azimuthal equal-area CRS centred on this geometry.

    Centring on the feature itself keeps distortion negligible over a slick and
    sidesteps picking a UTM zone, which breaks down for anything spanning a zone
    boundary or sitting at high latitude.
    """
    centroid = geometry.centroid
    return ProjCRS.from_proj4(
        f"+proj=laea +lat_0={centroid.y} +lon_0={centroid.x} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def to_equal_area(
    geometry: BaseGeometry, source_crs: str = WGS84
) -> Tuple[BaseGeometry, ProjCRS]:
    """Reproject a geometry into its own equal-area CRS."""
    target = equal_area_crs_for(
        geometry if str(source_crs).upper().endswith("4326") else geometry
    )
    transformer = Transformer.from_crs(source_crs, target, always_xy=True)
    return shapely_transform(transformer.transform, geometry), target


def area_km2(geometry: BaseGeometry, source_crs: str = WGS84) -> float:
    """Area of a geometry in km², computed in an equal-area projection."""
    if geometry.is_empty:
        return 0.0
    if not ProjCRS.from_user_input(source_crs).is_geographic:
        return float(geometry.area) / SQ_M_PER_SQ_KM
    projected, _ = to_equal_area(geometry, source_crs)
    return float(projected.area) / SQ_M_PER_SQ_KM


def perimeter_km(geometry: BaseGeometry, source_crs: str = WGS84) -> float:
    """Boundary length in km, measured in the same equal-area projection."""
    if geometry.is_empty:
        return 0.0
    if not ProjCRS.from_user_input(source_crs).is_geographic:
        return float(geometry.length) / 1000.0
    projected, _ = to_equal_area(geometry, source_crs)
    return float(projected.length) / 1000.0


# --------------------------------------------------------------------------- #
# mask -> polygon
# --------------------------------------------------------------------------- #
def mask_to_polygon(
    mask: np.ndarray,
    transform: Affine,
    simplify_tolerance_m: Optional[float] = None,
    min_area_km2: Optional[float] = None,
    crs: str = WGS84,
    settings: Optional[Settings] = None,
) -> List[Polygon]:
    """Vectorise a boolean mask into simplified polygons in map coordinates.

    Simplification is specified in metres and converted to the raster's own
    units, so the same tolerance behaves identically for a projected scene and
    a lon/lat one. Holes are preserved - a slick with a clear patch inside it is
    genuinely smaller than its outline.
    """
    settings = settings or get_settings()
    cfg = settings.characterization
    tolerance_m = (
        cfg.simplify_tolerance_m if simplify_tolerance_m is None else simplify_tolerance_m
    )
    min_area = cfg.min_polygon_area_km2 if min_area_km2 is None else min_area_km2

    binary = np.ascontiguousarray(mask.astype(np.uint8))
    if not binary.any():
        return []

    geometries: List[Polygon] = []
    for geometry, value in raster_shapes(binary, mask=binary.astype(bool),
                                         transform=transform):
        if not value:
            continue
        polygon = shape(geometry)
        if polygon.is_empty:
            continue
        geometries.append(polygon)

    tolerance = _tolerance_in_crs_units(tolerance_m, crs)
    cleaned: List[Polygon] = []
    for polygon in geometries:
        simplified = polygon.simplify(tolerance, preserve_topology=True)
        if simplified.is_empty:
            continue
        if not simplified.is_valid:
            simplified = simplified.buffer(0)
        for part in _iter_polygons(simplified):
            if area_km2(part, crs) >= min_area:
                cleaned.append(part)

    cleaned.sort(key=lambda g: area_km2(g, crs), reverse=True)
    logger.debug("vectorised %d polygon(s) from the mask", len(cleaned))
    return cleaned


def _iter_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms


def _tolerance_in_crs_units(tolerance_m: float, crs: str) -> float:
    """Metres to the raster CRS's units (degrees for a geographic CRS)."""
    if tolerance_m <= 0:
        return 0.0
    return (
        tolerance_m / 111_320.0
        if ProjCRS.from_user_input(crs).is_geographic
        else tolerance_m
    )


# --------------------------------------------------------------------------- #
# shape descriptors
# --------------------------------------------------------------------------- #
def orientation_and_elongation(
    geometry: BaseGeometry, source_crs: str = WGS84
) -> Tuple[float, float, float, float]:
    """(orientation°, elongation, major axis km, minor axis km).

    Taken from the minimum rotated rectangle in an equal-area projection, so the
    axes are real distances. Orientation is degrees clockwise from east, in
    [0, 180) - a slick and the same slick flipped point the same way.
    """
    if geometry.is_empty:
        return 0.0, 1.0, 0.0, 0.0

    projected = (
        to_equal_area(geometry, source_crs)[0]
        if ProjCRS.from_user_input(source_crs).is_geographic
        else geometry
    )
    rectangle = projected.minimum_rotated_rectangle
    corners = list(rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return 0.0, 1.0, 0.0, 0.0

    edge_a = np.subtract(corners[1], corners[0])
    edge_b = np.subtract(corners[2], corners[1])
    length_a = float(np.hypot(*edge_a))
    length_b = float(np.hypot(*edge_b))

    if length_a >= length_b:
        major, minor, axis = length_a, length_b, edge_a
    else:
        major, minor, axis = length_b, length_a, edge_b

    elongation = major / minor if minor > 1e-9 else float("inf")
    orientation = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return orientation, elongation, major / 1000.0, minor / 1000.0


# --------------------------------------------------------------------------- #
# spill objects
# --------------------------------------------------------------------------- #
def build_spill_object(
    geometry: BaseGeometry,
    scene_id: str,
    acquisition_timestamp: Optional[datetime] = None,
    spill_id: Optional[str] = None,
    confidence: Optional[float] = None,
    crs: str = WGS84,
    extra: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """One spill as a GeoJSON Feature with its measurements as properties.

    ``scene_id`` and ``acquisition_timestamp`` come straight from the ingestion
    record and are carried through untouched: AIS attribution matches vessels
    against that exact time, and drift prediction integrates forward from it.
    """
    settings = settings or get_settings()

    centroid = geometry.centroid
    west, south, east, north = geometry.bounds
    orientation, elongation, major_km, minor_km = orientation_and_elongation(geometry, crs)
    spill_area = area_km2(geometry, crs)

    timestamp = acquisition_timestamp
    if isinstance(timestamp, datetime):
        timestamp = (
            timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        ).astimezone(timezone.utc)

    properties: Dict[str, Any] = {
        "spill_id": spill_id or f"{scene_id}_spill",
        "scene_id": scene_id,
        "acquisition_timestamp": timestamp.isoformat() if timestamp else None,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "area_km2": round(spill_area, 6),
        "perimeter_km": round(perimeter_km(geometry, crs), 6),
        "centroid_lon": round(float(centroid.x), 6),
        "centroid_lat": round(float(centroid.y), 6),
        "bbox": [round(float(v), 6) for v in (west, south, east, north)],
        "orientation_deg": round(orientation, 2),
        "elongation": round(elongation, 3) if np.isfinite(elongation) else None,
        "major_axis_km": round(major_km, 4),
        "minor_axis_km": round(minor_km, 4),
        "mean_confidence": round(float(confidence), 4) if confidence is not None else None,
        "estimated_volume_m3": round(
            spill_area
            * SQ_M_PER_SQ_KM
            * settings.characterization.volume_estimate_thickness_um
            * 1e-6,
            3,
        ),
        "crs": crs,
    }
    if extra:
        properties.update(extra)

    return {"type": "Feature", "geometry": mapping(geometry), "properties": properties}


def spills_from_mask(
    mask: np.ndarray,
    transform: Affine,
    scene_id: str,
    acquisition_timestamp: Optional[datetime] = None,
    confidence: Optional[np.ndarray] = None,
    crs: str = WGS84,
    settings: Optional[Settings] = None,
) -> List[Dict[str, Any]]:
    """Vectorise a filtered oil mask and describe every spill in it."""
    settings = settings or get_settings()
    polygons = mask_to_polygon(mask, transform, crs=crs, settings=settings)

    features: List[Dict[str, Any]] = []
    for number, polygon in enumerate(polygons, start=1):
        features.append(
            build_spill_object(
                polygon,
                scene_id=scene_id,
                acquisition_timestamp=acquisition_timestamp,
                spill_id=f"{scene_id}_spill_{number:03d}",
                confidence=_mean_confidence(confidence, mask),
                crs=crs,
                settings=settings,
            )
        )
    return features


def _mean_confidence(
    confidence: Optional[np.ndarray], mask: np.ndarray
) -> Optional[float]:
    if confidence is None or not mask.any():
        return None
    values = confidence[mask.astype(bool)]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def feature_collection(
    features: Sequence[Dict[str, Any]],
    scene_id: Optional[str] = None,
    acquisition_timestamp: Optional[datetime] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrap features for writing out.

    A scene with no spills still produces a FeatureCollection with an empty
    feature list - "we looked and found nothing" is a real result, and a missing
    file would be indistinguishable from a scene that was never processed.
    """
    properties: Dict[str, Any] = {
        "scene_id": scene_id,
        "acquisition_timestamp": (
            acquisition_timestamp.isoformat()
            if isinstance(acquisition_timestamp, datetime)
            else acquisition_timestamp
        ),
        "spill_count": len(features),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        properties.update(extra)
    return {
        "type": "FeatureCollection",
        "properties": properties,
        "features": list(features),
    }


__all__ = [
    "WGS84",
    "mask_to_polygon",
    "build_spill_object",
    "spills_from_mask",
    "feature_collection",
    "area_km2",
    "perimeter_km",
    "orientation_and_elongation",
    "equal_area_crs_for",
    "to_equal_area",
]
