"""Shared ingestion types.

Everything downstream of ingestion consumes :class:`Scene` objects and nothing
else. Whether a scene came off the Copernicus catalogue over the network or off
a local disk is an ingestion-layer detail, so both
:mod:`src.ingestion.catalogue` and :mod:`src.ingestion.local_source` return the
same model and satisfy the same :class:`SceneSource` protocol.

Footprints are always shapely geometries in EPSG:4326 (lon/lat), and all
timestamps are timezone-aware UTC datetimes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

WGS84 = "EPSG:4326"

# S1A_IW_GRDH_1SDV_20230514T003332_20230514T003357_048456_05D45D_1A2B
SENTINEL1_NAME_RE = re.compile(
    r"""^
    (?P<platform>S1[ABCD])_
    (?P<mode>[A-Z0-9]{2})_
    (?P<product>GRD[HMF]?|SLC|OCN)[A-Z_]*_
    (?P<class_pol>[0-9][SA][A-Z]{2})_
    (?P<start>\d{8}T\d{6})_
    (?P<stop>\d{8}T\d{6})_
    (?P<absolute_orbit>\d{6})_
    (?P<mission_data_take>[0-9A-F]{6})_
    (?P<product_id>[0-9A-F]{4})
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Polarisation letter pairs used in the Sentinel-1 product name (e.g. "SDV").
_POLARISATION_MAP = {
    "SH": ["HH"],
    "SV": ["VV"],
    "DH": ["HH", "HV"],
    "DV": ["VV", "VH"],
}


class SceneSourceKind(str, Enum):
    """Where a scene came from. Downstream code should not branch on this."""

    CDSE = "cdse"
    LOCAL = "local"


class OrbitInfo(BaseModel):
    """Orbit geometry of an acquisition."""

    direction: Optional[str] = None  # ASCENDING / DESCENDING
    relative_orbit: Optional[int] = None
    absolute_orbit: Optional[int] = None

    @field_validator("direction")
    @classmethod
    def _normalise_direction(cls, v: Optional[str]) -> Optional[str]:
        return v.upper() if v else v


class Scene(BaseModel):
    """A single SAR acquisition, however it was obtained.

    This is the ingestion layer's only output type and the contract every
    downstream stage codes against.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene_id: str
    source: SceneSourceKind
    acquisition_time: datetime
    footprint: BaseGeometry

    acquisition_end: Optional[datetime] = None
    path: Optional[Path] = None
    download_url: Optional[str] = None

    platform: Optional[str] = None
    product_type: Optional[str] = None
    sensor_mode: Optional[str] = None
    polarisations: List[str] = Field(default_factory=list)
    orbit: OrbitInfo = Field(default_factory=OrbitInfo)

    size_bytes: Optional[int] = None
    footprint_crs: str = WGS84
    extra: Dict[str, Any] = Field(default_factory=dict)

    # -- validation ------------------------------------------------------- #
    @field_validator("acquisition_time", "acquisition_end")
    @classmethod
    def _as_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Naive datetimes are assumed UTC; aware ones are converted to it."""
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    @field_validator("footprint", mode="before")
    @classmethod
    def _coerce_footprint(cls, v: Any) -> BaseGeometry:
        """Accept a shapely geometry, a GeoJSON mapping, or a GeoJSON string."""
        if isinstance(v, BaseGeometry):
            geom = v
        elif isinstance(v, dict):
            geom = shape(v.get("geometry", v))
        elif isinstance(v, str):
            geom = shape(json.loads(v))
        else:
            raise TypeError(f"cannot interpret {type(v).__name__} as a footprint")
        if not isinstance(geom, (Polygon, MultiPolygon)):
            raise TypeError("footprint must be a Polygon or MultiPolygon")
        if geom.is_empty:
            raise ValueError("footprint must not be empty")
        return geom

    @field_validator("polarisations")
    @classmethod
    def _upper_polarisations(cls, v: List[str]) -> List[str]:
        return [p.upper() for p in v]

    @field_serializer("footprint")
    def _serialize_footprint(self, geom: BaseGeometry) -> Dict[str, Any]:
        return mapping(geom)

    # -- convenience ------------------------------------------------------ #
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Footprint bounds as (min_lon, min_lat, max_lon, max_lat)."""
        return tuple(self.footprint.bounds)  # type: ignore[return-value]

    @property
    def is_local(self) -> bool:
        """True when the pixels are already on disk."""
        return self.path is not None and Path(self.path).exists()

    def intersects(self, aoi: BaseGeometry) -> bool:
        return self.footprint.intersects(aoi)

    def to_geojson_feature(self) -> Dict[str, Any]:
        """GeoJSON Feature for quick inspection on a map."""
        payload = self.model_dump(mode="json")
        geometry = payload.pop("footprint")
        return {"type": "Feature", "geometry": geometry, "properties": payload}


class SceneSource(Protocol):
    """Interface both the catalogue client and the local reader implement."""

    def search(
        self,
        aoi: BaseGeometry,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
    ) -> List[Scene]:
        """Scenes intersecting ``aoi`` acquired within [start, end]."""
        ...

    def fetch(self, scene: Scene) -> Path:
        """Ensure the scene's pixels are on disk and return the path."""
        ...


# --------------------------------------------------------------------------- #
# helpers shared by both sources
# --------------------------------------------------------------------------- #
def load_aoi(source: Any) -> BaseGeometry:
    """Build an AOI geometry from a geojson path, mapping, or shapely geometry.

    Accepts a FeatureCollection, a Feature, or a bare geometry. Multiple
    features are dissolved into their union.
    """
    if isinstance(source, BaseGeometry):
        return source

    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif isinstance(source, dict):
        data = source
    else:
        raise TypeError(f"cannot interpret {type(source).__name__} as an AOI")

    kind = data.get("type")
    if kind == "FeatureCollection":
        features = data.get("features") or []
        if not features:
            raise ValueError("AOI FeatureCollection has no features")
        geoms = [shape(f["geometry"]) for f in features]
        merged = geoms[0]
        for geom in geoms[1:]:
            merged = merged.union(geom)
        return merged
    if kind == "Feature":
        return shape(data["geometry"])
    return shape(data)


def as_utc(value: datetime) -> datetime:
    """Timezone-aware UTC version of ``value`` (naive input is assumed UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_sentinel1_name(name: str) -> Dict[str, Any]:
    """Pull acquisition metadata out of a Sentinel-1 product name.

    Returns an empty dict when ``name`` is not a Sentinel-1 product name, so
    callers can fall back to other metadata sources.
    """
    match = SENTINEL1_NAME_RE.match(Path(name).stem)
    if match is None:
        return {}

    groups = match.groupdict()
    fmt = "%Y%m%dT%H%M%S"
    # class_pol is e.g. "1SDV": class, polarisation-count letter, channel pair.
    class_pol = groups["class_pol"].upper()[2:]
    return {
        "platform": groups["platform"].upper(),
        "sensor_mode": groups["mode"].upper(),
        "product_type": groups["product"].upper(),
        "polarisations": _POLARISATION_MAP.get(class_pol, []),
        "acquisition_time": datetime.strptime(groups["start"], fmt).replace(
            tzinfo=timezone.utc
        ),
        "acquisition_end": datetime.strptime(groups["stop"], fmt).replace(
            tzinfo=timezone.utc
        ),
        "absolute_orbit": int(groups["absolute_orbit"]),
        "mission_data_take": groups["mission_data_take"].upper(),
    }


def filter_scenes(
    scenes: Iterable[Scene],
    aoi: Optional[BaseGeometry] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Scene]:
    """Apply the common AOI / time-window / count filters, newest first."""
    selected: List[Scene] = []
    for scene in scenes:
        if aoi is not None and not scene.footprint.intersects(aoi):
            continue
        if start is not None and scene.acquisition_time < as_utc(start):
            continue
        if end is not None and scene.acquisition_time > as_utc(end):
            continue
        selected.append(scene)

    selected.sort(key=lambda s: s.acquisition_time, reverse=True)
    if limit is not None:
        selected = selected[:limit]
    return selected


def scenes_to_geojson(scenes: Sequence[Scene]) -> Dict[str, Any]:
    """FeatureCollection of scene footprints, handy for debugging on a map."""
    return {
        "type": "FeatureCollection",
        "features": [scene.to_geojson_feature() for scene in scenes],
    }


__all__ = [
    "WGS84",
    "Scene",
    "SceneSource",
    "SceneSourceKind",
    "OrbitInfo",
    "load_aoi",
    "as_utc",
    "parse_sentinel1_name",
    "filter_scenes",
    "scenes_to_geojson",
]
