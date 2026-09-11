"""Pydantic models matching the frontend contract exactly.

The frontend (frontend/src/types/schema.ts) was built against these shapes.
Do not rename or restructure any field — integration tests assert field-for-field
compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
class Point(BaseModel):
    lat: float
    lon: float


class BBox(BaseModel):
    minLon: float
    minLat: float
    maxLon: float
    maxLat: float


class PolygonGeometry(BaseModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: List[List[List[float]]]


class LineStringGeometry(BaseModel):
    type: Literal["LineString"] = "LineString"
    coordinates: List[List[float]]


# --------------------------------------------------------------------------- #
# SpillObject — the primary detection result
# --------------------------------------------------------------------------- #
class SpillObject(BaseModel):
    scene_id: str
    acquisition_timestamp: str
    confidence: float
    area_km2: float
    centroid: Point
    bbox: BBox
    polygon: PolygonGeometry
    major_axis_bearing: float
    elongation: float


# --------------------------------------------------------------------------- #
# Alert
# --------------------------------------------------------------------------- #
class Alert(BaseModel):
    spill_id: str
    status: Literal["new", "update", "possible", "none"] = "none"
    rules_fired: List[str] = Field(default_factory=list)
    first_seen: str
    last_updated: str


# --------------------------------------------------------------------------- #
# Vessel (AIS attribution)
# --------------------------------------------------------------------------- #
class Vessel(BaseModel):
    mmsi: str
    name: str
    vessel_type: str
    score: float
    explanation: str
    cpa_distance_km: float
    cpa_time: str
    track: LineStringGeometry


# --------------------------------------------------------------------------- #
# Drift forecast / hindcast
# --------------------------------------------------------------------------- #
class ForecastEntry(BaseModel):
    hours: int = Field(..., description="Forecast horizon in hours: 6, 12, 24, or 48")
    time: str
    polygon: PolygonGeometry


class HindcastEntry(BaseModel):
    time: str
    polygon: PolygonGeometry


class DriftForecast(BaseModel):
    forecast: List[ForecastEntry] = Field(default_factory=list)


class DriftHindcast(BaseModel):
    hindcast: List[HindcastEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Drift (combined forecast + hindcast)
# --------------------------------------------------------------------------- #
class Drift(BaseModel):
    forecast: List[ForecastEntry] = Field(default_factory=list)
    hindcast: List[HindcastEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Provenance (Step 8.2)
# --------------------------------------------------------------------------- #
class Provenance(BaseModel):
    sar_source: Optional[str] = None
    sar_scene_id: Optional[str] = None
    ais_source_label: Optional[str] = None
    wind_source: Optional[str] = None
    current_source: Optional[str] = None


# --------------------------------------------------------------------------- #
# PipelineRun — the top-level contract
# --------------------------------------------------------------------------- #
class PipelineRun(BaseModel):
    spill: SpillObject
    alert: Alert
    vessels: List[Vessel] = Field(default_factory=list)
    drift: Drift = Field(default_factory=Drift)
    provenance: Optional[Provenance] = None


# --------------------------------------------------------------------------- #
# Run summary (for listing)
# --------------------------------------------------------------------------- #
class RunSummary(BaseModel):
    scene_id: str
    acquisition_timestamp: Optional[str] = None
    area_km2: float
    confidence: float
    alert_status: str = "none"


# --------------------------------------------------------------------------- #
# Trigger request
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    scene_path: Optional[str] = None
    scene_id: Optional[str] = None
    stub_model: bool = True


# --------------------------------------------------------------------------- #
# Attribution Q&A (Step 8.1)
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    cited_vessels: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Demo region presets (Step 8.4)
# --------------------------------------------------------------------------- #
class Region(BaseModel):
    id: str
    label: str
    bbox: List[float]
    scene_id: Optional[str] = None
