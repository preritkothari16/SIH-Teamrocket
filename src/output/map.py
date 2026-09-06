"""A basic Folium map from one scripts/run_pipeline.py combined output.

Draws each spill's polygon plus every candidate vessel's track (from the
"vessels" list Step 2.3/3.3 already produced), coloured by rank, with a
popup naming the vessel's score and explanation string.

Deliberately basic - this is the Phase 3 integration milestone, not the
dashboard. Styling, layering, and interactivity are Phase 5's job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import folium
from shapely.geometry import shape as shapely_shape

#: rank 0 (best-scoring candidate) first; cycles if a spill has more
#: candidates than colours - all valid folium marker-icon colours too, so
#: track and marker for the same vessel always match.
RANK_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
SPILL_COLOR = "black"


class MapBuildError(RuntimeError):
    """A combined pipeline result could not be turned into a map."""


def _rank_color(rank: int) -> str:
    return RANK_COLORS[rank % len(RANK_COLORS)]


def _centroid(geometry: Dict[str, Any]) -> List[float]:
    """[lat, lon] of a GeoJSON geometry's centroid - folium wants lat first."""
    point = shapely_shape(geometry).centroid
    return [point.y, point.x]


def _spill_popup(spill: Dict[str, Any], alert: Dict[str, Any]) -> str:
    properties = spill.get("properties") or {}
    spill_id = alert.get("spill_id") or properties.get("spill_id") or "unknown"
    area_km2 = properties.get("area_km2")
    area_text = f"{area_km2:.3f} km²" if area_km2 is not None else "unknown area"
    return (
        f"<b>{spill_id}</b><br>"
        f"status: {alert.get('status', 'unknown')}<br>"
        f"area: {area_text}<br>"
        f"candidates: {len(alert.get('rules') or [])} rule(s) evaluated"
    )


def _vessel_popup(vessel: Dict[str, Any], rank: int) -> str:
    name = vessel.get("vessel_name") or "unknown name"
    return (
        f"<b>#{rank + 1} - MMSI {vessel.get('mmsi', 'unknown')}</b> ({name})<br>"
        f"score: {vessel.get('score', 'n/a')}<br>"
        f"{vessel.get('explanation', 'no explanation recorded')}"
    )


def build_map(result: Dict[str, Any]) -> folium.Map:
    """A folium.Map for one run_pipeline.py combined result dict.

    Every spill's polygon is drawn regardless of alert status (rejected
    spills too - explainability includes showing what didn't alert and why,
    via the popup). Vessel tracks are drawn only for spills that have any
    (an empty "vessels" list, e.g. no --ais given, just draws the spill).
    """
    spills = result.get("spills") or []
    if not spills:
        raise MapBuildError("pipeline result has no spills to map")

    fmap = folium.Map(location=_centroid(spills[0]["spill"]["geometry"]), zoom_start=9)

    for entry in spills:
        spill = entry["spill"]
        alert = entry.get("alert") or {}
        vessels = entry.get("vessels") or []

        folium.GeoJson(
            spill["geometry"],
            style_function=lambda _: {"color": SPILL_COLOR, "weight": 2, "fillOpacity": 0.25},
            tooltip=_spill_popup(spill, alert),
        ).add_to(fmap)

        for rank, vessel in enumerate(vessels):
            track = vessel.get("track") or []
            if not track:
                continue
            color = _rank_color(rank)
            coordinates = [[point["lat"], point["lon"]] for point in track]

            folium.PolyLine(coordinates, color=color, weight=3, opacity=0.8).add_to(fmap)
            folium.Marker(
                coordinates[-1],
                popup=folium.Popup(_vessel_popup(vessel, rank), max_width=300),
                icon=folium.Icon(color=color),
            ).add_to(fmap)

    return fmap


def save_map(result: Dict[str, Any], output_path: Path) -> Path:
    """Build the map and save it as a standalone HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_map(result).save(str(output_path))
    return output_path


__all__ = ["MapBuildError", "build_map", "save_map"]
