"""A Folium map from one scripts/run_pipeline.py combined output.

Draws each spill's polygon, its forward drift-forecast polygons (from the
"drift_forecast" list Step 4.5 added), an optional hindcast corridor (Step
5.4, computed on demand - it is not persisted in the combined JSON, see
:mod:`src.output.dashboard`), and every candidate vessel's track (from the
"vessels" list Step 2.3/3.3), coloured by rank, with a popup naming the
vessel's score and explanation string.

Each of those four is its own :class:`folium.FeatureGroup` under a
:class:`folium.LayerControl`, so a viewer can toggle any one off - this is
what :mod:`src.output.dashboard` (Step 5.4) means by its "layer toggle";
nothing dashboard-specific lives here; both it and :mod:`src.output.report`
just embed this map as-is.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import folium
from shapely.geometry import shape as shapely_shape

if TYPE_CHECKING:
    from src.drift.hindcast import OriginSnapshot

#: rank 0 (best-scoring candidate) first; cycles if a spill has more
#: candidates than colours - all valid folium marker-icon colours too, so
#: track and marker for the same vessel always match.
RANK_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]
SPILL_COLOR = "black"
#: dashed + a colour no vessel track/spill outline uses, so a forecast
#: polygon reads as neither a detection nor a track at a glance.
FORECAST_COLOR = "gray"
FORECAST_DASH_ARRAY = "6, 6"
#: distinct again from both the forecast (forward) and everything else - a
#: hindcast polygon must not be mistaken for a forecast one at a glance.
HINDCAST_COLOR = "purple"
HINDCAST_DASH_ARRAY = "2, 6"


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


def _forecast_popup(forecast: Dict[str, Any]) -> str:
    return f"+{forecast['hours_elapsed']:.0f}h forecast<br>{forecast.get('time', 'unknown time')}"


def _hindcast_popup(snapshot: "OriginSnapshot") -> str:
    return f"-{snapshot.hours_before_acquisition:.0f}h hindcast<br>{snapshot.time.isoformat()}"


def _vessel_popup(vessel: Dict[str, Any], rank: int) -> str:
    name = vessel.get("vessel_name") or "unknown name"
    return (
        f"<b>#{rank + 1} - MMSI {vessel.get('mmsi', 'unknown')}</b> ({name})<br>"
        f"score: {vessel.get('score', 'n/a')}<br>"
        f"{vessel.get('explanation', 'no explanation recorded')}"
    )


def build_map(
    result: Dict[str, Any],
    hindcast_corridors: Optional[Dict[str, Sequence["OriginSnapshot"]]] = None,
) -> folium.Map:
    """A folium.Map for one run_pipeline.py combined result dict.

    Every spill's polygon is drawn regardless of alert status (rejected
    spills too - explainability includes showing what didn't alert and why,
    via the popup), and so is its drift forecast (an empty/missing
    "drift_forecast" list just skips that spill's overlay - older combined
    results without step 4.5 still map fine). Vessel tracks are drawn only
    for spills that have any (an empty "vessels" list, e.g. no --ais given,
    just draws the spill).

    ``hindcast_corridors`` is optional and keyed by spill_id - the combined
    JSON does not carry it (see :mod:`src.drift.hindcast`'s own docstring for
    why it's compute-on-demand, not persisted); a caller that has one (e.g.
    :mod:`src.output.dashboard`) passes it in, everyone else just gets an
    empty (but still present, still toggleable) "Hindcast corridor" layer.

    All four kinds of overlay are their own :class:`folium.FeatureGroup`
    under one :class:`folium.LayerControl`, so any of them can be toggled
    off independently in the rendered map.
    """
    spills = result.get("spills") or []
    if not spills:
        raise MapBuildError("pipeline result has no spills to map")
    hindcast_corridors = hindcast_corridors or {}

    fmap = folium.Map(location=_centroid(spills[0]["spill"]["geometry"]), zoom_start=9)
    slick_layer = folium.FeatureGroup(name="Detected slicks", show=True)
    forecast_layer = folium.FeatureGroup(name="Drift forecast", show=True)
    hindcast_layer = folium.FeatureGroup(name="Hindcast corridor", show=False)
    vessel_layer = folium.FeatureGroup(name="Vessel tracks", show=True)

    for entry in spills:
        spill = entry["spill"]
        alert = entry.get("alert") or {}
        vessels = entry.get("vessels") or []
        spill_id = alert.get("spill_id") or (spill.get("properties") or {}).get("spill_id")

        folium.GeoJson(
            spill["geometry"],
            style_function=lambda _: {"color": SPILL_COLOR, "weight": 2, "fillOpacity": 0.25},
            tooltip=_spill_popup(spill, alert),
        ).add_to(slick_layer)

        for forecast in entry.get("drift_forecast") or []:
            folium.GeoJson(
                forecast["polygon"],
                style_function=lambda _: {
                    "color": FORECAST_COLOR, "weight": 2, "fillOpacity": 0.05,
                    "dashArray": FORECAST_DASH_ARRAY,
                },
                tooltip=_forecast_popup(forecast),
            ).add_to(forecast_layer)

        for snapshot in hindcast_corridors.get(spill_id) or []:
            folium.GeoJson(
                snapshot.polygon,  # shapely geometries implement __geo_interface__
                style_function=lambda _: {
                    "color": HINDCAST_COLOR, "weight": 2, "fillOpacity": 0.03,
                    "dashArray": HINDCAST_DASH_ARRAY,
                },
                tooltip=_hindcast_popup(snapshot),
            ).add_to(hindcast_layer)

        for rank, vessel in enumerate(vessels):
            track = vessel.get("track") or []
            if not track:
                continue
            color = _rank_color(rank)
            coordinates = [[point["lat"], point["lon"]] for point in track]

            folium.PolyLine(coordinates, color=color, weight=3, opacity=0.8).add_to(vessel_layer)
            folium.Marker(
                coordinates[-1],
                popup=folium.Popup(_vessel_popup(vessel, rank), max_width=300),
                icon=folium.Icon(color=color),
            ).add_to(vessel_layer)

    for layer in (slick_layer, forecast_layer, hindcast_layer, vessel_layer):
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap


def save_map(
    result: Dict[str, Any],
    output_path: Path,
    hindcast_corridors: Optional[Dict[str, Sequence["OriginSnapshot"]]] = None,
) -> Path:
    """Build the map and save it as a standalone HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_map(result, hindcast_corridors=hindcast_corridors).save(str(output_path))
    return output_path


__all__ = ["MapBuildError", "build_map", "save_map"]
