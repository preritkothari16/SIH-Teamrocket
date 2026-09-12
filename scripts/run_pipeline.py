"""Phase 3-4 end to end: a raw scene path in, alerts + ranked vessels + drift
forecast out.

The first true end-to-end script: raw scene -> ingestion -> preprocessing ->
detection -> look-alike filter -> characterization -> alert manager ->
forward drift forecast -> (if alerted) AIS query -> tracks -> filter ->
(if ``attribution.use_hindcasting``) hindcast corridor -> scoring.

Reuses each earlier phase's own script/module as a **library** -
:func:`scripts.run_detection.run` for steps 1.1-1.6,
:func:`scripts.run_attribution.attribute_spill` for steps 2.1-2.3, and
:func:`src.drift.forward.forecast_drift` / :func:`src.drift.hindcast.hindcast_origin`
for step 4.3/4.4 - rather than reimplementing chain assembly that already
exists and is already tested. Nothing here is shelled out to; every step is a
plain Python import and function call.

Usage::

    python scripts/run_pipeline.py --scene <tif> --stub-model
    python scripts/run_pipeline.py --scene <tif> --stub-model --ais data/ais/export.csv
    python scripts/run_pipeline.py --scene <tif> --stub-model --map
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shapely.geometry import mapping, shape

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_pipeline.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.ais.loader import load_ais  # noqa: E402
from src.alerts.manager import process_spill  # noqa: E402
from src.alerts.registry import SpillRegistry  # noqa: E402
from src.api.registry import forecast_to_contract, vessels_to_contract  # noqa: E402
from src.characterization.spill_object import geographic_centroid  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402
from src.db import database_url  # noqa: E402
from src.drift.forward import forecast_drift  # noqa: E402
from src.drift.hindcast import hindcast_origin  # noqa: E402
from src.env_data.service import get_environment  # noqa: E402
from src.output.map import MapBuildError, save_map  # noqa: E402
from src.output.report import ReportBuildError, save_report  # noqa: E402

from scripts import run_attribution  # noqa: E402
from scripts import run_detection  # noqa: E402

logger = logging.getLogger(__name__)


def _drift_forecast_payload(spill: Dict[str, Any], settings: Settings) -> List[Dict[str, Any]]:
    """Forward drift forecast for one spill (see :mod:`src.drift.forward`),
    JSON-ready: each horizon's swarm as a GeoJSON polygon (its convex hull)."""
    forecasts = forecast_drift(spill, settings=settings)
    return [
        {
            "hours_elapsed": f.hours_elapsed,
            "time": f.time.isoformat(),
            "polygon": mapping(f.polygon),
        }
        for f in forecasts
    ]


def _provenance_payload(
    detection: Dict[str, Any],
    scene_id_resolved: Optional[str],
    spills: List[Dict[str, Any]],
    ais_path: Optional[Path],
    ais_source_label: str,
    settings: Settings,
) -> Dict[str, Any]:
    """What actually produced this run - see the Step 8.2 brief for what was
    checked before writing this.

    ``sar_source`` ("cdse"|"local") is already sitting in
    ``detection["properties"]["source"]`` -
    :func:`scripts.run_detection.run` puts ``scene.source.value`` there via
    :func:`src.characterization.spill_object.feature_collection`'s own
    ``extra`` - no new plumbing needed. ``wind_source``/``current_source``
    are sampled once, at the first spill's centroid/time, purely to label
    this run - the per-spill env lookups inside :mod:`src.alerts.manager`
    and :mod:`src.drift.forward` are unaffected and unchanged.
    """
    properties = detection.get("properties") or {}
    provenance: Dict[str, Any] = {
        "sar_source": properties.get("source"),
        "sar_scene_id": scene_id_resolved,
        "ais_source_label": ais_source_label if ais_path else None,
        "wind_source": None,
        "current_source": None,
    }

    if spills:
        first_props = spills[0].get("properties", {}) or {}
        lon, lat = first_props.get("centroid_lon"), first_props.get("centroid_lat")
        timestamp = first_props.get("acquisition_timestamp")
        if lon is not None and lat is not None and timestamp:
            environment = get_environment(
                float(lat), float(lon), datetime.fromisoformat(timestamp), settings=settings,
            )
            wind, current = environment["wind"], environment["current"]
            provenance["wind_source"] = wind.source if wind is not None else None
            provenance["current_source"] = current.source if current is not None else None

    return provenance


def run(
    scene_id: Optional[str] = None,
    scene_path: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    stub_model: bool = False,
    tidetrace_checkpoint: Optional[Path] = None,
    ais_path: Optional[Path] = None,
    ais_source_label: str = "unspecified",
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    batch_size: Optional[int] = None,
    reuse_tiles: bool = False,
    output: Optional[Path] = None,
    write_map: bool = False,
    map_output: Optional[Path] = None,
    write_report: bool = False,
    report_output: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Run the whole chain - detection through attribution - for one scene.

    ``ais_path`` is optional: without it, an alerted spill still gets a full
    alert decision, just an empty vessel list rather than a crash - useful
    for running the pipeline before any AIS export is on hand.
    ``ais_source_label`` (step 8.2) is a free-text label for *which* export
    ``ais_path`` is (e.g. "NOAA Office for Coastal Management historical
    AIS") - :mod:`src.ais.loader` is deliberately source-agnostic and cannot
    report this itself, so it is threaded straight into the combined
    output's ``provenance`` object instead. Defaults to "unspecified", not
    None, so a provenance panel always has something to show once an AIS
    export was actually given.

    Every spill also gets a forward drift forecast (step 4.3, 6/12/24/48h -
    see :mod:`src.drift.forward`), independent of AIS/alert status, since it
    needs neither. If ``attribution.use_hindcasting`` is set, an alerted
    spill being attributed additionally gets a hindcast corridor (step 4.4 -
    see :mod:`src.drift.hindcast`) computed and threaded into scoring; it is
    not computed otherwise since nothing else would consume it.

    ``write_map`` additionally saves a basic Folium map (see
    :mod:`src.output.map`) of every spill, its drift forecast, and any
    candidate vessel tracks, alongside the combined JSON. ``write_report``
    (step 5.3, see :mod:`src.output.report`) additionally saves a standalone
    HTML incident report - spill summary, that same map embedded, the ranked
    vessel table with explanations, and the drift forecast - independent of
    ``write_map`` (the report builds and embeds its own map either way).
    """
    settings = settings or get_settings()

    detection = run_detection.run(
        scene_id=scene_id, scene_path=scene_path, checkpoint=checkpoint,
        stub_model=stub_model, tidetrace_checkpoint=tidetrace_checkpoint,
        tile_size=tile_size, overlap=overlap, batch_size=batch_size,
        reuse_tiles=reuse_tiles, settings=settings,
    )
    spills = detection.get("features", [])
    scene_id_resolved = (detection.get("properties") or {}).get("scene_id")
    print(f"\n[3.2] alert manager: {len(spills)} spill(s) to evaluate")

    ais = load_ais(ais_path) if ais_path else None
    if ais is not None:
        print(f"      AIS export loaded: {ais['mmsi'].nunique()} vessel(s), {len(ais)} report(s)")
    else:
        print("      no --ais given: alerted spills will have an empty vessel list")

    max_printed = 10
    results: List[Dict[str, Any]] = []
    alerted_count = 0
    for index, spill in enumerate(spills):
        with SpillRegistry(path=registry_path, settings=settings) as registry:
            decision = process_spill(spill, registry, settings=settings)
        if decision.alert:
            alerted_count += 1
        if index < max_printed:
            failed = [r.name for r in decision.rules if not r.passed]
            print(
                f"      {decision.spill_id or '(rejected)'}: {decision.status}"
                + (f" <- {', '.join(failed)}" if failed else "")
            )

        vessels: List[Dict[str, Any]] = []
        if decision.alert and ais is not None:
            hindcast_corridor = (
                hindcast_origin(spill, settings=settings)
                if settings.attribution.use_hindcasting else None
            )
            attribution = run_attribution.attribute_spill(
                spill, ais, settings=settings, hindcast_corridor=hindcast_corridor,
            )
            vessels = attribution["candidates"]
        elif decision.alert:
            logger.info(
                "spill %s alerted but no --ais given; vessel list left empty",
                decision.spill_id,
            )

        drift_forecast = _drift_forecast_payload(spill, settings)
        results.append({
            "spill": spill, "alert": decision.to_dict(), "vessels": vessels,
            "drift_forecast": drift_forecast,
        })

        # Step 10.2: attribution/drift only exist by this point in the loop
        # (register()/update() ran earlier, inside process_spill(), before
        # either existed) - so this is its own registry call, not folded
        # into that one. Postgres-only; a documented no-op on sqlite, but
        # skipped here entirely rather than even opening one, since sqlite
        # never needs it.
        if decision.alert and decision.spill_id and database_url(settings):
            with SpillRegistry(path=registry_path, settings=settings) as registry:
                registry.set_vessels_and_drift(
                    decision.spill_id,
                    vessels_to_contract(vessels),
                    {"forecast": forecast_to_contract(drift_forecast), "hindcast": []},
                )

    if len(spills) > max_printed:
        print(f"      ... {len(spills) - max_printed} more spill(s) evaluated")
    print(f"      {alerted_count} of {len(spills)} spill(s) alerted")
    print(f"[4.3] drift forecast computed for {len(spills)} spill(s) (6/12/24/48h)")

    # A scene where every candidate blob was rejected by the look-alike
    # filter (or none was found at all) has zero entries in `spills` - the
    # per-spill loop above never runs, so nothing about this scene is ever
    # registered anywhere, Postgres included (registration only happens
    # per-spill). That leaves a fully, honestly processed real scene with
    # no way to show up on the dashboard at all - not "shown at the wrong
    # place", just invisible. The same blind spot exists one level deeper:
    # `evaluate_alert()` returns `spill_id=None` (never calling
    # `registry.register()`) for *any* spill that fails the confidence or
    # area gate before ever reaching registration - a real, characterized
    # spill that is simply too small/low-confidence to alert on is just as
    # invisible in Postgres as a scene with none at all, even though
    # `spills` itself is non-empty. Both cases are fixed the same way:
    # register the best real evidence this scene actually has (never a
    # fabricated shape) as a status="none"/"rejected" record - "this scene
    # was processed, here is what was actually found, nothing was
    # confirmed" - rather than leaving a stale earlier run's row as the
    # only thing anyone ever sees.
    scene_centroid: Optional[Dict[str, float]] = None
    any_registered = any(r["alert"].get("spill_id") for r in results)
    if not any_registered:
        if not spills:
            scene = run_detection.resolve_scene(scene_id, scene_path, settings)
            # A satellite swath footprint is a large, skewed shape - averaging
            # its raw lon/lat degrees (geometry.centroid) lands measurably off
            # its true centre, unlike a single small spill polygon where that
            # error is negligible. geographic_centroid() reprojects to an
            # equal-area CRS first, same approach this pipeline already uses
            # for area.
            centroid = geographic_centroid(scene.footprint)
            scene_centroid = {"lat": centroid.y, "lon": centroid.x}
            placeholder_geometry = scene.footprint
            placeholder_status = "none"
            placeholder_confidence = 0.0
            placeholder_area = 0.0
            placeholder_seen_at = scene.acquisition_time
            placeholder_id = f"{scene_id_resolved}_no_spill"
        else:
            # Real spill(s) exist but every one was rejected pre-registration
            # (too small or too low-confidence to alert on) - show the
            # strongest real candidate rather than the scene's generic
            # footprint, since it is more informative and equally true.
            best = max(
                spills,
                key=lambda f: (f.get("properties") or {}).get("mean_confidence") or 0.0,
            )
            best_props = best.get("properties") or {}
            best_geometry = shape(best["geometry"])
            centroid = geographic_centroid(best_geometry)
            scene_centroid = {"lat": centroid.y, "lon": centroid.x}
            placeholder_geometry = best_geometry
            placeholder_status = "rejected"
            placeholder_confidence = float(best_props.get("mean_confidence") or 0.0)
            placeholder_area = float(best_props.get("area_km2") or 0.0)
            best_timestamp = best_props.get("acquisition_timestamp")
            placeholder_seen_at = (
                datetime.fromisoformat(best_timestamp) if best_timestamp
                else datetime.now(timezone.utc)
            )
            # best_props["spill_id"] already carries the scene_id prefix
            # (build_spill_object's own default), so use it as-is rather
            # than doubling the prefix.
            placeholder_id = best_props.get("spill_id") or f"{scene_id_resolved}_best_candidate"

        if database_url(settings):
            with SpillRegistry(path=registry_path, settings=settings) as registry:
                # A --force re-run of an already-processed scene hits this
                # same placeholder id again - register() raises on a
                # duplicate, so an existing row is updated in place instead
                # (same "seen before -> update" pattern evaluate_alert's own
                # dedup step already uses).
                placeholder_kwargs = dict(
                    geometry=placeholder_geometry,
                    centroid_lon=centroid.x,
                    centroid_lat=centroid.y,
                    area_km2=placeholder_area,
                    seen_at=placeholder_seen_at,
                    status=placeholder_status,
                    scene_id=scene_id_resolved,
                )
                if registry.get(placeholder_id) is not None:
                    registry.update(placeholder_id, **placeholder_kwargs)
                else:
                    registry.register(
                        spill_id=placeholder_id,
                        confidence=placeholder_confidence,
                        rules_fired=[],
                        **placeholder_kwargs,
                    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": scene_id_resolved,
        "stub_model": bool(stub_model),
        "ais_source": str(ais_path) if ais_path else None,
        "provenance": _provenance_payload(
            detection, scene_id_resolved, spills, ais_path, ais_source_label, settings,
        ),
        "spills": results,
        **({"scene_centroid": scene_centroid} if scene_centroid else {}),
    }

    processed_root = settings.paths.resolve(settings.paths.processed_dir)
    target = Path(output) if output else (
        processed_root / (scene_id_resolved or "scene") / "pipeline_result.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[3.3] wrote {target}\n")

    if write_map:
        map_target = Path(map_output) if map_output else target.with_name("map.html")
        try:
            save_map(payload, map_target)
            print(f"[3.4] wrote {map_target}\n")
        except MapBuildError as exc:
            logger.info("skipping map: %s", exc)

    if write_report:
        report_target = Path(report_output) if report_output else target.with_name("report.html")
        try:
            save_report(payload, report_target)
            print(f"[5.3] wrote {report_target}\n")
        except ReportBuildError as exc:
            logger.info("skipping report: %s", exc)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Phase 1-3 chain over one local scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-id", help="scene id as found under data/raw/")
    parser.add_argument("--scene", dest="scene_path", type=Path,
                        help="explicit path to a scene raster")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stub-model", action="store_true",
                        help="threshold stand-in, for wiring checks before training")
    parser.add_argument("--tidetrace-checkpoint", type=Path, default=None,
                        help="path to TideTrace's oil_unet_best.pt (real trained "
                             "UNet++ detector) - see src/detection/tidetrace.py")
    parser.add_argument("--ais", dest="ais_path", type=Path, default=None,
                        help="AIS export (.csv or .parquet); omit to skip attribution")
    parser.add_argument("--ais-source-label", type=str, default="unspecified",
                        help="free-text label for --ais's provenance (e.g. 'NOAA "
                             "Office for Coastal Management historical AIS')")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="inference batch size; defaults to detection.batch_size")
    parser.add_argument("--reuse-tiles", action="store_true",
                        help="skip preprocessing if this scene is already tiled")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--map", dest="write_map", action="store_true",
                        help="also save a basic Folium map (spill + vessel tracks)")
    parser.add_argument("--map-output", type=Path, default=None,
                        help="where the map HTML goes; defaults next to --output")
    parser.add_argument("--report", dest="write_report", action="store_true",
                        help="also save a standalone HTML incident report")
    parser.add_argument("--report-output", type=Path, default=None,
                        help="where the report HTML goes; defaults next to --output")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(
        scene_id=args.scene_id,
        scene_path=args.scene_path,
        checkpoint=args.checkpoint,
        stub_model=args.stub_model,
        tidetrace_checkpoint=args.tidetrace_checkpoint,
        ais_path=args.ais_path,
        ais_source_label=args.ais_source_label,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        reuse_tiles=args.reuse_tiles,
        output=args.output,
        write_map=args.write_map,
        map_output=args.map_output,
        write_report=args.write_report,
        report_output=args.report_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
