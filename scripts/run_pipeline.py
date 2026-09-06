"""Phase 3 end to end: a raw scene path in, alerts + ranked vessels out.

The first true end-to-end script: raw scene -> ingestion -> preprocessing ->
detection -> look-alike filter -> characterization -> alert manager ->
(if alerted) AIS query -> tracks -> filter -> scoring.

Reuses each earlier phase's own script as a **library** -
:func:`scripts.run_detection.run` for steps 1.1-1.6 and
:func:`scripts.run_attribution.attribute_spill` for steps 2.1-2.3 - rather
than reimplementing chain assembly that already exists and is already
tested. Nothing here is shelled out to; every step is a plain Python import
and function call.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_pipeline.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.ais.loader import load_ais  # noqa: E402
from src.alerts.manager import process_spill  # noqa: E402
from src.alerts.registry import SpillRegistry  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402
from src.output.map import MapBuildError, save_map  # noqa: E402

from scripts import run_attribution  # noqa: E402
from scripts import run_detection  # noqa: E402

logger = logging.getLogger(__name__)


def run(
    scene_id: Optional[str] = None,
    scene_path: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    stub_model: bool = False,
    ais_path: Optional[Path] = None,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    reuse_tiles: bool = False,
    output: Optional[Path] = None,
    write_map: bool = False,
    map_output: Optional[Path] = None,
    registry_path: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Run the whole chain - detection through attribution - for one scene.

    ``ais_path`` is optional: without it, an alerted spill still gets a full
    alert decision, just an empty vessel list rather than a crash - useful
    for running the pipeline before any AIS export is on hand.

    ``write_map`` additionally saves a basic Folium map (see
    :mod:`src.output.map`) of every spill and candidate vessel track,
    alongside the combined JSON.
    """
    settings = settings or get_settings()

    detection = run_detection.run(
        scene_id=scene_id, scene_path=scene_path, checkpoint=checkpoint,
        stub_model=stub_model, tile_size=tile_size, overlap=overlap,
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
    with SpillRegistry(path=registry_path, settings=settings) as registry:
        results: List[Dict[str, Any]] = []
        alerted_count = 0
        for index, spill in enumerate(spills):
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
                attribution = run_attribution.attribute_spill(spill, ais, settings=settings)
                vessels = attribution["candidates"]
            elif decision.alert:
                logger.info(
                    "spill %s alerted but no --ais given; vessel list left empty",
                    decision.spill_id,
                )

            results.append({"spill": spill, "alert": decision.to_dict(), "vessels": vessels})

        if len(spills) > max_printed:
            print(f"      ... {len(spills) - max_printed} more spill(s) evaluated")
        print(f"      {alerted_count} of {len(spills)} spill(s) alerted")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": scene_id_resolved,
        "stub_model": bool(stub_model),
        "ais_source": str(ais_path) if ais_path else None,
        "spills": results,
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
    parser.add_argument("--ais", dest="ais_path", type=Path, default=None,
                        help="AIS export (.csv or .parquet); omit to skip attribution")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--reuse-tiles", action="store_true",
                        help="skip preprocessing if this scene is already tiled")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--map", dest="write_map", action="store_true",
                        help="also save a basic Folium map (spill + vessel tracks)")
    parser.add_argument("--map-output", type=Path, default=None,
                        help="where the map HTML goes; defaults next to --output")
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
        ais_path=args.ais_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        reuse_tiles=args.reuse_tiles,
        output=args.output,
        write_map=args.write_map,
        map_output=args.map_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
