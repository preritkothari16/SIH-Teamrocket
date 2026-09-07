"""Step 5.1: continuously poll the CDSE catalogue for new scenes over an AOI
and run the full Phase 1-4 pipeline (:mod:`scripts.run_pipeline`) on each one
automatically.

A separate, optional entry point - the local pre-downloaded-scene demo path
(``python scripts/run_pipeline.py --scene <tif>`` against ``data/raw/``)
does not import this module and is unaffected by it either way.

Usage::

    python scripts/run_live.py --aoi configs/aoi.geojson --interval 900 --stub-model
    python scripts/run_live.py --aoi configs/aoi.geojson --interval 900 --max-cycles 1

Needs CDSE credentials (see ``.env.example``) to actually reach the network;
:func:`run_forever` accepts any :class:`~src.ingestion.types.SceneSource` in
its place, which is how ``tests/test_poller.py`` exercises this without one.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_live.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.ingestion.catalogue import CDSECatalogue  # noqa: E402
from src.ingestion.poller import load_last_checked, poll_once, save_last_checked  # noqa: E402
from src.ingestion.types import SceneSource, load_aoi  # noqa: E402

from scripts import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("data/ingestion_poll_state.json")


def run_forever(
    aoi_path: Path,
    interval_seconds: float,
    catalogue: Optional[SceneSource] = None,
    settings: Optional[Settings] = None,
    state_path: Optional[Path] = None,
    max_cycles: Optional[int] = None,
    stub_model: bool = False,
    ais_path: Optional[Path] = None,
    write_map: bool = False,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    registry_path: Optional[Path] = None,
) -> int:
    """Poll every ``interval_seconds``, running the Phase 1-4 pipeline on
    every new scene each cycle finds. Returns the total number of scenes run.

    Runs forever when ``max_cycles`` is None (the real, live use); a finite
    ``max_cycles`` is what lets a test exercise this deterministically.
    ``catalogue`` defaults to a real, network-hitting
    :class:`~src.ingestion.catalogue.CDSECatalogue`.
    """
    settings = settings or get_settings()
    aoi = load_aoi(aoi_path)
    state_path = Path(state_path) if state_path else settings.paths.resolve(DEFAULT_STATE_PATH)
    catalogue = catalogue or CDSECatalogue(settings=settings)

    since = load_last_checked(
        state_path, default_lookback_hours=settings.ingestion.search_lookback_days * 24.0,
    )
    total_processed = 0
    cycle = 0
    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        try:
            result = poll_once(aoi, since, catalogue=catalogue, settings=settings)
        except Exception:
            logger.exception("poll failed (cycle %d); retrying next interval", cycle)
            if max_cycles is None or cycle < max_cycles:
                time.sleep(interval_seconds)
            continue

        print(
            f"[5.1] poll {cycle}: {len(result.new_scenes)} new scene(s) "
            f"since {since.isoformat()}"
        )

        all_succeeded = True
        for scene in result.new_scenes:
            print(f"[5.1] running pipeline for scene {scene.scene_id}")
            try:
                run_pipeline.run(
                    scene_path=scene.path, stub_model=stub_model, ais_path=ais_path,
                    tile_size=tile_size, overlap=overlap, write_map=write_map,
                    registry_path=registry_path, settings=settings,
                )
                total_processed += 1
            except Exception:
                all_succeeded = False
                logger.exception(
                    "pipeline failed for scene %s; will retry on next poll",
                    scene.scene_id,
                )

        if all_succeeded:
            since = result.checked_at
            save_last_checked(state_path, since)

        if max_cycles is None or cycle < max_cycles:
            time.sleep(interval_seconds)

    return total_processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll CDSE for new scenes over an AOI and run the pipeline on each one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--aoi", dest="aoi_path", type=Path, required=True,
                        help="AOI GeoJSON (Feature, FeatureCollection, or bare geometry)")
    parser.add_argument("--interval", dest="interval_seconds", type=float, default=900.0,
                        help="seconds between polls")
    parser.add_argument("--state-path", type=Path, default=None,
                        help="where the last-checked timestamp is persisted")
    parser.add_argument("--max-cycles", type=int, default=None,
                        help="stop after this many polls instead of running forever")
    parser.add_argument("--stub-model", action="store_true",
                        help="threshold stand-in, for wiring checks before training")
    parser.add_argument("--ais", dest="ais_path", type=Path, default=None,
                        help="AIS export (.csv or .parquet); omit to skip attribution")
    parser.add_argument("--map", dest="write_map", action="store_true",
                        help="also save a basic Folium map per scene")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--registry", dest="registry_path", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_forever(
        aoi_path=args.aoi_path,
        interval_seconds=args.interval_seconds,
        state_path=args.state_path,
        max_cycles=args.max_cycles,
        stub_model=args.stub_model,
        ais_path=args.ais_path,
        write_map=args.write_map,
        tile_size=args.tile_size,
        overlap=args.overlap,
        registry_path=args.registry_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
