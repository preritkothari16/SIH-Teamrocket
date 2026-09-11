"""Run every not-yet-processed scene under ``data/raw/`` through the full
Phase 3 chain (:func:`scripts.run_pipeline.run`), stub model, so it shows up
in ``GET /api/runs`` / the live dashboard the same way every other processed
scene already does — no new plumbing, this only calls what already exists.

"Not yet processed" = no ``data/processed/<scene_id>/pipeline_result.json``
yet, so re-running this script is a safe no-op for scenes already done;
use ``--force`` to reprocess anyway.

Usage::

    .venv/Scripts/python.exe scripts/ingest_raw_scenes.py
    .venv/Scripts/python.exe scripts/ingest_raw_scenes.py --scene-id S1B_IW_...
    .venv/Scripts/python.exe scripts/ingest_raw_scenes.py --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/ingest_raw_scenes.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_settings  # noqa: E402
from src.ingestion.local_source import LocalSceneSource  # noqa: E402
from src.ingestion.types import Scene  # noqa: E402

from scripts.run_pipeline import run as run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


def already_processed(scene: Scene, processed_dir: Path) -> bool:
    return (processed_dir / scene.scene_id / "pipeline_result.json").is_file()


def ingest(
    scene_ids: Optional[List[str]] = None,
    force: bool = False,
    stub_model: bool = True,
) -> int:
    """Process every discovered (or explicitly named) scene. Returns the
    number of scenes that failed, so a caller/CI can treat a nonzero count
    as a real failure without this raising and losing partial progress."""
    settings = get_settings()
    processed_dir = settings.paths.resolve(settings.paths.processed_dir)

    source = LocalSceneSource(settings=settings)
    scenes = source.scenes()
    if scene_ids:
        wanted = set(scene_ids)
        scenes = [s for s in scenes if s.scene_id in wanted]
        missing = wanted - {s.scene_id for s in scenes}
        for scene_id in missing:
            logger.error("no local scene with id %r found under %s", scene_id, source.root)

    if not scenes:
        logger.warning("no scenes to process under %s", source.root)
        return 0

    failures = 0
    for scene in scenes:
        if not force and already_processed(scene, processed_dir):
            logger.info("skipping %s — already processed (pass --force to redo)", scene.scene_id)
            continue

        logger.info(
            "processing %s (%s, acquired %s) — this is a full-resolution real "
            "scene, expect a long run",
            scene.scene_id, scene.platform, scene.acquisition_time,
        )
        started = time.monotonic()
        try:
            run_pipeline(
                scene_id=scene.scene_id,
                stub_model=stub_model,
                ais_source_label="unspecified",
            )
        except Exception:
            failures += 1
            logger.exception("failed processing %s", scene.scene_id)
            continue
        elapsed = time.monotonic() - started
        logger.info("done %s in %.1fs — data/processed/%s/pipeline_result.json",
                    scene.scene_id, elapsed, scene.scene_id)

    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", dest="scene_ids", action="append",
                        help="process only this scene id (repeatable); default is every "
                             "not-yet-processed scene under data/raw/")
    parser.add_argument("--force", action="store_true",
                        help="reprocess even if data/processed/<scene_id>/pipeline_result.json exists")
    parser.add_argument("--no-stub-model", dest="stub_model", action="store_false",
                        help="use a real trained checkpoint instead of the stub — "
                             "only meaningful once models/best.pt actually exists")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    failures = ingest(scene_ids=args.scene_ids, force=args.force, stub_model=args.stub_model)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
