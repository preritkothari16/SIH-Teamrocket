"""Controlled stub-vs-TideTrace comparison over one already-tiled scene.

Runs both detectors through the exact same downstream steps (stitching,
look-alike filter, characterisation) that a real pipeline run uses, so the
region counts/areas/centroids reported here are the real numbers each
detector would actually produce, not an approximation. Writes nothing to
data/processed/ or the registry/database - this is validation only (step
12 of the TideTrace integration), never a substitute for the real
scripts/run_pipeline.py run.

Usage::

    .venv/Scripts/python.exe scripts/compare_detectors.py --scene-id 00000
    .venv/Scripts/python.exe scripts/compare_detectors.py \
        --scene-id S1D_IW_GRDH_1SDV_... --tidetrace-checkpoint models/tidetrace_oil_unet_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.characterization.spill_object import spills_from_blobs  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.detection.dataset import OIL_CLASS  # noqa: E402
from src.detection.infer import NODATA_CLASS, infer_scene, stitch_backscatter  # noqa: E402
from src.detection.lookalike_filter import filter_lookalikes  # noqa: E402
from src.detection.tidetrace import load_tidetrace_model  # noqa: E402
from src.ingestion.local_source import LocalSceneSource  # noqa: E402

from scripts.run_detection import ThresholdStubModel  # noqa: E402


def _run_one(label: str, scene_dir: Path, scene_id: str, model, prepare_fn, settings) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = infer_scene(
        scene_dir, model=model, scene_id=scene_id, settings=settings,
        save=False, prepare_fn=prepare_fn,
    )
    infer_s = time.perf_counter() - t0

    backscatter = stitch_backscatter(scene_dir, grid=result.grid)
    filtered = filter_lookalikes(
        result.mask, backscatter, confidence=result.confidence, settings=settings,
    )
    features = spills_from_blobs(
        filtered, transform=result.grid.transform, scene_id=scene_id,
        acquisition_timestamp=None, crs=result.grid.crs.to_string() if result.grid.crs else "EPSG:4326",
        settings=settings,
    )
    total_s = time.perf_counter() - t0

    covered = result.mask != NODATA_CLASS
    oil_fraction = result.class_fraction(OIL_CLASS)

    regions = []
    for f in features:
        props = f["properties"]
        geom = f["geometry"]
        centroid = None
        if geom and geom.get("type") == "Polygon" and geom.get("coordinates"):
            from shapely.geometry import shape
            centroid = list(shape(geom).centroid.coords)[0]
        regions.append({
            "area_km2": props.get("area_km2"),
            "mean_confidence": props.get("mean_confidence"),
            "bbox": props.get("bbox"),
            "centroid_lon_lat": centroid,
        })

    return {
        "label": label,
        "inference_seconds": round(infer_s, 3),
        "total_seconds": round(total_s, 3),
        "tiles": result.tiles,
        "oil_pixel_fraction": round(oil_fraction, 5),
        "covered_fraction": round(float(covered.mean()), 5),
        "blobs_examined": len(filtered.blobs),
        "blobs_rejected": len(filtered.rejected),
        "regions_after_filter": len(features),
        "regions": regions,
    }


def compare(scene_id: str, tidetrace_checkpoint: Path) -> Dict[str, Any]:
    settings = get_settings()
    processed_root = settings.paths.resolve(settings.paths.processed_dir)
    scene_dir = processed_root / scene_id
    if not (scene_dir / "tile_index.parquet").is_file():
        raise SystemExit(
            f"{scene_dir} has no tile_index.parquet - run the pipeline (with "
            "--reuse-tiles once tiled) before comparing detectors on it"
        )

    stub = ThresholdStubModel(num_classes=settings.training.num_classes)
    stub_result = _run_one("stub", scene_dir, scene_id, stub, None, settings)

    detector = load_tidetrace_model(tidetrace_checkpoint)
    tt_result = _run_one("tidetrace", scene_dir, scene_id, detector, detector.prepare, settings)

    return {"scene_id": scene_id, "stub": stub_result, "tidetrace": tt_result}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--tidetrace-checkpoint", type=Path,
                        default=Path("models/tidetrace_oil_unet_best.pt"))
    args = parser.parse_args(argv)

    result = compare(args.scene_id, args.tidetrace_checkpoint)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
