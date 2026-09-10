"""Phase 1 end to end: a local scene in, a spill GeoJSON out.

Chains every step built so far for one scene::

    1.1 ingestion       LocalSceneSource reads the scene off disk
    1.2 preprocessing   calibrate -> despeckle -> geocode -> mask land -> tile
    1.4 inference       segment every tile, stitch back to a full-scene mask
    1.5 look-alike      drop dark patches that do not behave like oil
    1.6 characterization polygonise, measure, emit GeoJSON

Writes ``data/processed/spills/<scene_id>.geojson`` - an empty
FeatureCollection when nothing survives the filter, because a scene that was
checked and found clean must be distinguishable from one never processed.

Usage::

    python scripts/run_detection.py --scene-id 00000
    python scripts/run_detection.py --scene D:/data/some_scene.tif
    python scripts/run_detection.py --scene-id 00000 --stub-model

Inference needs a trained checkpoint at ``models/best.pt``. Nothing has trained
one yet, so ``--stub-model`` runs the same chain with a threshold stand-in that
calls dark water oil - enough to prove the wiring end to end, and not a
detector. It says so loudly in its output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_detection.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Settings, get_settings  # noqa: E402
from src.characterization.spill_object import (  # noqa: E402
    feature_collection,
    spills_from_blobs,
)
from src.detection.dataset import CLASS_NAMES, OIL_CLASS  # noqa: E402
from src.detection.infer import (  # noqa: E402
    NODATA_CLASS,
    infer_scene,
    load_model,
    stitch_backscatter,
)
from src.detection.lookalike_filter import filter_lookalikes  # noqa: E402
from src.env_data.wind import get_wind  # noqa: E402
from src.env_data.grid import EnvDataError  # noqa: E402
from src.ingestion.local_source import LocalSceneSource  # noqa: E402
from src.ingestion.types import Scene  # noqa: E402
from src.preprocessing.pipeline import run_pipeline  # noqa: E402

logger = logging.getLogger(__name__)


class ThresholdStubModel(torch.nn.Module):
    """Stand-in for the trained network: dark pixels are called oil.

    Exists so the Phase 1 chain can be run and demonstrated before a checkpoint
    exists. It is a threshold, not a detector - it knows nothing about texture,
    shape or context, and every look-alike in the scene will sail straight
    through it into the filter.
    """

    def __init__(self, num_classes: int = 5, dark_quantile: float = 0.15) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dark_quantile = dark_quantile

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        intensity = batch.mean(dim=1, keepdim=True)
        cutoff = torch.quantile(
            intensity.flatten(1), self.dark_quantile, dim=1
        ).view(-1, 1, 1, 1)

        logits = torch.zeros(
            batch.shape[0], self.num_classes, batch.shape[2], batch.shape[3],
            dtype=batch.dtype, device=batch.device,
        )
        dark = (intensity <= cutoff).squeeze(1)
        logits[:, 0] = torch.where(dark, -4.0, 4.0)          # sea
        logits[:, OIL_CLASS] = torch.where(dark, 4.0, -4.0)  # oil
        return logits


def resolve_scene(
    scene_id: Optional[str],
    scene_path: Optional[Path],
    settings: Settings,
) -> Scene:
    """Find the scene to process, by explicit path or by id under data/raw/."""
    source = LocalSceneSource(settings=settings)
    if scene_path:
        return source.read_scene(Path(scene_path))
    if scene_id:
        return source.get(scene_id)

    scenes = source.scenes()
    if not scenes:
        raise SystemExit(
            f"no scenes found under {source.root}. Put a GeoTIFF there, or pass "
            "--scene /path/to/scene.tif"
        )
    logger.info("no scene given; using the most recent one under %s", source.root)
    return scenes[0]


def run(
    scene_id: Optional[str] = None,
    scene_path: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    stub_model: bool = False,
    output: Optional[Path] = None,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    reuse_tiles: bool = False,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Run Phase 1 for one scene and return the FeatureCollection written."""
    settings = settings or get_settings()

    scene = resolve_scene(scene_id, scene_path, settings)
    print(f"\n[1.1] scene {scene.scene_id}  {scene.acquisition_time.isoformat()}")
    print(f"      {scene.path}")

    processed_root = settings.paths.resolve(settings.paths.processed_dir)
    scene_dir = processed_root / scene.scene_id

    if reuse_tiles and (scene_dir / "tile_index.parquet").is_file():
        print(f"[1.2] reusing the tiles already in {scene_dir}")
    else:
        preprocessed = run_pipeline(
            scene, output_dir=scene_dir, tile_size=tile_size, overlap=overlap,
            settings=settings,
        )
        print(f"[1.2] {len(preprocessed.tiles)} tile(s) -> {scene_dir}")

    if stub_model:
        print("[1.4] STUB MODEL: dark pixels called oil. This is not a detector.")
        model = ThresholdStubModel(num_classes=settings.training.num_classes)
    else:
        model = load_model(checkpoint, settings=settings)

    result = infer_scene(
        scene_dir, model=model, scene_id=scene.scene_id, settings=settings
    )
    covered = result.mask != NODATA_CLASS
    print(
        f"[1.4] stitched {result.tiles} tile(s) -> {result.mask.shape[0]}x"
        f"{result.mask.shape[1]}, {100.0 * result.class_fraction(OIL_CLASS):.2f}% oil"
    )

    backscatter = stitch_backscatter(scene_dir, grid=result.grid)
    wind_speed: Optional[float] = None
    try:
        centroid = scene.footprint.centroid
        wind_data = get_wind(
            centroid.y, centroid.x, scene.acquisition_time, settings=settings,
        )
        if wind_data is not None:
            wind_speed = wind_data.vector.speed_ms
    except (EnvDataError, Exception):
        pass  # unavailable wind is a fact, not an error
    filtered = filter_lookalikes(
        result.mask, backscatter, confidence=result.confidence,
        wind_speed=wind_speed, settings=settings,
    )
    print(f"[1.5] {filtered.summary()}")

    features = spills_from_blobs(
        filtered,
        transform=result.grid.transform,
        scene_id=scene.scene_id,
        acquisition_timestamp=scene.acquisition_time,
        crs=result.grid.crs.to_string() if result.grid.crs else "EPSG:4326",
        settings=settings,
    )
    collection = feature_collection(
        features,
        scene_id=scene.scene_id,
        acquisition_timestamp=scene.acquisition_time,
        extra={
            "source": scene.source.value,
            "tiles": result.tiles,
            "blobs_examined": len(filtered.blobs),
            "blobs_rejected": len(filtered.rejected),
            "stub_model": bool(stub_model),
            "covered_fraction": round(float(covered.mean()), 4),
        },
    )

    target = Path(output) if output else (
        processed_root / settings.characterization.spills_dirname
        / f"{scene.scene_id}.geojson"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    total_area = sum(f["properties"]["area_km2"] for f in features)
    print(f"[1.6] {len(features)} spill(s), {total_area:.3f} km² -> {target}\n")
    return collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 detection chain over one local scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-id", help="scene id as found under data/raw/")
    parser.add_argument("--scene", dest="scene_path", type=Path,
                        help="explicit path to a scene raster")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stub-model", action="store_true",
                        help="threshold stand-in, for wiring checks before training")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--reuse-tiles", action="store_true",
                        help="skip preprocessing if this scene is already tiled")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list] = None) -> int:
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
        output=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        reuse_tiles=args.reuse_tiles,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
