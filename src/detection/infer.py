"""Run the segmentation model over a preprocessed scene and stitch it back.

Step 1.2 cut a scene into overlapping 512 px tiles and recorded where each one
came from in ``tile_index.parquet``. This module runs the model over those
tiles and reassembles a single full-scene mask carrying the scene's original
geotransform, so every predicted pixel is back on the map.

Overlap handling
----------------
**Class probabilities are averaged wherever tiles overlap**, not trimmed.

The alternative - cropping each tile back to its non-overlapping core - throws
away exactly the predictions most worth keeping: a slick lying on a tile
boundary is seen whole by one tile and cut in half by its neighbour, and the
whole reason Step 1.2 overlaps tiles is to guarantee that. Averaging is a soft
vote that uses both views, and it removes the visible seams that trimming
leaves behind at tile edges, where a convolutional model has the least context
and is least reliable. The cost is one float32 accumulator of shape
(classes, height, width) - 84 MB for a 5-class 2048x2048 scene, which is fine
at these sizes but is the thing to revisit for very large mosaics.

Domain gap
----------
The model is trained on 8-bit JPEG patches; tiles here are float32 dB. The dB
window in ``detection.input_db_min/max`` is mapped onto 0-255 to put inference
inputs back in the range the model saw during training. **This mapping is an
assumption, not a calibration** - revisit it once a checkpoint is trained on
real preprocessed tiles.

Running it
----------
Needs a trained checkpoint at ``models/best.pt``; nothing has trained one yet
(no GPU and no labelled dataset in this environment), so this module is written
against that path and is exercised in tests with a stub model::

    python -m src.detection.infer --scene-id 00000
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS

from src.config import Settings, get_settings
from src.detection.dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, OIL_CLASS
from src.detection.model import load_checkpoint
from src.preprocessing.pipeline import transform_from_index_row

logger = logging.getLogger(__name__)

TILE_INDEX_NAME = "tile_index.parquet"

#: Written into the mask where no tile covered a pixel, or the tile was nodata.
NODATA_CLASS = 255


class InferenceError(RuntimeError):
    """Inference could not run over a scene."""


@dataclass
class SceneGrid:
    """The full-scene raster the tiles were cut from."""

    height: int
    width: int
    transform: Affine
    crs: Optional[CRS]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class InferenceResult:
    """A stitched full-scene prediction."""

    scene_id: str
    mask: np.ndarray  # (H, W) uint8 class indices, NODATA_CLASS where uncovered
    probabilities: np.ndarray  # (C, H, W) float32, averaged across overlaps
    grid: SceneGrid
    class_names: Sequence[str]
    mask_path: Optional[Path] = None
    tiles: int = 0

    @property
    def confidence(self) -> np.ndarray:
        """Winning class probability per pixel."""
        return self.probabilities.max(axis=0)

    def class_fraction(self, class_index: int) -> float:
        valid = self.mask != NODATA_CLASS
        if not valid.any():
            return 0.0
        return float((self.mask[valid] == class_index).mean())


# --------------------------------------------------------------------------- #
# reading what preprocessing wrote
# --------------------------------------------------------------------------- #
def load_tile_index(scene_dir: Path) -> pd.DataFrame:
    """Read ``tile_index.parquet`` for a preprocessed scene."""
    path = Path(scene_dir) / TILE_INDEX_NAME
    if not path.is_file():
        raise InferenceError(
            f"no {TILE_INDEX_NAME} in {scene_dir}; run the preprocessing pipeline "
            "for this scene first"
        )
    frame = pd.read_parquet(path)
    if frame.empty:
        raise InferenceError(f"{path} is empty - the scene produced no tiles")
    return frame


def scene_grid_from_index(frame: pd.DataFrame) -> SceneGrid:
    """Rebuild the parent scene's size and geotransform from the tile index.

    Tiling preserves the parent transform, so the tile at pixel offset (0, 0)
    carries the scene origin, and the extent is the furthest tile edge.
    """
    height = int((frame["row_off"] + frame["height"]).max())
    width = int((frame["col_off"] + frame["width"]).max())

    origin = frame[(frame["row_off"] == 0) & (frame["col_off"] == 0)]
    if origin.empty:
        # No corner tile (it can be dropped for being mostly land): reconstruct
        # the origin by walking back from any tile's own transform.
        row = frame.iloc[0]
        tile_transform = transform_from_index_row(row)
        transform = tile_transform * Affine.translation(
            -float(row["col_off"]), -float(row["row_off"])
        )
    else:
        transform = transform_from_index_row(origin.iloc[0])

    crs_value = frame["crs"].iloc[0]
    crs = CRS.from_string(crs_value) if isinstance(crs_value, str) and crs_value else None
    return SceneGrid(height=height, width=width, transform=transform, crs=crs)


def tile_paths(frame: pd.DataFrame, scene_dir: Path) -> List[Path]:
    """Where each tile's GeoTIFF lives, honouring the index's own paths."""
    scene_dir = Path(scene_dir)
    if "path" in frame.columns and frame["path"].notna().all():
        return [Path(p) for p in frame["path"]]
    return [scene_dir / f"{tile_id}.tif" for tile_id in frame["tile_id"]]


def read_tile(path: Path) -> np.ndarray:
    """Read one preprocessed tile as ``(bands, rows, cols)`` float32 dB."""
    with rasterio.open(path) as dataset:
        return dataset.read().astype(np.float32)


# --------------------------------------------------------------------------- #
# model input
# --------------------------------------------------------------------------- #
def db_to_model_input(
    tile: np.ndarray,
    in_channels: int = 3,
    db_min: float = -35.0,
    db_max: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """Map a float32 dB tile onto the input distribution the model was trained on.

    Clips to the dB window, rescales it to 0-255 to match the 8-bit training
    patches, then applies the same ImageNet normalisation the dataset used.
    Invalid pixels (NaN land/nodata) become the window's floor, which reads as
    very dark - the model's own "dark" class assignment there is discarded
    afterwards using the validity mask, so the fill value never reaches output.
    """
    scaled = (np.nan_to_num(tile, nan=db_min) - db_min) / max(db_max - db_min, 1e-6)
    scaled = np.clip(scaled, 0.0, 1.0)

    bands = scaled.shape[0]
    if bands < in_channels:
        # For 2-band (VH/VV) tiles with a 3-channel model, repeat the
        # second band (VV) rather than the first — VV carries more oil
        # contrast. For 1-band tiles, repeat what we have.
        repeat_band = scaled[min(1, bands - 1):min(1, bands - 1) + 1]
        padding = np.repeat(repeat_band, in_channels - bands, axis=0)
        scaled = np.vstack([scaled, padding])
    elif bands > in_channels:
        scaled = scaled[:in_channels]

    if normalize:
        mean = np.asarray(IMAGENET_MEAN[:in_channels], dtype=np.float32)
        std = np.asarray(IMAGENET_STD[:in_channels], dtype=np.float32)
        scaled = (scaled - mean[:, None, None]) / std[:, None, None]
    return scaled.astype(np.float32)


def tile_valid_mask(tile: np.ndarray) -> np.ndarray:
    """True where a tile carries real backscatter in every band."""
    return np.isfinite(tile).all(axis=0)


# --------------------------------------------------------------------------- #
# prediction + stitching
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_tiles(
    model: Callable[[torch.Tensor], torch.Tensor],
    tiles: Sequence[np.ndarray],
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> np.ndarray:
    """Class probabilities for a list of prepared tiles: ``(N, C, H, W)``.

    ``model`` is anything callable that maps a batch of tiles to logits, which
    is what lets the stitching be tested without a trained checkpoint.
    """
    if not tiles:
        raise InferenceError("no tiles to predict")

    device = device or torch.device("cpu")
    if hasattr(model, "eval"):
        model.eval()

    outputs: List[np.ndarray] = []
    for start in range(0, len(tiles), batch_size):
        batch = np.stack(tiles[start : start + batch_size])
        logits = model(torch.from_numpy(batch).to(device))
        outputs.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def stitch_tiles(
    probabilities: Sequence[np.ndarray],
    frame: pd.DataFrame,
    grid: SceneGrid,
    valid_masks: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reassemble per-tile probabilities into one full-scene array.

    Overlapping tiles are averaged (see the module docstring). Returns the
    averaged ``(C, H, W)`` probabilities and the ``(H, W)`` coverage count, which
    is zero wherever no tile contributed.
    """
    if len(probabilities) != len(frame):
        raise InferenceError(
            f"{len(probabilities)} prediction(s) for {len(frame)} tile(s) in the index"
        )

    num_classes = int(np.asarray(probabilities[0]).shape[0])
    total = np.zeros((num_classes, grid.height, grid.width), dtype=np.float32)
    coverage = np.zeros(grid.shape, dtype=np.float32)

    for position, (_, row) in enumerate(frame.iterrows()):
        patch = np.asarray(probabilities[position], dtype=np.float32)
        row_off, col_off = int(row["row_off"]), int(row["col_off"])
        height, width = int(row["height"]), int(row["width"])
        patch = patch[:, :height, :width]

        weight = np.ones((height, width), dtype=np.float32)
        if valid_masks is not None:
            weight *= np.asarray(valid_masks[position], dtype=np.float32)[
                :height, :width
            ]

        window = (
            slice(row_off, row_off + height),
            slice(col_off, col_off + width),
        )
        total[:, window[0], window[1]] += patch * weight
        coverage[window] += weight

    with np.errstate(invalid="ignore", divide="ignore"):
        averaged = np.where(coverage > 0, total / np.maximum(coverage, 1e-6), 0.0)
    return averaged.astype(np.float32), coverage


def mask_from_probabilities(
    probabilities: np.ndarray, coverage: np.ndarray
) -> np.ndarray:
    """Argmax to class indices, marking uncovered pixels as nodata."""
    mask = probabilities.argmax(axis=0).astype(np.uint8)
    return np.where(coverage > 0, mask, np.uint8(NODATA_CLASS)).astype(np.uint8)


def stitch_backscatter(
    scene_dir: Path,
    frame: Optional[pd.DataFrame] = None,
    grid: Optional[SceneGrid] = None,
) -> np.ndarray:
    """Reconstruct the scene's preprocessed backscatter on the mask's own grid.

    Reads the same despeckled, geocoded, land-masked tiles the model saw and
    averages overlaps exactly like :func:`stitch_tiles` does for class
    probabilities, so whatever consumes this (the look-alike filter) measures
    blobs against backscatter that is pixel-for-pixel aligned with the
    stitched mask - never a fresh, ungeocoded read of the raw scene, which
    would silently misalign the two the moment geocoding actually reprojects.

    Pass ``grid`` (e.g. ``InferenceResult.grid``) to guarantee the result
    shares the exact transform/CRS/shape the mask was built on rather than one
    independently rebuilt from the tile index.

    Uncovered pixels - no tile reached them, or every tile that did was
    entirely land/nodata there - come back as NaN, not zero: a fabricated 0 dB
    fill would let land silently pollute the look-alike filter's contrast ring
    around a coastal blob.
    """
    scene_dir = Path(scene_dir)
    if frame is None:
        frame = load_tile_index(scene_dir)
    if grid is None:
        grid = scene_grid_from_index(frame)

    paths = tile_paths(frame, scene_dir)
    tiles = [read_tile(path) for path in paths]
    valid_masks = [tile_valid_mask(t) for t in tiles]

    stitched, coverage = stitch_tiles(tiles, frame, grid, valid_masks)
    return np.where(coverage > 0, stitched, np.float32(np.nan)).astype(np.float32)


# --------------------------------------------------------------------------- #
# the whole scene
# --------------------------------------------------------------------------- #
def infer_scene(
    scene_dir: Path,
    model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    checkpoint: Optional[Path] = None,
    scene_id: Optional[str] = None,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    output_path: Optional[Path] = None,
    save: bool = True,
    settings: Optional[Settings] = None,
) -> InferenceResult:
    """Predict over every tile of a preprocessed scene and stitch the result.

    Pass ``model`` to use an already-loaded network (or a stub, in tests);
    otherwise a checkpoint is loaded from ``checkpoint`` or
    ``detection.checkpoint_path``.
    """
    settings = settings or get_settings()
    cfg = settings.detection
    scene_dir = Path(scene_dir)
    scene_id = scene_id or scene_dir.name
    batch_size = batch_size or cfg.batch_size

    frame = load_tile_index(scene_dir)
    grid = scene_grid_from_index(frame)
    paths = tile_paths(frame, scene_dir)

    if model is None:
        model = load_model(checkpoint, settings=settings)

    in_channels = settings.training.in_channels
    prepared: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    for path in paths:
        tile = read_tile(path)
        valid_masks.append(tile_valid_mask(tile))
        prepared.append(
            db_to_model_input(
                tile,
                in_channels=in_channels,
                db_min=cfg.input_db_min,
                db_max=cfg.input_db_max,
            )
        )

    probabilities = predict_tiles(model, prepared, device=device, batch_size=batch_size)
    stitched, coverage = stitch_tiles(probabilities, frame, grid, valid_masks)
    mask = mask_from_probabilities(stitched, coverage)

    class_names = list(settings.training.class_names)[: stitched.shape[0]]
    logger.info(
        "scene %s: %d tile(s) -> %dx%d mask, %.3f%% oil",
        scene_id, len(paths), grid.height, grid.width,
        100.0 * float((mask == OIL_CLASS).mean()),
    )

    result = InferenceResult(
        scene_id=scene_id,
        mask=mask,
        probabilities=stitched,
        grid=grid,
        class_names=class_names,
        tiles=len(paths),
    )

    if save:
        target = Path(output_path) if output_path else scene_dir / f"{scene_id}_mask.tif"
        result.mask_path = save_mask(target, result)
    return result


def load_model(
    checkpoint: Optional[Path] = None,
    device: Optional[torch.device] = None,
    settings: Optional[Settings] = None,
) -> torch.nn.Module:
    """Load the trained network through the model factory."""
    settings = settings or get_settings()
    path = Path(checkpoint) if checkpoint else settings.paths.resolve(
        settings.detection.checkpoint_path
    )
    if not path.is_file():
        raise InferenceError(
            f"no checkpoint at {path}. Train one with "
            "`python -m src.detection.train`, or pass an explicit --checkpoint."
        )
    model, payload = load_checkpoint(path, map_location=device or "cpu", settings=settings)
    logger.info("loaded checkpoint %s (epoch %s)", path, payload.get("epoch"))
    return model


def save_mask(path: Path, result: InferenceResult) -> Path:
    """Write the stitched mask as a GeoTIFF on the scene's own geotransform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=result.grid.height,
        width=result.grid.width,
        count=1,
        dtype="uint8",
        crs=result.grid.crs,
        transform=result.grid.transform,
        nodata=NODATA_CLASS,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(result.mask, 1)
        dst.set_band_description(1, "class_index")
        dst.update_tags(
            scene_id=result.scene_id,
            classes=",".join(result.class_names),
            tiles=str(result.tiles),
        )
    logger.info("wrote %s", path)
    return path


def save_confidence(path: Path, result: InferenceResult) -> Path:
    """Write the winning-class probability per pixel, for inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=result.grid.height, width=result.grid.width,
        count=1, dtype="float32", crs=result.grid.crs, transform=result.grid.transform,
        nodata=float("nan"), compress="deflate", tiled=True,
    ) as dst:
        dst.write(result.confidence.astype(np.float32), 1)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run segmentation over a preprocessed scene and stitch it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene-id", help="folder name under data/processed/")
    parser.add_argument("--scene-dir", type=Path, help="explicit scene directory")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    if args.scene_dir:
        scene_dir = args.scene_dir
    elif args.scene_id:
        scene_dir = settings.paths.resolve(settings.paths.processed_dir) / args.scene_id
    else:
        print("give --scene-id or --scene-dir", file=sys.stderr)
        return 2

    result = infer_scene(
        scene_dir,
        checkpoint=args.checkpoint,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        output_path=args.output,
        settings=settings,
    )
    print(f"{result.scene_id}: {result.tiles} tiles -> {result.mask_path}")
    for index, name in enumerate(result.class_names):
        print(f"  {name:<12} {100.0 * result.class_fraction(index):6.2f}%")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


__all__ = [
    "NODATA_CLASS",
    "InferenceError",
    "InferenceResult",
    "SceneGrid",
    "infer_scene",
    "load_model",
    "load_tile_index",
    "scene_grid_from_index",
    "tile_paths",
    "read_tile",
    "db_to_model_input",
    "tile_valid_mask",
    "predict_tiles",
    "stitch_tiles",
    "stitch_backscatter",
    "mask_from_probabilities",
    "save_mask",
    "save_confidence",
    "main",
]
