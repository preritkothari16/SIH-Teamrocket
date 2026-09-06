"""Inference and stitching tests.

The stitching is tested with a stub "model" that returns a fixed class per
tile, against a synthetic 2x2 tile index. No trained checkpoint is needed
anywhere here - the point is that pixels land where the index says they should.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.detection.infer import (
    NODATA_CLASS,
    InferenceError,
    SceneGrid,
    db_to_model_input,
    infer_scene,
    load_model,
    load_tile_index,
    mask_from_probabilities,
    predict_tiles,
    read_tile,
    save_mask,
    scene_grid_from_index,
    stitch_backscatter,
    stitch_tiles,
    tile_paths,
    tile_valid_mask,
)
from src.preprocessing.pipeline import RasterImage, tile, tile_index_to_frame

ORIGIN_LON, ORIGIN_LAT = 70.0, 22.0
PIXEL_DEG = 0.001
NUM_CLASSES = 5


def scene_transform() -> Affine:
    return from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG)


# --------------------------------------------------------------------------- #
# stub models
# --------------------------------------------------------------------------- #
class FixedClassModel(torch.nn.Module):
    """Returns a preset class for each tile, in the order tiles arrive."""

    def __init__(self, classes: List[int], num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.classes = list(classes)
        self.num_classes = num_classes
        self.seen = 0

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        n, _, h, w = batch.shape
        logits = torch.full((n, self.num_classes, h, w), -5.0)
        for i in range(n):
            logits[i, self.classes[self.seen + i]] = 5.0
        self.seen += n
        return logits


class ConstantProbabilityModel(torch.nn.Module):
    """Emits fixed logits, so the averaging arithmetic is predictable."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logit_vector = logits

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        n, _, h, w = batch.shape
        return self.logit_vector.view(1, -1, 1, 1).expand(n, -1, h, w).clone()


# --------------------------------------------------------------------------- #
# fixtures: a synthetic 2x2 tile grid
# --------------------------------------------------------------------------- #
def make_index(
    tile_size: int = 32, overlap: int = 0, scene_px: int = 64
) -> pd.DataFrame:
    """A tile index built by the real tiler, so it matches what Step 1.2 writes."""
    image = RasterImage(
        data=np.zeros((1, scene_px, scene_px), dtype=np.float32),
        transform=scene_transform(),
        crs=CRS.from_epsg(4326),
        scene_id="synthetic",
    )
    _, index = tile(image, tile_size=tile_size, overlap=overlap)
    return tile_index_to_frame(index)


@pytest.fixture
def index_2x2() -> pd.DataFrame:
    """Four 32 px tiles covering a 64x64 scene, no overlap."""
    frame = make_index()
    assert len(frame) == 4
    return frame


@pytest.fixture
def grid_2x2(index_2x2: pd.DataFrame) -> SceneGrid:
    return scene_grid_from_index(index_2x2)


def probabilities_for(classes: List[int], size: int = 32) -> List[np.ndarray]:
    """One-hot probability tiles, one per class in ``classes``."""
    out = []
    for class_index in classes:
        patch = np.zeros((NUM_CLASSES, size, size), dtype=np.float32)
        patch[class_index] = 1.0
        out.append(patch)
    return out


def write_scene_dir(
    tmp_path: Path, frame: pd.DataFrame, fill: float = -12.0, scene_px: int = 64
) -> Path:
    """Write tile GeoTIFFs plus the index, the way run_pipeline does."""
    scene_dir = tmp_path / "synthetic"
    scene_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for _, row in frame.iterrows():
        path = scene_dir / f"{row['tile_id']}.tif"
        transform = Affine.from_gdal(
            row["gt_origin_x"], row["gt_pixel_width"], row["gt_row_rotation"],
            row["gt_origin_y"], row["gt_col_rotation"], row["gt_pixel_height"],
        )
        with rasterio.open(
            path, "w", driver="GTiff", height=int(row["height"]),
            width=int(row["width"]), count=2, dtype="float32",
            crs=CRS.from_epsg(4326), transform=transform, nodata=float("nan"),
        ) as dst:
            dst.write(np.full((2, int(row["height"]), int(row["width"])), fill,
                              dtype=np.float32))
        paths.append(str(path))

    frame = frame.copy()
    frame["path"] = paths
    frame.to_parquet(scene_dir / "tile_index.parquet", index=False)
    return scene_dir


# --------------------------------------------------------------------------- #
# reading the index
# --------------------------------------------------------------------------- #
def test_scene_grid_is_rebuilt_from_the_index(index_2x2: pd.DataFrame) -> None:
    grid = scene_grid_from_index(index_2x2)
    assert grid.shape == (64, 64)
    assert grid.transform == scene_transform()
    assert grid.crs == CRS.from_epsg(4326)


def test_scene_grid_survives_a_missing_corner_tile(index_2x2: pd.DataFrame) -> None:
    """Preprocessing drops mostly-land tiles, so tile (0,0) may not exist."""
    without_corner = index_2x2[
        ~((index_2x2["row_off"] == 0) & (index_2x2["col_off"] == 0))
    ]
    grid = scene_grid_from_index(without_corner)
    assert grid.shape == (64, 64)
    assert grid.transform.c == pytest.approx(ORIGIN_LON)
    assert grid.transform.f == pytest.approx(ORIGIN_LAT)


def test_load_tile_index_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InferenceError, match="no tile_index.parquet"):
        load_tile_index(tmp_path)


def test_tile_paths_prefers_the_recorded_path(tmp_path: Path, index_2x2) -> None:
    scene_dir = write_scene_dir(tmp_path, index_2x2)
    frame = load_tile_index(scene_dir)
    assert all(p.is_file() for p in tile_paths(frame, scene_dir))


def test_tile_paths_falls_back_to_tile_ids(index_2x2: pd.DataFrame) -> None:
    paths = tile_paths(index_2x2.drop(columns=[], errors="ignore"), Path("/scenes/x"))
    assert paths[0].name.endswith(".tif")
    assert paths[0].stem == index_2x2["tile_id"].iloc[0]


# --------------------------------------------------------------------------- #
# stitching - the core of this step
# --------------------------------------------------------------------------- #
def test_stitching_places_each_tile_in_the_right_quadrant(
    index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    """Four tiles, four different classes: each must land in its own quadrant."""
    order = list(zip(index_2x2["row_off"], index_2x2["col_off"]))
    classes = [1, 2, 3, 4]

    stitched, coverage = stitch_tiles(probabilities_for(classes), index_2x2, grid_2x2)
    mask = mask_from_probabilities(stitched, coverage)

    assert mask.shape == (64, 64)
    for (row_off, col_off), class_index in zip(order, classes):
        quadrant = mask[row_off : row_off + 32, col_off : col_off + 32]
        assert (quadrant == class_index).all(), f"quadrant at {row_off},{col_off}"


def test_stitching_covers_every_pixel_exactly_once_without_overlap(
    index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    _, coverage = stitch_tiles(probabilities_for([1, 1, 1, 1]), index_2x2, grid_2x2)
    assert (coverage == 1).all()


def test_uncovered_pixels_are_marked_nodata(
    index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    """A dropped tile leaves a hole, and the hole must not read as sea."""
    partial = index_2x2.iloc[:3]
    stitched, coverage = stitch_tiles(probabilities_for([1, 1, 1]), partial, grid_2x2)
    mask = mask_from_probabilities(stitched, coverage)

    missing = index_2x2.iloc[3]
    row_off, col_off = int(missing["row_off"]), int(missing["col_off"])
    assert (mask[row_off : row_off + 32, col_off : col_off + 32] == NODATA_CLASS).all()
    assert (mask[:32, :32] == 1).all()


def test_overlapping_tiles_are_averaged_not_overwritten() -> None:
    """The documented overlap rule: a soft vote, so both views count."""
    frame = make_index(tile_size=32, overlap=16, scene_px=48)
    grid = scene_grid_from_index(frame)
    assert len(frame) == 4  # offsets 0 and 16 on both axes

    # first tile votes class 1, the rest vote class 2
    stitched, coverage = stitch_tiles(
        probabilities_for([1, 2, 2, 2]), frame, grid
    )

    assert coverage.max() == 4  # the centre is seen by all four tiles
    centre = stitched[:, 24, 24]
    assert centre[1] == pytest.approx(0.25)
    assert centre[2] == pytest.approx(0.75)

    corner = stitched[:, 0, 0]  # only the first tile reaches here
    assert corner[1] == pytest.approx(1.0)


def test_averaging_beats_a_single_disagreeing_tile() -> None:
    """A slick clipped by one tile edge survives because the other tile saw it."""
    frame = make_index(tile_size=32, overlap=16, scene_px=48)
    grid = scene_grid_from_index(frame)

    confident_oil = np.zeros((NUM_CLASSES, 32, 32), dtype=np.float32)
    confident_oil[1] = 0.9
    confident_oil[0] = 0.1
    unsure_sea = np.zeros((NUM_CLASSES, 32, 32), dtype=np.float32)
    unsure_sea[0] = 0.55
    unsure_sea[1] = 0.45

    stitched, coverage = stitch_tiles(
        [confident_oil, unsure_sea, unsure_sea, unsure_sea], frame, grid
    )
    mask = mask_from_probabilities(stitched, coverage)
    assert mask[24, 24] == 1


def test_stitching_respects_the_valid_mask(
    index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    """Land/nodata pixels contribute nothing and end up uncovered."""
    valid = [np.ones((32, 32), dtype=bool) for _ in range(4)]
    valid[0][:16, :] = False

    stitched, coverage = stitch_tiles(
        probabilities_for([1, 1, 1, 1]), index_2x2, grid_2x2, valid_masks=valid
    )
    mask = mask_from_probabilities(stitched, coverage)
    assert (mask[:16, :32] == NODATA_CLASS).all()
    assert (mask[16:32, :32] == 1).all()


def test_stitching_rejects_a_prediction_count_mismatch(
    index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    with pytest.raises(InferenceError, match="for 4 tile"):
        stitch_tiles(probabilities_for([1, 2]), index_2x2, grid_2x2)


def test_stitching_handles_edge_tiles_smaller_than_the_tile_size() -> None:
    """The tiler pads offsets flush to the edge; stitching must not overrun."""
    frame = make_index(tile_size=32, overlap=0, scene_px=40)
    grid = scene_grid_from_index(frame)
    stitched, coverage = stitch_tiles(
        probabilities_for([1] * len(frame)), frame, grid
    )
    assert stitched.shape[1:] == (40, 40)
    assert (coverage > 0).all()


# --------------------------------------------------------------------------- #
# stitch_backscatter - the array the look-alike filter measures blobs against
# --------------------------------------------------------------------------- #
def write_backscatter_scene_dir(
    tmp_path: Path, frame: pd.DataFrame, fills: List[float]
) -> Path:
    """Like write_scene_dir, but each tile gets its own fill value.

    Stands in for the despeckled/geocoded/land-masked tiles run_pipeline
    writes: each tile's own dB level here plays the same role a real slick or
    real land NaN would.
    """
    scene_dir = tmp_path / "synthetic_backscatter"
    scene_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for (_, row), fill in zip(frame.iterrows(), fills):
        path = scene_dir / f"{row['tile_id']}.tif"
        transform = Affine.from_gdal(
            row["gt_origin_x"], row["gt_pixel_width"], row["gt_row_rotation"],
            row["gt_origin_y"], row["gt_col_rotation"], row["gt_pixel_height"],
        )
        with rasterio.open(
            path, "w", driver="GTiff", height=int(row["height"]),
            width=int(row["width"]), count=2, dtype="float32",
            crs=CRS.from_epsg(4326), transform=transform, nodata=float("nan"),
        ) as dst:
            dst.write(
                np.full((2, int(row["height"]), int(row["width"])), fill, dtype=np.float32)
            )
        paths.append(str(path))

    frame = frame.copy()
    frame["path"] = paths
    frame.to_parquet(scene_dir / "tile_index.parquet", index=False)
    return scene_dir


def test_stitch_backscatter_places_each_tiles_value_in_its_quadrant(
    tmp_path: Path, index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    fills = [-8.0, -14.0, -20.0, -26.0]
    scene_dir = write_backscatter_scene_dir(tmp_path, index_2x2, fills)
    frame = load_tile_index(scene_dir)

    backscatter = stitch_backscatter(scene_dir, frame=frame, grid=grid_2x2)

    assert backscatter.shape == (2, 64, 64)
    for (_, row), fill in zip(frame.iterrows(), fills):
        row_off, col_off = int(row["row_off"]), int(row["col_off"])
        quadrant = backscatter[:, row_off : row_off + 32, col_off : col_off + 32]
        assert quadrant == pytest.approx(fill)


def test_stitch_backscatter_shares_the_grid_infer_scene_produced(
    tmp_path: Path, index_2x2: pd.DataFrame
) -> None:
    """The whole point of the fix: same grid the mask was stitched on."""
    scene_dir = write_scene_dir(tmp_path, index_2x2, fill=-15.0)
    result = infer_scene(scene_dir, model=FixedClassModel([1, 1, 1, 1]),
                         scene_id="synthetic", save=False)

    backscatter = stitch_backscatter(scene_dir, grid=result.grid)

    assert backscatter.shape[1:] == result.mask.shape
    assert backscatter[0, 10, 10] == pytest.approx(-15.0)


def test_stitch_backscatter_marks_uncovered_pixels_as_nan_not_zero(
    tmp_path: Path, index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    """A gap left by a dropped tile must read as missing, not as calm water."""
    partial = index_2x2.iloc[:3]
    scene_dir = write_backscatter_scene_dir(tmp_path, partial, [-8.0, -8.0, -8.0])
    frame = load_tile_index(scene_dir)

    backscatter = stitch_backscatter(scene_dir, frame=frame, grid=grid_2x2)

    missing = index_2x2.iloc[3]
    row_off, col_off = int(missing["row_off"]), int(missing["col_off"])
    assert np.isnan(backscatter[:, row_off : row_off + 32, col_off : col_off + 32]).all()
    assert not np.isnan(backscatter[:, :32, :32]).any()


def test_stitch_backscatter_keeps_land_nodata_as_nan_not_zero(
    tmp_path: Path, index_2x2: pd.DataFrame, grid_2x2: SceneGrid
) -> None:
    """Land inside a tile (not just a missing tile) must not resolve to 0 dB.

    A fabricated 0 dB fill where land actually is would let the look-alike
    filter's contrast ring pick up land backscatter as if it were sea.
    """
    fills = [-8.0, -8.0, -8.0, -8.0]
    scene_dir = write_backscatter_scene_dir(tmp_path, index_2x2, fills)
    frame = load_tile_index(scene_dir)

    corner = frame.iloc[0]
    path = Path(corner["path"])
    with rasterio.open(path) as dataset:
        profile = dataset.profile
    data = np.full((2, int(corner["height"]), int(corner["width"])), -8.0, dtype=np.float32)
    data[:, :16, :] = np.nan  # half the tile is land
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    backscatter = stitch_backscatter(scene_dir, frame=load_tile_index(scene_dir), grid=grid_2x2)

    row_off, col_off = int(corner["row_off"]), int(corner["col_off"])
    land_region = backscatter[:, row_off : row_off + 16, col_off : col_off + 32]
    assert np.isnan(land_region).all()
    assert not (land_region == 0).any()


# --------------------------------------------------------------------------- #
# model input preparation
# --------------------------------------------------------------------------- #
def test_db_to_model_input_maps_the_window_and_expands_channels() -> None:
    tile_data = np.array([[[-35.0, -17.5, 0.0]]], dtype=np.float32)
    prepared = db_to_model_input(tile_data, in_channels=3, db_min=-35.0, db_max=0.0,
                                normalize=False)
    assert prepared.shape == (3, 1, 3)
    assert prepared[0].ravel() == pytest.approx([0.0, 0.5, 1.0])


def test_db_to_model_input_clips_outside_the_window() -> None:
    tile_data = np.array([[[-80.0, 20.0]]], dtype=np.float32)
    prepared = db_to_model_input(tile_data, normalize=False)
    assert prepared.min() == 0.0 and prepared.max() == 1.0


def test_db_to_model_input_replaces_nan_and_stays_finite() -> None:
    tile_data = np.full((1, 4, 4), np.nan, dtype=np.float32)
    assert np.isfinite(db_to_model_input(tile_data)).all()


def test_db_to_model_input_trims_extra_bands() -> None:
    tile_data = np.zeros((2, 4, 4), dtype=np.float32)
    assert db_to_model_input(tile_data, in_channels=3).shape[0] == 3
    assert db_to_model_input(tile_data, in_channels=1).shape[0] == 1


def test_tile_valid_mask_needs_every_band() -> None:
    tile_data = np.zeros((2, 4, 4), dtype=np.float32)
    tile_data[1, 0, 0] = np.nan
    assert not tile_valid_mask(tile_data)[0, 0]
    assert tile_valid_mask(tile_data)[1, 1]


def test_predict_tiles_batches_and_returns_probabilities() -> None:
    tiles = [np.zeros((3, 8, 8), dtype=np.float32) for _ in range(5)]
    model = ConstantProbabilityModel(torch.tensor([0.0, 5.0, 0.0, 0.0, 0.0]))

    probabilities = predict_tiles(model, tiles, batch_size=2)

    assert probabilities.shape == (5, NUM_CLASSES, 8, 8)
    assert probabilities.sum(axis=1) == pytest.approx(np.ones((5, 8, 8)), abs=1e-5)
    assert probabilities[:, 1].min() > 0.9


def test_predict_tiles_rejects_an_empty_list() -> None:
    with pytest.raises(InferenceError, match="no tiles"):
        predict_tiles(ConstantProbabilityModel(torch.zeros(5)), [])


# --------------------------------------------------------------------------- #
# whole-scene inference
# --------------------------------------------------------------------------- #
def test_infer_scene_writes_a_georeferenced_mask(
    tmp_path: Path, index_2x2: pd.DataFrame
) -> None:
    scene_dir = write_scene_dir(tmp_path, index_2x2)
    model = FixedClassModel([1, 2, 3, 4])

    result = infer_scene(scene_dir, model=model, scene_id="synthetic")

    assert result.tiles == 4
    assert result.mask.shape == (64, 64)
    assert result.mask_path is not None and result.mask_path.is_file()

    with rasterio.open(result.mask_path) as dataset:
        assert dataset.crs == CRS.from_epsg(4326)
        assert dataset.transform == scene_transform()
        assert dataset.nodata == NODATA_CLASS
        assert dataset.tags()["scene_id"] == "synthetic"
        assert np.array_equal(dataset.read(1), result.mask)


def test_infer_scene_reports_class_fractions(
    tmp_path: Path, index_2x2: pd.DataFrame
) -> None:
    scene_dir = write_scene_dir(tmp_path, index_2x2)
    result = infer_scene(scene_dir, model=FixedClassModel([1, 1, 0, 0]),
                         scene_id="synthetic", save=False)
    assert result.class_fraction(1) == pytest.approx(0.5)
    assert result.class_fraction(0) == pytest.approx(0.5)


def test_infer_scene_confidence_is_the_winning_probability(
    tmp_path: Path, index_2x2: pd.DataFrame
) -> None:
    scene_dir = write_scene_dir(tmp_path, index_2x2)
    result = infer_scene(scene_dir, model=FixedClassModel([1, 1, 1, 1]),
                         scene_id="synthetic", save=False)
    assert result.confidence.shape == (64, 64)
    assert result.confidence.max() <= 1.0
    assert result.confidence[covered_pixel := (0, 0)] > 0.5


def test_infer_scene_marks_nodata_tiles_as_uncovered(
    tmp_path: Path, index_2x2: pd.DataFrame
) -> None:
    """A tile that is entirely land contributes nothing to the stitched mask."""
    scene_dir = write_scene_dir(tmp_path, index_2x2)
    frame = load_tile_index(scene_dir)
    blank = Path(frame["path"].iloc[0])
    with rasterio.open(blank) as dataset:
        profile = dataset.profile
    with rasterio.open(blank, "w", **profile) as dst:
        dst.write(np.full((2, 32, 32), np.nan, dtype=np.float32))

    result = infer_scene(scene_dir, model=FixedClassModel([1, 1, 1, 1]),
                         scene_id="synthetic", save=False)
    row = int(frame["row_off"].iloc[0])
    col = int(frame["col_off"].iloc[0])
    assert (result.mask[row : row + 32, col : col + 32] == NODATA_CLASS).all()


def test_read_tile_returns_float32_bands(tmp_path: Path, index_2x2) -> None:
    scene_dir = write_scene_dir(tmp_path, index_2x2, fill=-15.0)
    tile_data = read_tile(Path(load_tile_index(scene_dir)["path"].iloc[0]))
    assert tile_data.shape == (2, 32, 32)
    assert tile_data.dtype == np.float32
    assert tile_data[0, 0, 0] == pytest.approx(-15.0)


def test_load_model_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    """Nothing has trained a checkpoint yet, so this is the expected path."""
    with pytest.raises(InferenceError, match="no checkpoint at"):
        load_model(tmp_path / "best.pt")


def test_save_mask_round_trips(tmp_path: Path, index_2x2, grid_2x2) -> None:
    from src.detection.infer import InferenceResult

    mask = np.full((64, 64), 3, dtype=np.uint8)
    result = InferenceResult(
        scene_id="s", mask=mask,
        probabilities=np.zeros((NUM_CLASSES, 64, 64), dtype=np.float32),
        grid=grid_2x2, class_names=list("abcde"), tiles=4,
    )
    path = save_mask(tmp_path / "mask.tif", result)
    with rasterio.open(path) as dataset:
        assert np.array_equal(dataset.read(1), mask)
        assert dataset.transform == scene_transform()
