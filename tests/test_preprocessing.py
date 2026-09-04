"""Preprocessing tests.

Everything runs on synthetic arrays and synthetic GeoTIFFs written in a tmp
dir - no real Sentinel-1 product, no network. Land masking is checked against
both a hand-made land polygon (so the expected pixels are known exactly) and
the clipped Natural Earth layer vendored in ``configs/coastline/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from src.config import get_settings
from src.ingestion.types import Scene, SceneSourceKind
from src.preprocessing.pipeline import (
    PreprocessingError,
    RasterImage,
    calibrate,
    dn_to_sigma0,
    geocode,
    land_mask,
    load_coastline,
    looks_like_db,
    mask_land,
    run_pipeline,
    speckle_filter,
    tile,
    tile_index_to_frame,
    to_db,
    to_linear,
    transform_from_index_row,
)

# Raster grid: 0.001 deg pixels from (70.0 E, 22.0 N) - a 256 px scene is ~28 km.
ORIGIN_LON, ORIGIN_LAT = 70.0, 22.0
PIXEL_DEG = 0.001
SCENE_TIME = "2023-05-14T00:33:32+00:00"


def make_transform(size: int = 256) -> Affine:
    return from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG)


def speckled_field(
    shape: Tuple[int, int] = (128, 128), mean_db: float = -12.0, seed: int = 0
) -> np.ndarray:
    """A synthetic dB backscatter field with multiplicative speckle."""
    rng = np.random.default_rng(seed)
    linear = to_linear(np.full(shape, mean_db))
    speckle = rng.gamma(shape=4.4, scale=1 / 4.4, size=shape)
    return to_db(linear * speckle).astype(np.float32)


def write_scene_tif(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    transform: Affine | None = None,
    nodata: float | None = None,
) -> Path:
    """Write a synthetic multi-band scene to disk."""
    data = np.atleast_3d(data.T).T if data.ndim == 2 else data
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs=crs,
        transform=transform if transform is not None else make_transform(),
        nodata=nodata,
    ) as dst:
        dst.write(data.astype("float32"))
    return path


def make_scene(path: Path, scene_id: str = "synthetic_scene") -> Scene:
    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
    return Scene(
        scene_id=scene_id,
        source=SceneSourceKind.LOCAL,
        acquisition_time=SCENE_TIME,
        footprint=box(*bounds),
        path=path,
        polarisations=["VV", "VH"],
    )


@pytest.fixture
def db_scene(tmp_path: Path) -> Scene:
    """A two-band scene already stored in dB, like most public SAR datasets."""
    data = np.stack([speckled_field((128, 128), -12.0, 1), speckled_field((128, 128), -18.0, 2)])
    return make_scene(write_scene_tif(tmp_path / "scene_db.tif", data))


@pytest.fixture
def land_geojson(tmp_path: Path) -> Path:
    """Land covering the northern half of the synthetic raster."""
    north_half = box(
        ORIGIN_LON - 0.01,
        ORIGIN_LAT - 128 * PIXEL_DEG,  # halfway down a 256 px raster
        ORIGIN_LON + 0.5,
        ORIGIN_LAT + 0.01,
    )
    path = tmp_path / "land.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": mapping(north_half), "properties": {}}
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# 1. calibration
# --------------------------------------------------------------------------- #
def test_db_linear_roundtrip() -> None:
    values = np.array([-25.0, -12.0, 0.0, 3.0], dtype=np.float32)
    assert to_db(to_linear(values)) == pytest.approx(values, abs=1e-4)


def test_dn_to_sigma0_squares_the_counts() -> None:
    dn = np.array([[1.0, 2.0, 10.0]])
    assert dn_to_sigma0(dn).ravel() == pytest.approx([1.0, 4.0, 100.0])
    # a +20 dB calibration constant divides amplitude by 10, power by 100
    assert dn_to_sigma0(dn, 20.0).ravel() == pytest.approx([0.01, 0.04, 1.0])


def test_looks_like_db_separates_db_from_counts() -> None:
    assert looks_like_db(np.array([-30.0, -5.0, 2.0]))
    assert not looks_like_db(np.array([0.0, 1200.0, 8000.0]))
    assert not looks_like_db(np.array([np.nan, np.nan]))


def test_calibrate_preserves_shape_and_georeferencing(db_scene: Scene) -> None:
    image = calibrate(db_scene)
    assert image.count == 2
    assert (image.height, image.width) == (128, 128)
    assert image.data.dtype == np.float32
    assert image.crs == CRS.from_epsg(4326)
    assert image.transform == make_transform()
    assert image.scene_id == db_scene.scene_id
    assert image.band_names == ["VV", "VH"]


def test_calibrate_passes_through_data_already_in_db(db_scene: Scene) -> None:
    with rasterio.open(db_scene.path) as dataset:
        original = dataset.read().astype(np.float32)
    assert calibrate(db_scene).data == pytest.approx(original, nan_ok=True)


def test_calibrate_converts_raw_counts(tmp_path: Path) -> None:
    counts = np.full((1, 32, 32), 100.0, dtype=np.float32)
    scene = make_scene(write_scene_tif(tmp_path / "scene_dn.tif", counts), "dn_scene")

    image = calibrate(scene)

    # sigma0 = DN^2 = 10000 -> 40 dB
    assert image.data == pytest.approx(np.full((1, 32, 32), 40.0), abs=1e-4)


def test_calibrate_honours_the_source_nodata(tmp_path: Path) -> None:
    data = np.full((1, 16, 16), -12.0, dtype=np.float32)
    data[0, 0, 0] = -9999.0
    scene = make_scene(
        write_scene_tif(tmp_path / "nodata.tif", data, nodata=-9999.0), "nodata_scene"
    )
    image = calibrate(scene)
    assert math.isnan(float(image.data[0, 0, 0]))
    assert np.isfinite(image.data).sum() == 255


def test_calibrate_rejects_a_scene_with_no_pixels(db_scene: Scene) -> None:
    Path(db_scene.path).unlink()
    with pytest.raises(PreprocessingError, match="missing from disk"):
        calibrate(db_scene)


# --------------------------------------------------------------------------- #
# 2. speckle filtering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["lee", "refined_lee"])
def test_speckle_filter_preserves_shape_and_dtype(method: str) -> None:
    field = speckled_field((64, 64))
    out = speckle_filter(field, method=method, size=5)
    assert out.shape == field.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.parametrize("method", ["lee", "refined_lee"])
def test_speckle_filter_reduces_variance_on_flat_water(method: str) -> None:
    field = speckled_field((96, 96), mean_db=-14.0, seed=3)
    out = speckle_filter(field, method=method, size=7)
    assert out.std() < field.std()
    # the average level must survive; despeckling is not supposed to bias it
    assert out.mean() == pytest.approx(field.mean(), abs=1.5)


@pytest.mark.parametrize("method", ["lee", "refined_lee"])
def test_speckle_filter_keeps_a_strong_edge(method: str) -> None:
    """A dark slick against bright water must stay dark after filtering."""
    field = np.full((64, 64), -8.0, dtype=np.float32)
    field[:, 32:] = -22.0
    out = np.asarray(speckle_filter(field, method=method, size=5))
    assert out[:, :28].mean() > out[:, 36:].mean() + 10.0


def test_speckle_filter_handles_3d_and_raster_image() -> None:
    stack = np.stack([speckled_field((32, 32), -12.0, 1), speckled_field((32, 32), -20.0, 2)])
    assert speckle_filter(stack, method="lee", size=5).shape == stack.shape

    image = RasterImage(data=stack, transform=make_transform(), crs=CRS.from_epsg(4326))
    filtered = speckle_filter(image, method="lee", size=5)
    assert isinstance(filtered, RasterImage)
    assert filtered.shape == image.shape
    assert filtered.transform == image.transform


def test_speckle_filter_leaves_nodata_holes_alone() -> None:
    field = speckled_field((32, 32))
    field[4:8, 4:8] = np.nan
    out = np.asarray(speckle_filter(field, method="lee", size=5))
    assert np.isnan(out[4:8, 4:8]).all()
    assert np.isfinite(out[16:, 16:]).all()


def test_speckle_filter_none_is_a_passthrough() -> None:
    field = speckled_field((16, 16))
    assert speckle_filter(field, method="none") is field


def test_speckle_filter_rejects_an_even_window() -> None:
    with pytest.raises(PreprocessingError, match="odd"):
        speckle_filter(speckled_field((16, 16)), method="lee", size=4)


# --------------------------------------------------------------------------- #
# 3. geocoding
# --------------------------------------------------------------------------- #
def test_geocode_leaves_a_wgs84_image_untouched(db_scene: Scene) -> None:
    image = calibrate(db_scene)
    out = geocode(image, db_scene)
    assert out.crs == CRS.from_epsg(4326)
    assert out.transform == image.transform
    assert out.data.shape == image.data.shape


def test_geocode_warps_a_projected_image(tmp_path: Path) -> None:
    data = np.full((1, 64, 64), -12.0, dtype=np.float32)
    scene = make_scene(
        write_scene_tif(
            tmp_path / "utm.tif",
            data,
            crs="EPSG:32643",
            transform=from_origin(200_000.0, 2_400_000.0, 100.0, 100.0),
        ),
        "utm_scene",
    )

    out = geocode(calibrate(scene), scene)

    assert out.crs == CRS.from_epsg(4326)
    west, south, east, north = out.bounds
    assert 60 < west < 75 and 15 < south < 30
    assert west < east and south < north
    assert np.isfinite(out.data).any()


def test_geocode_uses_gcps_when_there_is_no_crs() -> None:
    from rasterio.control import GroundControlPoint

    gcps = [
        GroundControlPoint(row=0, col=0, x=70.0, y=22.0),
        GroundControlPoint(row=0, col=64, x=70.064, y=22.0),
        GroundControlPoint(row=64, col=0, x=70.0, y=21.936),
        GroundControlPoint(row=64, col=64, x=70.064, y=21.936),
    ]
    image = RasterImage(
        data=np.full((1, 64, 64), -12.0, dtype=np.float32),
        transform=Affine.identity(),
        crs=None,
        gcps=gcps,
        gcp_crs=CRS.from_epsg(4326),
    )

    out = geocode(image, None)

    assert out.crs == CRS.from_epsg(4326)
    west, south, east, north = out.bounds
    assert west == pytest.approx(70.0, abs=0.01)
    assert north == pytest.approx(22.0, abs=0.01)


def test_geocode_without_crs_or_gcps_raises() -> None:
    image = RasterImage(
        data=np.zeros((1, 8, 8), dtype=np.float32),
        transform=Affine.identity(),
        crs=None,
    )
    with pytest.raises(PreprocessingError, match="neither a CRS nor GCPs"):
        geocode(image, None)


# --------------------------------------------------------------------------- #
# 4. land masking
# --------------------------------------------------------------------------- #
def test_land_mask_marks_exactly_the_land_half(land_geojson: Path) -> None:
    mask = land_mask((256, 256), make_transform(), crs=CRS.from_epsg(4326),
                     coastline=land_geojson, buffer_m=0.0)
    assert mask.shape == (256, 256)
    assert mask[:128].all(), "northern half is land and must be masked"
    assert not mask[129:].any(), "southern half is water and must survive"


def test_mask_land_nulls_land_pixels_only(land_geojson: Path) -> None:
    data = np.full((1, 256, 256), -12.0, dtype=np.float32)
    image = RasterImage(data=data, transform=make_transform(), crs=CRS.from_epsg(4326))

    out = mask_land(image, coastline=land_geojson, buffer_m=0.0)

    assert np.isnan(out.data[0, :128, :]).all()
    assert np.isfinite(out.data[0, 129:, :]).all()
    assert out.data[0, 200, 200] == pytest.approx(-12.0)


def test_mask_land_uses_nan_not_zero(land_geojson: Path) -> None:
    """Zeroed land would look exactly like an oil slick to the detector."""
    data = np.full((1, 256, 256), -12.0, dtype=np.float32)
    out = mask_land(
        RasterImage(data=data, transform=make_transform(), crs=CRS.from_epsg(4326)),
        coastline=land_geojson,
        buffer_m=0.0,
    )
    assert not (out.data == 0).any()


def test_mask_land_buffer_widens_the_masked_area(land_geojson: Path) -> None:
    args = ((256, 256), make_transform())
    kwargs = {"crs": CRS.from_epsg(4326), "coastline": land_geojson}
    tight = land_mask(*args, buffer_m=0.0, **kwargs)
    wide = land_mask(*args, buffer_m=2000.0, **kwargs)
    assert wide.sum() > tight.sum()


def test_mask_land_accepts_a_bare_array_with_a_transform(land_geojson: Path) -> None:
    data = np.full((256, 256), -12.0, dtype=np.float32)
    out = mask_land(data, make_transform(), crs=CRS.from_epsg(4326),
                    coastline=land_geojson, buffer_m=0.0)
    assert out.shape == data.shape
    assert np.isnan(out[:128]).all()


def test_mask_land_needs_a_transform_for_a_bare_array() -> None:
    with pytest.raises(PreprocessingError, match="needs a transform"):
        mask_land(np.zeros((8, 8), dtype=np.float32))


def test_mask_land_leaves_open_ocean_alone() -> None:
    """A raster far offshore keeps every pixel."""
    offshore = from_origin(60.0, 10.0, PIXEL_DEG, PIXEL_DEG)
    data = np.full((1, 64, 64), -12.0, dtype=np.float32)
    out = mask_land(
        RasterImage(data=data, transform=offshore, crs=CRS.from_epsg(4326)),
        coastline=box(0.0, 0.0, 1.0, 1.0),  # land nowhere near the raster
    )
    assert np.isfinite(out.data).all()


def test_vendored_coastline_covers_the_demo_aoi() -> None:
    """The clipped Natural Earth layer shipped in the repo is usable as-is."""
    coastline = load_coastline()
    assert len(coastline) > 0
    west, south, east, north = coastline.total_bounds
    settings = get_settings()
    aoi_west, aoi_south, aoi_east, aoi_north = 68.0, 20.0, 72.5, 23.5
    assert west <= aoi_west and east >= aoi_east
    assert south <= aoi_south and north >= aoi_north
    assert settings.preprocessing.coastline_path.suffix == ".geojson"


def test_land_mask_against_the_vendored_coastline() -> None:
    """A raster straddling the Gujarat coast is part land, part water."""
    transform = from_origin(69.5, 22.6, 0.002, 0.002)
    mask = land_mask((512, 512), transform, crs=CRS.from_epsg(4326))
    assert 0.0 < mask.mean() < 1.0


def test_load_coastline_reports_a_missing_layer(tmp_path: Path) -> None:
    with pytest.raises(PreprocessingError, match="coastline layer not found"):
        load_coastline(path=tmp_path / "nope.geojson")


# --------------------------------------------------------------------------- #
# 5. tiling
# --------------------------------------------------------------------------- #
def test_tile_splits_evenly_without_overlap() -> None:
    data = np.zeros((1, 256, 256), dtype=np.float32)
    image = RasterImage(data=data, transform=make_transform(), crs=CRS.from_epsg(4326),
                        scene_id="s1")

    tiles, index = tile(image, tile_size=128, overlap=0)

    assert len(tiles) == len(index) == 4
    assert all(t.shape == (1, 128, 128) for t in tiles)
    assert {(e["row_off"], e["col_off"]) for e in index} == {
        (0, 0), (0, 128), (128, 0), (128, 128)
    }


def test_tile_offsets_follow_the_overlap() -> None:
    image = RasterImage(
        data=np.zeros((1, 256, 256), dtype=np.float32), transform=make_transform()
    )
    _, index = tile(image, tile_size=128, overlap=64)
    assert sorted({e["col_off"] for e in index}) == [0, 64, 128]


def test_tile_pulls_the_last_tile_flush_with_the_edge() -> None:
    """Every tile is full size, so nothing ragged reaches the model."""
    image = RasterImage(
        data=np.zeros((1, 300, 300), dtype=np.float32), transform=make_transform()
    )
    tiles, index = tile(image, tile_size=128, overlap=0)
    assert all(t.shape == (1, 128, 128) for t in tiles)
    assert max(e["row_off"] for e in index) == 300 - 128


def test_tile_index_round_trips_to_the_right_pixels() -> None:
    """The index must locate a tile's pixels back in the scene exactly."""
    rng = np.random.default_rng(7)
    data = rng.normal(-12.0, 2.0, size=(1, 256, 256)).astype(np.float32)
    image = RasterImage(data=data, transform=make_transform(), crs=CRS.from_epsg(4326),
                        scene_id="scene_a")

    tiles, index = tile(image, tile_size=64, overlap=16)

    for patch, entry in zip(tiles, index):
        row, col = entry["row_off"], entry["col_off"]
        expected = data[:, row : row + entry["height"], col : col + entry["width"]]
        assert patch == pytest.approx(expected)


def test_tile_index_round_trips_to_the_right_map_coordinates() -> None:
    """A pixel inside a tile lands on the same lon/lat as it does in the scene."""
    transform = make_transform()
    image = RasterImage(
        data=np.zeros((1, 256, 256), dtype=np.float32),
        transform=transform,
        crs=CRS.from_epsg(4326),
        scene_id="scene_a",
    )
    _, index = tile(image, tile_size=64, overlap=16)
    frame = tile_index_to_frame(index)

    for _, row in frame.iterrows():
        tile_transform = transform_from_index_row(row)
        # tile pixel (10, 5) must be scene pixel (row_off + 10, col_off + 5)
        assert tile_transform @ (5, 10) == pytest.approx(
            transform @ (row["col_off"] + 5, row["row_off"] + 10)
        )


def test_tile_index_records_scene_and_tile_identity() -> None:
    image = RasterImage(
        data=np.zeros((1, 128, 128), dtype=np.float32),
        transform=make_transform(),
        crs=CRS.from_epsg(4326),
        scene_id="scene_b",
    )
    _, index = tile(image, tile_size=64, overlap=0)
    entry = index[0]
    assert entry["scene_id"] == "scene_b"
    assert entry["tile_id"] == "scene_b_r00000_c00000"
    assert entry["crs"] == "EPSG:4326"
    assert len({e["tile_id"] for e in index}) == len(index)


def test_tile_drops_tiles_below_the_valid_fraction() -> None:
    data = np.full((1, 128, 128), np.nan, dtype=np.float32)
    data[:, :64, :64] = -12.0  # only the top-left quarter carries data
    image = RasterImage(data=data, transform=make_transform(), scene_id="s")

    kept, index = tile(image, tile_size=64, overlap=0, min_valid_fraction=0.5)

    assert len(kept) == 1
    assert (index[0]["row_off"], index[0]["col_off"]) == (0, 0)
    assert index[0]["valid_fraction"] == pytest.approx(1.0)


def test_tile_rejects_overlap_larger_than_the_tile() -> None:
    image = RasterImage(
        data=np.zeros((1, 64, 64), dtype=np.float32), transform=make_transform()
    )
    with pytest.raises(PreprocessingError, match="overlap must be smaller"):
        tile(image, tile_size=32, overlap=32)


def test_tile_handles_an_image_smaller_than_one_tile() -> None:
    image = RasterImage(
        data=np.zeros((1, 40, 40), dtype=np.float32), transform=make_transform()
    )
    tiles, index = tile(image, tile_size=64, overlap=0)
    assert len(tiles) == 1 and tiles[0].shape == (1, 40, 40)
    assert index[0]["height"] == 40


def test_tile_index_frame_flattens_for_parquet(tmp_path: Path) -> None:
    image = RasterImage(
        data=np.zeros((1, 128, 128), dtype=np.float32),
        transform=make_transform(),
        crs=CRS.from_epsg(4326),
        scene_id="s",
    )
    _, index = tile(image, tile_size=64, overlap=0)
    frame = tile_index_to_frame(index)

    assert {"tile_id", "scene_id", "row_off", "col_off", "west", "north",
            "gt_origin_x", "gt_pixel_width"} <= set(frame.columns)

    path = tmp_path / "index.parquet"
    frame.to_parquet(path, index=False)
    assert pd.read_parquet(path).equals(frame)


# --------------------------------------------------------------------------- #
# 6. the whole chain
# --------------------------------------------------------------------------- #
def test_run_pipeline_writes_tiles_and_index(tmp_path: Path, db_scene: Scene) -> None:
    out_dir = tmp_path / "processed" / db_scene.scene_id

    result = run_pipeline(
        db_scene,
        output_dir=out_dir,
        tile_size=64,
        overlap=16,
        coastline=box(0.0, 0.0, 1.0, 1.0),  # land far away, nothing gets masked
    )

    assert result.scene_id == db_scene.scene_id
    assert result.tile_index_path == out_dir / "tile_index.parquet"
    assert result.tile_index_path.is_file()
    assert len(result.tiles) == len(result.index) > 0
    assert all(path.is_file() for path in result.tiles)


def test_run_pipeline_tiles_stay_georeferenced(tmp_path: Path, db_scene: Scene) -> None:
    result = run_pipeline(
        db_scene,
        output_dir=tmp_path / "out",
        tile_size=64,
        overlap=0,
        coastline=box(0.0, 0.0, 1.0, 1.0),
    )

    frame = pd.read_parquet(result.tile_index_path)
    for path in result.tiles:
        entry = frame[frame["tile_id"] == path.stem].iloc[0]
        with rasterio.open(path) as dataset:
            assert dataset.crs == CRS.from_epsg(4326)
            assert dataset.count == 2
            assert dataset.transform == transform_from_index_row(entry)
            assert dataset.tags()["scene_id"] == db_scene.scene_id
            assert dataset.descriptions == ("VV", "VH")


def test_run_pipeline_index_covers_the_scene(tmp_path: Path, db_scene: Scene) -> None:
    result = run_pipeline(
        db_scene,
        output_dir=tmp_path / "out",
        tile_size=64,
        overlap=0,
        coastline=box(0.0, 0.0, 1.0, 1.0),
    )
    frame = result.index
    scene_bounds = result.image.bounds
    assert frame["west"].min() == pytest.approx(scene_bounds[0])
    assert frame["north"].max() == pytest.approx(scene_bounds[3])
    assert (frame["scene_id"] == db_scene.scene_id).all()


def test_run_pipeline_masks_land_before_tiling(
    tmp_path: Path, db_scene: Scene, land_geojson: Path
) -> None:
    """Land pixels must already be nulled in the tiles handed downstream."""
    result = run_pipeline(
        db_scene,
        output_dir=tmp_path / "out",
        tile_size=64,
        overlap=0,
        coastline=land_geojson,
    )
    # the synthetic land polygon covers the whole 128 px scene from the north
    assert np.isnan(result.image.data).any()
    for path in result.tiles:
        with rasterio.open(path) as dataset:
            assert math.isnan(dataset.nodata)


def test_run_pipeline_can_skip_writing_tiles(tmp_path: Path, db_scene: Scene) -> None:
    result = run_pipeline(
        db_scene,
        output_dir=tmp_path / "out",
        tile_size=64,
        overlap=0,
        coastline=box(0.0, 0.0, 1.0, 1.0),
        write_tiles=False,
    )
    assert result.tiles == []
    assert result.tile_index_path.is_file()
    assert len(result.index) > 0
