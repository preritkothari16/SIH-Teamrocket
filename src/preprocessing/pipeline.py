"""SAR preprocessing: raw scene -> analysis-ready georeferenced tiles.

The five stages are separate, individually testable functions:

1. :func:`calibrate`      - DN -> sigma0 backscatter, expressed in dB
2. :func:`speckle_filter` - Lee or refined Lee despeckling
3. :func:`geocode`        - guarantee every pixel maps to lon/lat
4. :func:`mask_land`      - blank out land and the noisy coastal strip
5. :func:`tile`           - cut georeferenced tiles plus a tile index

:func:`run_pipeline` chains them, writes the tiles to
``data/processed/<scene_id>/`` and the index to ``tile_index.parquet``. The
index is what lets detections found in tile pixel space be stitched back to
scene and map coordinates later.

Stage order is deliberate: despeckling happens *before* geocoding, while the
pixels are still on the radar grid and their speckle is still uncorrelated.
Resampling first would smear the statistics the Lee filter depends on.

Example::

    from src.ingestion import LocalSceneSource
    from src.preprocessing.pipeline import run_pipeline

    scene = LocalSceneSource().scenes()[0]
    result = run_pipeline(scene)
    print(result.tile_index_path, len(result.tiles))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_gcps
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.ndimage import correlate, uniform_filter
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings
from src.ingestion.types import Scene

logger = logging.getLogger(__name__)

METRES_PER_DEGREE = 111_320.0  # good enough to widen a coastline buffer
EPS = 1e-10

# A value range that only makes sense for data already in dB.
_DB_MIN, _DB_MAX = -80.0, 60.0

ImageLike = Union[np.ndarray, "RasterImage"]


class PreprocessingError(RuntimeError):
    """A preprocessing stage could not complete."""


# --------------------------------------------------------------------------- #
# carrier type
# --------------------------------------------------------------------------- #
@dataclass
class RasterImage:
    """Pixels plus the georeferencing needed to keep them locatable.

    ``data`` is always 3D, ``(bands, rows, cols)``, float32, with invalid
    pixels set to ``nodata`` (NaN by default).
    """

    data: np.ndarray
    transform: Affine
    crs: Optional[CRS] = None
    nodata: float = float("nan")
    band_names: List[str] = field(default_factory=list)
    scene_id: Optional[str] = None
    gcps: Optional[Sequence[Any]] = None
    gcp_crs: Optional[CRS] = None

    def __post_init__(self) -> None:
        self.data = _as_3d(self.data)
        if not self.band_names:
            self.band_names = [f"band_{i + 1}" for i in range(self.count)]

    @property
    def count(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(west, south, east, north) in the image CRS."""
        west, north = self.transform @ (0, 0)
        east, south = self.transform @ (self.width, self.height)
        return (min(west, east), min(south, north), max(west, east), max(south, north))

    @property
    def valid_mask(self) -> np.ndarray:
        """True where a pixel carries real data, per band."""
        if math.isnan(self.nodata):
            return np.isfinite(self.data)
        return np.isfinite(self.data) & (self.data != self.nodata)

    def profile(self, **overrides: Any) -> Dict[str, Any]:
        """A rasterio creation profile for writing this image out."""
        prof: Dict[str, Any] = {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": self.count,
            "dtype": "float32",
            "crs": self.crs,
            "transform": self.transform,
            "nodata": self.nodata,
            "compress": "deflate",
            "tiled": True,
        }
        prof.update(overrides)
        return prof


def _as_3d(data: np.ndarray) -> np.ndarray:
    """Normalise to (bands, rows, cols) float32."""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    if array.ndim == 3:
        return array
    raise PreprocessingError(f"expected a 2D or 3D array, got shape {array.shape}")


def _data_of(image: ImageLike) -> np.ndarray:
    return image.data if isinstance(image, RasterImage) else _as_3d(image)


def _like(image: ImageLike, data: np.ndarray) -> ImageLike:
    """Return ``data`` wrapped the same way ``image`` arrived.

    Lets each stage accept a bare array (convenient in tests) or a
    :class:`RasterImage` (what the pipeline threads through) and hand back the
    same kind, so the stages compose either way.
    """
    if isinstance(image, RasterImage):
        return replace(image, data=data)
    original = np.asarray(image)
    return data[0] if original.ndim == 2 else data


# --------------------------------------------------------------------------- #
# 1. calibration
# --------------------------------------------------------------------------- #
def looks_like_db(data: np.ndarray) -> bool:
    """Guess whether an array already holds dB rather than raw DN.

    dB backscatter is signed and small; DN counts are non-negative and usually
    run into the thousands.
    """
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    low, high = float(finite.min()), float(finite.max())
    return low < 0.0 and _DB_MIN <= low and high <= _DB_MAX


def dn_to_sigma0(
    dn: np.ndarray, calibration_constant_db: float = 0.0
) -> np.ndarray:
    """Digital numbers to linear sigma0.

    Sentinel-1 GRD calibration is ``sigma0 = DN^2 / A^2`` with ``A`` read from
    the product's sigmaNought LUT. Products that arrive without their annotation
    (GeoTIFF exports, most public training sets) have no LUT, so a single
    constant stands in for it - configurable, and 0 dB by default, which leaves
    the relative radiometry intact even when the absolute level is unknown.
    """
    amplitude = 10.0 ** (calibration_constant_db / 20.0)
    with np.errstate(invalid="ignore"):
        return np.square(dn.astype(np.float64)) / (amplitude**2)


def to_db(linear: np.ndarray) -> np.ndarray:
    """Linear power to dB, with non-positive samples marked invalid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.where(linear > 0, linear, np.nan))
    return db.astype(np.float32)


def to_linear(db: np.ndarray) -> np.ndarray:
    """dB back to linear power."""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def calibrate(
    scene: Scene,
    settings: Optional[Settings] = None,
    bands: Optional[Sequence[int]] = None,
) -> RasterImage:
    """Read a scene and return calibrated sigma0 backscatter in dB.

    Products already stored in dB (which is how most public SAR training sets
    ship) are passed through rather than squared a second time; set
    ``preprocessing.input_is_db`` in the config to override the detection.
    """
    settings = settings or get_settings()
    cfg = settings.preprocessing

    if scene.path is None:
        raise PreprocessingError(
            f"scene {scene.scene_id} has no local path; fetch it before preprocessing"
        )
    path = Path(scene.path)
    if not path.exists():
        raise PreprocessingError(f"scene {scene.scene_id} is missing from disk: {path}")

    with rasterio.open(path) as dataset:
        indexes = list(bands) if bands else list(dataset.indexes)
        raw = dataset.read(indexes).astype(np.float32)
        transform = dataset.transform
        crs = dataset.crs
        src_nodata = dataset.nodata
        gcps, gcp_crs = dataset.get_gcps()
        descriptions = [dataset.descriptions[i - 1] for i in indexes]

    if src_nodata is not None:
        raw = np.where(raw == src_nodata, np.nan, raw)

    is_db = cfg.input_is_db if cfg.input_is_db is not None else looks_like_db(raw)
    if is_db:
        logger.debug("scene %s already in dB; skipping DN conversion", scene.scene_id)
        db = raw.astype(np.float32)
    else:
        db = to_db(dn_to_sigma0(raw, cfg.calibration_constant_db))

    band_names = [
        name or default
        for name, default in zip(descriptions, _default_band_names(scene, len(indexes)))
    ]

    return RasterImage(
        data=db,
        transform=transform,
        crs=crs,
        nodata=cfg.nodata,
        band_names=band_names,
        scene_id=scene.scene_id,
        gcps=gcps or None,
        gcp_crs=gcp_crs,
    )


def _default_band_names(scene: Scene, count: int) -> List[str]:
    """Name bands after the scene's polarisations when we know them."""
    if len(scene.polarisations) == count:
        return list(scene.polarisations)
    return [f"band_{i + 1}" for i in range(count)]


# --------------------------------------------------------------------------- #
# 2. speckle filtering
# --------------------------------------------------------------------------- #
def _local_stats(
    data: np.ndarray, valid: np.ndarray, size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware local mean and variance over a ``size`` square window."""
    filled = np.where(valid, data, 0.0)
    weight = uniform_filter(valid.astype(np.float64), size=size, mode="nearest")
    mean = uniform_filter(filled, size=size, mode="nearest")
    mean_sq = uniform_filter(np.square(filled), size=size, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(weight > 0, mean / weight, np.nan)
        mean_sq = np.where(weight > 0, mean_sq / weight, np.nan)
    variance = np.maximum(mean_sq - np.square(mean), 0.0)
    return mean, variance


def _lee_weight(mean: np.ndarray, variance: np.ndarray, looks: float) -> np.ndarray:
    """Lee's adaptive weight: 0 in flat areas, 1 on edges and point targets."""
    cu2 = 1.0 / max(looks, EPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = (variance - cu2 * np.square(mean)) / (variance * (1.0 + cu2) + EPS)
    return np.clip(np.nan_to_num(weight, nan=0.0), 0.0, 1.0)


def _directional_kernels(size: int) -> np.ndarray:
    """Eight half-window masks of a ``size`` square, one per edge direction.

    Each mask keeps the half of the window lying on one side of an edge through
    the centre, which is what lets refined Lee average along an edge instead of
    across it.
    """
    radius = size // 2
    rows, cols = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    halves = [
        cols >= 0,  # E
        (cols - rows) >= 0,  # NE
        rows <= 0,  # N
        (cols + rows) <= 0,  # NW
        cols <= 0,  # W
        (cols - rows) <= 0,  # SW
        rows >= 0,  # S
        (cols + rows) >= 0,  # SE
    ]
    return np.stack([h.astype(np.float64) for h in halves])


def _lee(
    data: np.ndarray, valid: np.ndarray, size: int, looks: float
) -> np.ndarray:
    """Classic Lee filter on linear intensity."""
    mean, variance = _local_stats(data, valid, size)
    weight = _lee_weight(mean, variance, looks)
    filtered = mean + weight * (np.where(valid, data, mean) - mean)
    return np.where(valid, filtered, np.nan)


def _refined_lee(
    data: np.ndarray, valid: np.ndarray, size: int, looks: float
) -> np.ndarray:
    """Lee filtering restricted to the most homogeneous half-window.

    For every pixel the eight directional half-windows are scored against the
    pixel's own small-window mean; the closest one is taken to lie on the same
    side of any edge, and Lee's weight is then computed from that half alone.
    """
    kernels = _directional_kernels(size)
    filled = np.where(valid, data, 0.0)
    valid_f = valid.astype(np.float64)

    means = np.empty((len(kernels), *data.shape))
    variances = np.empty_like(means)
    for i, kernel in enumerate(kernels):
        weight = correlate(valid_f, kernel, mode="nearest")
        total = correlate(filled, kernel, mode="nearest")
        total_sq = correlate(np.square(filled), kernel, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(weight > 0, total / weight, np.nan)
            mean_sq = np.where(weight > 0, total_sq / weight, np.nan)
        means[i] = mean
        variances[i] = np.maximum(mean_sq - np.square(mean), 0.0)

    local_mean, _ = _local_stats(data, valid, 3)
    distance = np.abs(means - local_mean[np.newaxis, ...])
    chosen = np.nanargmin(np.where(np.isfinite(distance), distance, np.inf), axis=0)

    take = chosen[np.newaxis, ...]
    mean = np.take_along_axis(means, take, axis=0)[0]
    variance = np.take_along_axis(variances, take, axis=0)[0]

    weight = _lee_weight(mean, variance, looks)
    filtered = mean + weight * (np.where(valid, data, mean) - mean)
    return np.where(valid, filtered, np.nan)


def speckle_filter(
    image: ImageLike,
    method: Optional[str] = None,
    size: Optional[int] = None,
    looks: Optional[float] = None,
    in_db: bool = True,
    settings: Optional[Settings] = None,
) -> ImageLike:
    """Despeckle SAR backscatter, preserving the input's shape and type.

    Filtering runs on linear intensity, since speckle is multiplicative there;
    dB input (the default, and what :func:`calibrate` produces) is converted in
    and back out. Invalid pixels stay invalid.
    """
    settings = settings or get_settings()
    cfg = settings.preprocessing
    method = (method or cfg.speckle_filter).lower()
    size = size or cfg.speckle_filter_size
    looks = looks if looks is not None else cfg.number_of_looks

    if method == "none":
        return image
    if size % 2 == 0:
        raise PreprocessingError("speckle filter window size must be odd")

    data = _data_of(image)
    out = np.empty_like(data, dtype=np.float32)
    for index in range(data.shape[0]):
        band = data[index].astype(np.float64)
        valid = np.isfinite(band)
        if not valid.any():
            out[index] = band.astype(np.float32)
            continue

        intensity = to_linear(band) if in_db else band
        if method == "lee":
            filtered = _lee(intensity, valid, size, looks)
        elif method == "refined_lee":
            filtered = _refined_lee(intensity, valid, size, looks)
        else:
            raise PreprocessingError(f"unknown speckle filter {method!r}")

        out[index] = (to_db(filtered) if in_db else filtered).astype(np.float32)

    return _like(image, out)


# --------------------------------------------------------------------------- #
# 3. geocoding
# --------------------------------------------------------------------------- #
def geocode(
    image: RasterImage,
    scene_metadata: Optional[Union[Scene, Dict[str, Any]]] = None,
    dst_crs: Optional[str] = None,
    resolution_m: Optional[float] = None,
    resampling: Resampling = Resampling.bilinear,
    settings: Optional[Settings] = None,
) -> RasterImage:
    """Guarantee every pixel has a lon/lat position.

    Handles the three states a scene arrives in:

    * already map-projected in the target CRS - returned untouched, since
      resampling would only blur it;
    * map-projected in some other CRS - warped across;
    * ungeoreferenced but carrying GCPs (Sentinel-1 GRD in radar geometry) -
      a transform is fitted to the GCPs first, then warped.
    """
    settings = settings or get_settings()
    cfg = settings.preprocessing
    target = CRS.from_string(dst_crs or cfg.output_crs)

    src_crs = image.crs
    src_transform = image.transform

    if src_crs is None and image.gcps:
        logger.info("fitting a transform to %d GCPs", len(image.gcps))
        src_transform = from_gcps(image.gcps)
        src_crs = image.gcp_crs or CRS.from_epsg(4326)

    if src_crs is None:
        raise PreprocessingError(
            f"scene {image.scene_id} has neither a CRS nor GCPs, so its pixels "
            "cannot be placed on the ground"
        )

    if src_crs == target and resolution_m is None:
        return replace(image, crs=src_crs, transform=src_transform)

    resolution = _target_resolution(target, resolution_m or cfg.target_resolution_m)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        target,
        image.width,
        image.height,
        *_source_bounds(src_transform, image.width, image.height),
        resolution=resolution,
    )

    destination = np.full(
        (image.count, dst_height, dst_width), np.nan, dtype=np.float32
    )
    reproject(
        source=image.data,
        destination=destination,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling,
    )

    return replace(
        image, data=destination, transform=dst_transform, crs=target, gcps=None
    )


def _source_bounds(
    transform: Affine, width: int, height: int
) -> Tuple[float, float, float, float]:
    west, north = transform @ (0, 0)
    east, south = transform @ (width, height)
    return (min(west, east), min(south, north), max(west, east), max(south, north))


def _target_resolution(crs: CRS, resolution_m: float) -> float:
    """Pixel size in the target CRS's own units."""
    return resolution_m / METRES_PER_DEGREE if crs.is_geographic else resolution_m


# --------------------------------------------------------------------------- #
# 4. land masking
# --------------------------------------------------------------------------- #
def load_coastline(
    path: Optional[Path] = None, settings: Optional[Settings] = None
) -> "Any":
    """Read the land polygons used for masking as a GeoDataFrame.

    Defaults to the clipped Natural Earth 10 m land layer vendored in
    ``configs/coastline/``. Call :func:`download_coastline` for global coverage.
    """
    import geopandas as gpd

    settings = settings or get_settings()
    source = Path(path) if path else settings.paths.resolve(
        settings.preprocessing.coastline_path
    )
    if not source.exists():
        raise PreprocessingError(
            f"coastline layer not found at {source}. Run "
            "src.preprocessing.pipeline.download_coastline() for the full "
            "Natural Earth 10 m land layer, or point "
            "preprocessing.coastline_path at your own."
        )
    return gpd.read_file(source)


def download_coastline(
    dest: Optional[Path] = None, settings: Optional[Settings] = None
) -> Path:
    """Fetch the full Natural Earth 10 m land layer.

    The repo vendors only a clip around the demo AOI, which is enough for the
    tests and the demo but not for scenes elsewhere in the world.
    """
    import geopandas as gpd
    import requests

    settings = settings or get_settings()
    cfg = settings.preprocessing
    target = Path(dest) if dest else settings.paths.resolve(
        cfg.coastline_path
    ).with_name("ne_10m_land.geojson")
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s", cfg.coastline_url)
    response = requests.get(cfg.coastline_url, timeout=300)
    response.raise_for_status()

    archive = target.with_suffix(".zip")
    archive.write_bytes(response.content)
    gpd.read_file(f"zip://{archive}").to_file(target, driver="GeoJSON")
    archive.unlink()
    logger.info("coastline written to %s", target)
    return target


def land_mask(
    shape_hw: Tuple[int, int],
    transform: Affine,
    crs: Optional[CRS] = None,
    coastline: Optional[Any] = None,
    buffer_m: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> np.ndarray:
    """Boolean raster, True over land (and the coastal buffer).

    Split out from :func:`mask_land` so the geometry can be inspected and
    tested on its own.
    """
    settings = settings or get_settings()
    cfg = settings.preprocessing
    buffer_m = cfg.land_buffer_m if buffer_m is None else buffer_m
    height, width = shape_hw

    geometries = _coastline_geometries(coastline, settings)
    if not geometries:
        return np.zeros((height, width), dtype=bool)

    raster_crs = CRS.from_user_input(crs) if crs is not None else CRS.from_epsg(4326)
    bounds = _source_bounds(transform, width, height)
    window = box(*bounds)

    buffer_units = (
        buffer_m / METRES_PER_DEGREE if raster_crs.is_geographic else buffer_m
    )
    selected = []
    for geometry in geometries:
        if not geometry.intersects(window):
            continue
        selected.append(geometry.buffer(buffer_units) if buffer_units else geometry)

    if not selected:
        return np.zeros((height, width), dtype=bool)

    return rasterize(
        [(geometry, 1) for geometry in selected],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def _coastline_geometries(
    coastline: Optional[Any], settings: Settings
) -> List[BaseGeometry]:
    """Normalise whatever the caller passed into a list of shapely geometries."""
    if coastline is None:
        coastline = load_coastline(settings=settings)

    if isinstance(coastline, (str, Path)):
        coastline = load_coastline(path=Path(coastline), settings=settings)

    if isinstance(coastline, BaseGeometry):
        return [coastline]

    if hasattr(coastline, "geometry"):  # GeoDataFrame / GeoSeries
        return [g for g in coastline.geometry if g is not None and not g.is_empty]

    if isinstance(coastline, dict):  # GeoJSON
        if coastline.get("type") == "FeatureCollection":
            return [shape(f["geometry"]) for f in coastline.get("features", [])]
        if coastline.get("type") == "Feature":
            return [shape(coastline["geometry"])]
        return [shape(coastline)]

    if isinstance(coastline, (list, tuple)):
        return [
            g if isinstance(g, BaseGeometry) else shape(g)
            for g in coastline
        ]

    raise PreprocessingError(
        f"cannot interpret {type(coastline).__name__} as a coastline layer"
    )


def mask_land(
    image: ImageLike,
    transform: Optional[Affine] = None,
    crs: Optional[CRS] = None,
    coastline: Optional[Any] = None,
    buffer_m: Optional[float] = None,
    nodata: Optional[float] = None,
    settings: Optional[Settings] = None,
) -> ImageLike:
    """Blank out land pixels so only water reaches the detector.

    Masked pixels become ``nodata`` (NaN by default) rather than zero, so they
    stay distinguishable from genuinely dark water - which is exactly what an
    oil slick looks like.
    """
    settings = settings or get_settings()
    if transform is None:
        if not isinstance(image, RasterImage):
            raise PreprocessingError("mask_land needs a transform for a bare array")
        transform = image.transform
    if crs is None and isinstance(image, RasterImage):
        crs = image.crs
    if nodata is None:
        nodata = image.nodata if isinstance(image, RasterImage) else settings.preprocessing.nodata

    data = _data_of(image)
    mask = land_mask(
        (data.shape[1], data.shape[2]),
        transform,
        crs=crs,
        coastline=coastline,
        buffer_m=buffer_m,
        settings=settings,
    )

    masked = np.where(mask[np.newaxis, ...], np.float32(nodata), data).astype(np.float32)
    logger.debug("masked %.1f%% of pixels as land", 100.0 * mask.mean())
    return _like(image, masked)


# --------------------------------------------------------------------------- #
# 5. tiling
# --------------------------------------------------------------------------- #
def _offsets(extent: int, tile_size: int, stride: int) -> List[int]:
    """Tile start positions along one axis, with the last one flush to the edge."""
    if extent <= tile_size:
        return [0]
    positions = list(range(0, extent - tile_size + 1, stride))
    if positions[-1] != extent - tile_size:
        positions.append(extent - tile_size)
    return positions


def tile(
    image: ImageLike,
    transform: Optional[Affine] = None,
    tile_size: int = 512,
    overlap: Optional[int] = None,
    scene_id: Optional[str] = None,
    crs: Optional[CRS] = None,
    nodata: Optional[float] = None,
    min_valid_fraction: float = 0.0,
    settings: Optional[Settings] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """Cut the image into georeferenced tiles plus an index describing them.

    Overlap defaults to ``detection.tile_overlap``; it exists so a slick sitting
    on a tile boundary is seen whole in at least one tile.

    Each index entry records the tile's pixel offset in the scene and its own
    geotransform, which is what lets a detection found in tile pixel space be
    put back into scene pixel space and then into map coordinates.

    Tiles smaller than ``tile_size`` are never produced: the last row and column
    are pulled flush with the image edge instead, so every tile is model-ready.
    """
    settings = settings or get_settings()
    if overlap is None:
        overlap = settings.detection.tile_overlap
    if transform is None:
        if not isinstance(image, RasterImage):
            raise PreprocessingError("tile needs a transform for a bare array")
        transform = image.transform
    if crs is None and isinstance(image, RasterImage):
        crs = image.crs
    if nodata is None:
        nodata = image.nodata if isinstance(image, RasterImage) else float("nan")
    if scene_id is None and isinstance(image, RasterImage):
        scene_id = image.scene_id

    if overlap >= tile_size:
        raise PreprocessingError("tile overlap must be smaller than the tile size")

    data = _data_of(image)
    _, height, width = data.shape
    stride = tile_size - overlap

    tiles: List[np.ndarray] = []
    index: List[Dict[str, Any]] = []
    for row_off in _offsets(height, tile_size, stride):
        for col_off in _offsets(width, tile_size, stride):
            patch = data[
                :,
                row_off : row_off + tile_size,
                col_off : col_off + tile_size,
            ]
            valid_fraction = _valid_fraction(patch, nodata)
            if valid_fraction < min_valid_fraction:
                continue

            window = Window(col_off, row_off, patch.shape[2], patch.shape[1])
            patch_transform = window_transform(window, transform)
            tiles.append(patch.copy())
            index.append(
                {
                    "tile_id": f"{scene_id or 'scene'}_r{row_off:05d}_c{col_off:05d}",
                    "scene_id": scene_id,
                    "row_off": int(row_off),
                    "col_off": int(col_off),
                    "height": int(patch.shape[1]),
                    "width": int(patch.shape[2]),
                    "transform": [float(v) for v in patch_transform.to_gdal()],
                    "crs": crs.to_string() if crs is not None else None,
                    "bounds": [
                        float(v)
                        for v in _source_bounds(
                            patch_transform, patch.shape[2], patch.shape[1]
                        )
                    ],
                    "valid_fraction": float(valid_fraction),
                }
            )

    logger.info("cut %d tile(s) of %dpx (overlap %d)", len(tiles), tile_size, overlap)
    return tiles, index


def _valid_fraction(patch: np.ndarray, nodata: float) -> float:
    """Share of pixels in a patch that carry real data."""
    if patch.size == 0:
        return 0.0
    if math.isnan(nodata):
        valid = np.isfinite(patch)
    else:
        valid = np.isfinite(patch) & (patch != nodata)
    return float(valid.mean())


def tile_index_to_frame(index: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Tile index as a DataFrame with the nested fields flattened for parquet."""
    frame = pd.DataFrame(list(index))
    if frame.empty:
        return frame

    transforms = pd.DataFrame(
        frame.pop("transform").tolist(),
        columns=["gt_origin_x", "gt_pixel_width", "gt_row_rotation",
                 "gt_origin_y", "gt_col_rotation", "gt_pixel_height"],
        index=frame.index,
    )
    bounds = pd.DataFrame(
        frame.pop("bounds").tolist(),
        columns=["west", "south", "east", "north"],
        index=frame.index,
    )
    return pd.concat([frame, transforms, bounds], axis=1)


def transform_from_index_row(row: Union[pd.Series, Dict[str, Any]]) -> Affine:
    """Rebuild a tile's geotransform from an index row - the stitch-back path."""
    return Affine.from_gdal(
        float(row["gt_origin_x"]),
        float(row["gt_pixel_width"]),
        float(row["gt_row_rotation"]),
        float(row["gt_origin_y"]),
        float(row["gt_col_rotation"]),
        float(row["gt_pixel_height"]),
    )


# --------------------------------------------------------------------------- #
# 6. the whole chain
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    """What :func:`run_pipeline` produced for one scene."""

    scene_id: str
    output_dir: Path
    tile_index_path: Path
    tiles: List[Path]
    index: pd.DataFrame
    image: RasterImage

    def __len__(self) -> int:
        return len(self.tiles)


def run_pipeline(
    scene: Scene,
    output_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
    tile_size: Optional[int] = None,
    overlap: Optional[int] = None,
    coastline: Optional[Any] = None,
    write_tiles: bool = True,
) -> PreprocessResult:
    """Calibrate, despeckle, geocode, mask land and tile one scene.

    Writes each tile as a GeoTIFF to ``data/processed/<scene_id>/`` and the tile
    index to ``tile_index.parquet`` alongside them.
    """
    settings = settings or get_settings()
    cfg = settings.preprocessing
    tile_size = tile_size or settings.detection.tile_size
    overlap = settings.detection.tile_overlap if overlap is None else overlap

    logger.info("preprocessing scene %s", scene.scene_id)
    image = calibrate(scene, settings=settings)
    image = speckle_filter(image, settings=settings)
    image = geocode(image, scene, settings=settings)
    image = mask_land(image, coastline=coastline, settings=settings)

    tiles, index = tile(
        image,
        tile_size=tile_size,
        overlap=overlap,
        min_valid_fraction=cfg.tile_min_valid_fraction,
        settings=settings,
    )

    out_dir = Path(output_dir) if output_dir else settings.paths.resolve(
        settings.paths.processed_dir
    ) / scene.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    if write_tiles:
        for patch, entry in zip(tiles, index):
            path = out_dir / f"{entry['tile_id']}.tif"
            _write_tile(path, patch, entry, image)
            entry["path"] = str(path)
            written.append(path)

    frame = tile_index_to_frame(index)
    index_path = out_dir / "tile_index.parquet"
    frame.to_parquet(index_path, index=False)
    logger.info("wrote %d tile(s) and %s", len(written), index_path)

    return PreprocessResult(
        scene_id=scene.scene_id,
        output_dir=out_dir,
        tile_index_path=index_path,
        tiles=written,
        index=frame,
        image=image,
    )


def _write_tile(
    path: Path, patch: np.ndarray, entry: Dict[str, Any], image: RasterImage
) -> None:
    """Write one tile as a compressed GeoTIFF carrying its own georeferencing."""
    profile = image.profile(
        height=patch.shape[1],
        width=patch.shape[2],
        count=patch.shape[0],
        transform=Affine.from_gdal(*entry["transform"]),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(patch.astype(np.float32))
        for band_index, name in enumerate(image.band_names[: patch.shape[0]], start=1):
            dst.set_band_description(band_index, name)
        dst.update_tags(
            scene_id=str(entry.get("scene_id")),
            tile_id=entry["tile_id"],
            row_off=entry["row_off"],
            col_off=entry["col_off"],
        )


__all__ = [
    "RasterImage",
    "PreprocessResult",
    "PreprocessingError",
    "calibrate",
    "speckle_filter",
    "geocode",
    "mask_land",
    "land_mask",
    "tile",
    "run_pipeline",
    "load_coastline",
    "download_coastline",
    "tile_index_to_frame",
    "transform_from_index_row",
    "dn_to_sigma0",
    "to_db",
    "to_linear",
    "looks_like_db",
]
