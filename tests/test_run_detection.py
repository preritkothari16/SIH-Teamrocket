"""Integration tests for the Phase 1 chain script (scripts/run_detection.py).

These exercise the actual sequence run_detection.py wires together -
run_pipeline -> infer_scene -> stitch_backscatter -> filter_lookalikes ->
spills_from_blobs - end to end on synthetic scenes written to disk, rather
than unit-testing each stage in isolation. That sequencing is exactly where
the two bugs this module guards against used to live:

* a scene that actually needs reprojection (real Sentinel-1 GRD, unlike the
  one local dataset this repo has, which happens to need no reprojection);
* land sitting close enough to a detection to reach into the look-alike
  filter's contrast ring.

No trained checkpoint exists yet, so every test here uses the same
``ThresholdStubModel`` the script itself falls back to.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from src.config import load_settings
from src.detection.dataset import OIL_CLASS
from src.detection.infer import infer_scene, stitch_backscatter
from src.detection.lookalike_filter import filter_lookalikes
from src.ingestion.types import Scene, SceneSourceKind
from src.preprocessing.pipeline import calibrate, run_pipeline
from scripts.run_detection import ThresholdStubModel, run

# Away from configs/coastline's vendored (Gujarat-only) coverage, so a scene
# placed here is never accidentally land-masked by the real coastline data.
ORIGIN_LON, ORIGIN_LAT = -160.0, -5.0
PIXEL_DEG = 0.001


class DarkThresholdStub(torch.nn.Module):
    """Absolute-cutoff stand-in, deterministic regardless of how much of a
    tile is actually painted dark.

    Unlike ``ThresholdStubModel`` (the real script's per-tile 15th-percentile
    threshold), a fixed cutoff on the normalised model input is what these
    alignment checks need: they paint a small, precisely-placed dark patch and
    must know exactly which pixels get called oil, not "whatever the darkest
    15% of this tile happens to be" when most of the tile is flat sea.
    """

    def __init__(self, num_classes: int = 5, cutoff: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cutoff = cutoff

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        intensity = batch.mean(dim=1, keepdim=True)
        logits = torch.zeros(
            batch.shape[0], self.num_classes, batch.shape[2], batch.shape[3],
            dtype=batch.dtype, device=batch.device,
        )
        dark = (intensity <= self.cutoff).squeeze(1)
        logits[:, 0] = torch.where(dark, -4.0, 4.0)
        logits[:, OIL_CLASS] = torch.where(dark, 4.0, -4.0)
        return logits


def write_scene_tif(
    path: Path,
    data: np.ndarray,
    crs: str = "EPSG:4326",
    transform: Affine | None = None,
) -> Path:
    """Write a synthetic single-band scene to disk."""
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2],
        count=data.shape[0], dtype="float32", crs=crs,
        transform=transform if transform is not None else from_origin(
            ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG
        ),
    ) as dst:
        dst.write(data.astype("float32"))
    return path


def make_scene(path: Path, scene_id: str = "synthetic_scene") -> Scene:
    with rasterio.open(path) as dataset:
        bounds = dataset.bounds
    return Scene(
        scene_id=scene_id,
        source=SceneSourceKind.LOCAL,
        acquisition_time="2023-05-14T00:33:32+00:00",
        footprint=box(*bounds),
        path=path,
    )


def painted_streak(
    shape: Tuple[int, int], centre, semi_major=35.0, semi_minor=6.0, angle_deg=0.0
) -> np.ndarray:
    """A long, thin, hard-edged blob - clears the look-alike filter on its own."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dr = rows - centre[0]
    dc = cols - centre[1]
    theta = np.radians(angle_deg)
    major = dc * np.cos(theta) + dr * np.sin(theta)
    minor = -dc * np.sin(theta) + dr * np.cos(theta)
    return (major / semi_major) ** 2 + (minor / semi_minor) ** 2 <= 1.0


# --------------------------------------------------------------------------- #
# smoke test: the script's own wiring, end to end
# --------------------------------------------------------------------------- #
def test_run_end_to_end_smoke(tmp_path: Path) -> None:
    """run() must still work after the fix - proves the wiring, not the physics."""
    shape = (128, 128)
    data = np.full(shape, -10.0, dtype=np.float32)
    data[painted_streak(shape, (64, 64))] = -19.0
    scene_path = write_scene_tif(tmp_path / "scene.tif", data)

    settings = load_settings()
    settings = settings.model_copy(update={
        "paths": settings.paths.model_copy(update={
            "processed_dir": tmp_path / "processed",
            "raw_dir": tmp_path / "raw",
        }),
    })

    output = tmp_path / "spills.geojson"
    collection = run(
        scene_path=scene_path,
        stub_model=True,
        tile_size=128,
        overlap=0,
        output=output,
        settings=settings,
    )

    assert output.is_file()
    assert json.loads(output.read_text())["type"] == "FeatureCollection"
    assert collection["type"] == "FeatureCollection"
    assert collection["properties"]["spill_count"] == len(collection["features"])


# --------------------------------------------------------------------------- #
# fix #2a: backscatter and mask must stay aligned when geocoding reprojects
# --------------------------------------------------------------------------- #
def test_backscatter_stays_aligned_with_the_mask_when_geocoding_reprojects(
    tmp_path: Path,
) -> None:
    """The bug this guards against: scene_backscatter() used to re-read the raw,
    ungeocoded scene and pad/crop it to shape - which only accidentally lined
    up with the mask when reprojection was a no-op. This scene is deliberately
    in a projected CRS, so geocode() actually warps it.
    """
    shape = (160, 160)
    sea_db = -10.0
    data = np.full(shape, sea_db, dtype=np.float32)
    oil = painted_streak(shape, (80, 80), semi_major=35.0, semi_minor=6.0, angle_deg=20.0)
    data[oil] = sea_db - 9.0

    scene = make_scene(
        write_scene_tif(
            tmp_path / "utm_scene.tif",
            data,
            crs="EPSG:32643",
            transform=from_origin(200_000.0, 2_400_000.0, 100.0, 100.0),
        ),
        "utm_scene",
    )

    settings = load_settings().model_copy(update={
        "preprocessing": load_settings().preprocessing.model_copy(
            update={"speckle_filter": "none"}
        ),
    })

    scene_dir = tmp_path / "processed" / scene.scene_id
    run_pipeline(
        scene, output_dir=scene_dir, tile_size=512, overlap=0,
        coastline=box(0.0, 0.0, 1.0, 1.0),  # land far away, nothing masked
        settings=settings,
    )

    model = DarkThresholdStub(num_classes=settings.training.num_classes)
    result = infer_scene(scene_dir, model=model, scene_id=scene.scene_id, settings=settings)
    assert (result.mask == OIL_CLASS).any(), "the painted patch must survive reprojection"

    backscatter = stitch_backscatter(scene_dir, grid=result.grid)
    assert backscatter.shape[1:] == result.mask.shape

    # the whole point: wherever the mask says oil, the reconstructed
    # backscatter must actually BE dark there, not sea-level or NaN from a
    # misaligned crop.
    oil_pixels = backscatter[0][result.mask == OIL_CLASS]
    oil_pixels = oil_pixels[np.isfinite(oil_pixels)]
    assert oil_pixels.size > 0
    assert oil_pixels.mean() < sea_db - 3.0

    # this must not raise: filter_lookalikes hard-checks shape equality, and a
    # silently-misaligned crop would still have "matched" that check before -
    # the assertion above is what actually catches misalignment.
    filtered = filter_lookalikes(result.mask, backscatter, settings=settings)
    assert len(filtered.kept) >= 1


# --------------------------------------------------------------------------- #
# fix #2b: land must not contaminate the look-alike ring
# --------------------------------------------------------------------------- #
def test_land_backscatter_does_not_leak_into_the_lookalike_ring(tmp_path: Path) -> None:
    """Land is painted with a very different backscatter level than the sea.

    Under the old scene_backscatter(), land was never masked at all (it read
    straight from calibrate(), which runs before mask_land), so a coastal
    blob's surrounding ring would sample that value instead of open water.
    """
    shape = (160, 160)
    sea_db = -10.0
    land_db = 8.0  # rough terrain reads bright and unlike open water

    data = np.full(shape, sea_db, dtype=np.float32)
    data[:80, :] = land_db  # northern half is land
    oil = painted_streak(shape, (95, 80), semi_major=35.0, semi_minor=6.0, angle_deg=0.0)
    data[oil] = sea_db - 9.0

    scene_path = write_scene_tif(tmp_path / "coastal_scene.tif", data)
    scene = make_scene(scene_path, "coastal_scene")

    land_polygon = box(
        ORIGIN_LON - 0.01, ORIGIN_LAT - 80 * PIXEL_DEG, ORIGIN_LON + 0.5, ORIGIN_LAT + 0.01
    )
    land_geojson = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": mapping(land_polygon), "properties": {}}],
    }

    settings = load_settings()
    scene_dir = tmp_path / "processed" / scene.scene_id
    run_pipeline(
        scene, output_dir=scene_dir, tile_size=160, overlap=0,
        coastline=land_geojson, settings=settings,
    )

    model = DarkThresholdStub(num_classes=settings.training.num_classes)
    result = infer_scene(scene_dir, model=model, scene_id=scene.scene_id, settings=settings)

    backscatter = stitch_backscatter(scene_dir, grid=result.grid)
    assert np.isnan(backscatter[0, :80, :]).all(), "land must stay NaN, never a real value"

    filtered_correct = filter_lookalikes(result.mask, backscatter, settings=settings)
    coastal = next(b for b in filtered_correct.blobs if b.centroid[0] > 80)
    # the ring reaches into land (rows < 80); with land correctly NaN'd out,
    # the surrounding water must read close to the true sea level.
    assert coastal.surround_db == pytest.approx(sea_db, abs=1.5)

    # reproduce the old bug for comparison: a raw, unmasked read of the scene.
    unmasked = calibrate(scene, settings=settings).data[0]
    filtered_contaminated = filter_lookalikes(result.mask, unmasked, settings=settings)
    contaminated = next(b for b in filtered_contaminated.blobs if b.centroid[0] > 80)
    assert contaminated.surround_db != pytest.approx(sea_db, abs=1.5), (
        "the old unmasked path should visibly disagree - that disagreement is the bug"
    )
