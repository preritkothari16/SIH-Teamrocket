"""src/ingestion/poller.py + scripts/run_live.py tests.

A stub catalogue (satisfying src.ingestion.types.SceneSource: search()/
fetch(), no network) stands in for CDSECatalogue throughout - proving the
whole poll -> download -> run_pipeline chain works without live Copernicus
access, per this step's acceptance criterion. The stub's "download" just
hands back an already-written synthetic scene's local path, mirroring how
LocalSceneSource.fetch() treats an already-local scene.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from src.config import load_settings
from src.ingestion.poller import load_last_checked, poll_once, save_last_checked
from src.ingestion.types import Scene, SceneSourceKind

from scripts import run_live

ORIGIN_LON, ORIGIN_LAT = -150.0, -10.0
PIXEL_DEG = 0.001
SCENE_SHAPE = (128, 128)


def painted_streak(shape, centre, semi_major=65.0, semi_minor=16.0):
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dr = rows - centre[0]
    dc = cols - centre[1]
    return (dc / semi_major) ** 2 + (dr / semi_minor) ** 2 <= 1.0


def write_scene_tif(path: Path) -> Path:
    """A noisy sea background with a painted oil-sized streak - same recipe
    as test_pipeline_integration.py's fixture, so the stub-model detector
    reliably finds something to alert on."""
    rng = np.random.default_rng(0)
    data = (-10.0 + rng.normal(0.0, 0.3, size=SCENE_SHAPE)).astype(np.float32)
    data[painted_streak(SCENE_SHAPE, (64, 64))] = -19.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=SCENE_SHAPE[0], width=SCENE_SHAPE[1], count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG),
    ) as dst:
        dst.write(data[np.newaxis, ...])
    return path


def scene_footprint():
    return box(
        ORIGIN_LON, ORIGIN_LAT - SCENE_SHAPE[0] * PIXEL_DEG,
        ORIGIN_LON + SCENE_SHAPE[1] * PIXEL_DEG, ORIGIN_LAT,
    )


class StubCatalogue:
    """A SceneSource satisfying search()/fetch() with canned scenes - no
    network, standing in for CDSECatalogue in every test here."""

    def __init__(self, scenes: List[Scene]) -> None:
        self._scenes = scenes
        self.fetch_calls: List[str] = []
        self.search_calls: List[tuple] = []

    def search(self, aoi, start, end, limit: Optional[int] = None) -> List[Scene]:
        self.search_calls.append((start, end))
        return [s for s in self._scenes if start <= s.acquisition_time <= end]

    def fetch(self, scene: Scene) -> Path:
        self.fetch_calls.append(scene.scene_id)
        return Path(scene.path)


@pytest.fixture
def aoi_path(tmp_path: Path) -> Path:
    path = tmp_path / "aoi.geojson"
    path.write_text(
        json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(scene_footprint())}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def settings(tmp_path: Path):
    base = load_settings()
    return base.model_copy(update={
        "paths": base.paths.model_copy(update={
            "processed_dir": tmp_path / "processed",
            "raw_dir": tmp_path / "raw",
        }),
    })


# --------------------------------------------------------------------------- #
# poll_once / state persistence
# --------------------------------------------------------------------------- #
def test_poll_once_returns_only_scenes_strictly_newer_than_since(settings) -> None:
    since = datetime(2023, 6, 1, tzinfo=timezone.utc)
    on_boundary = Scene(
        scene_id="boundary", source=SceneSourceKind.CDSE,
        acquisition_time=since, footprint=box(0, 0, 1, 1),
    )
    newer = Scene(
        scene_id="newer", source=SceneSourceKind.CDSE,
        acquisition_time=since + timedelta(hours=1), footprint=box(0, 0, 1, 1),
        path=Path("unused"),
    )
    catalogue = StubCatalogue([on_boundary, newer])

    result = poll_once(box(-180, -90, 180, 90), since, catalogue=catalogue, settings=settings)

    assert [s.scene_id for s in result.new_scenes] == ["newer"]
    assert catalogue.fetch_calls == ["newer"]  # the boundary scene is never re-fetched


def test_poll_once_downloads_every_new_scene(tmp_path: Path, settings) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    scene_path = write_scene_tif(tmp_path / "scene.tif")
    scene = Scene(
        scene_id="00000", source=SceneSourceKind.CDSE,
        acquisition_time=datetime.now(timezone.utc) - timedelta(minutes=1),
        footprint=scene_footprint(), path=scene_path,
    )
    catalogue = StubCatalogue([scene])

    result = poll_once(scene_footprint(), since, catalogue=catalogue, settings=settings)

    assert len(result.new_scenes) == 1
    assert result.new_scenes[0].path == scene_path
    assert catalogue.fetch_calls == ["00000"]


def test_poll_once_can_skip_downloading(tmp_path: Path, settings) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    scene = Scene(
        scene_id="00000", source=SceneSourceKind.CDSE,
        acquisition_time=datetime.now(timezone.utc) - timedelta(minutes=1),
        footprint=scene_footprint(),
    )
    catalogue = StubCatalogue([scene])

    result = poll_once(
        scene_footprint(), since, catalogue=catalogue, settings=settings, download=False,
    )
    assert len(result.new_scenes) == 1
    assert catalogue.fetch_calls == []


def test_load_last_checked_falls_back_to_a_lookback_window_when_no_state_exists(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    before = datetime.now(timezone.utc) - timedelta(hours=3.0, seconds=5)
    fallback = load_last_checked(state_path, default_lookback_hours=3.0)
    assert before < fallback < datetime.now(timezone.utc)


def test_save_and_load_last_checked_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    checked_at = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
    save_last_checked(state_path, checked_at)
    assert load_last_checked(state_path, default_lookback_hours=3.0) == checked_at


# --------------------------------------------------------------------------- #
# run_live.run_forever - the actual acceptance criterion for this step
# --------------------------------------------------------------------------- #
def test_run_live_runs_the_pipeline_for_each_new_scene_without_network(
    tmp_path: Path, aoi_path: Path, settings,
) -> None:
    """The whole point of this step: run_live.py must complete a poll cycle
    and trigger scripts/run_pipeline.py end to end against a stubbed
    catalogue, with no live Copernicus access anywhere in the process.

    Filename matches the scene's id, matching how the real
    CDSECatalogue.fetch() names what it downloads (``<scene_id>.zip``) -
    scripts/run_pipeline.py re-reads the scene from its path via
    LocalSceneSource, which derives scene_id from the filename, so a real
    fetch and this stub agree on id the same way.
    """
    scene_path = write_scene_tif(tmp_path / "00000.tif")
    scene = Scene(
        scene_id="00000", source=SceneSourceKind.CDSE,
        acquisition_time=datetime.now(timezone.utc) - timedelta(minutes=5),
        footprint=scene_footprint(), path=scene_path,
    )
    catalogue = StubCatalogue([scene])

    processed = run_live.run_forever(
        aoi_path=aoi_path, interval_seconds=0, catalogue=catalogue, settings=settings,
        state_path=tmp_path / "state.json", max_cycles=1, stub_model=True,
        tile_size=128, overlap=0, registry_path=tmp_path / "registry.sqlite3",
    )

    assert processed == 1
    result_path = settings.paths.resolve(settings.paths.processed_dir) / "00000" / "pipeline_result.json"
    assert result_path.is_file()
    on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    assert on_disk["spills"]


def test_run_live_persists_last_checked_between_cycles(
    tmp_path: Path, aoi_path: Path, settings,
) -> None:
    catalogue = StubCatalogue([])  # nothing new; only the state file matters here
    state_path = tmp_path / "state.json"

    run_live.run_forever(
        aoi_path=aoi_path, interval_seconds=0, catalogue=catalogue, settings=settings,
        state_path=state_path, max_cycles=2, tile_size=128, overlap=0,
        registry_path=tmp_path / "registry.sqlite3",
    )

    assert state_path.is_file()
    assert len(catalogue.search_calls) == 2
    # the second cycle's window starts where the first cycle's ended.
    assert catalogue.search_calls[1][0] == catalogue.search_calls[0][1]


def test_run_live_continues_after_one_scenes_pipeline_fails(
    tmp_path: Path, aoi_path: Path, settings,
) -> None:
    """A broken scene (unreadable path here) must not take down the whole
    poll loop - the next poll cycle must still run."""
    broken = Scene(
        scene_id="broken", source=SceneSourceKind.CDSE,
        acquisition_time=datetime.now(timezone.utc) - timedelta(minutes=5),
        footprint=scene_footprint(), path=Path(tmp_path / "does_not_exist.tif"),
    )
    catalogue = StubCatalogue([broken])

    processed = run_live.run_forever(
        aoi_path=aoi_path, interval_seconds=0, catalogue=catalogue, settings=settings,
        state_path=tmp_path / "state.json", max_cycles=1, stub_model=True,
        tile_size=128, overlap=0, registry_path=tmp_path / "registry.sqlite3",
    )
    assert processed == 0  # failed, but did not raise


def test_run_pipeline_against_local_data_raw_is_unaffected(tmp_path: Path, settings) -> None:
    """The demo path this step must not disturb: scripts/run_pipeline.py
    against data/raw/ takes no dependency on run_live/poller at all."""
    import scripts.run_pipeline as run_pipeline_module

    assert "poller" not in run_pipeline_module.__dict__
    assert "run_live" not in run_pipeline_module.__dict__
