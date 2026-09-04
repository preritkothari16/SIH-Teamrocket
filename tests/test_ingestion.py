"""Ingestion tests.

``local_source`` is tested end to end against synthetic GeoTIFFs written by
rasterio in a tmp dir - no real Sentinel-1 product needed. ``catalogue`` is
tested against a fake requests session, so nothing here touches the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from src.ingestion import catalogue as catalogue_module
from src.ingestion.catalogue import (
    CDSECatalogue,
    CatalogueAuthError,
    CatalogueError,
    product_to_scene,
)
from src.ingestion.local_source import (
    LocalSceneSource,
    LocalSourceError,
    load_local_scenes,
)
from src.ingestion.types import (
    Scene,
    SceneSourceKind,
    filter_scenes,
    load_aoi,
    parse_sentinel1_name,
    scenes_to_geojson,
)

# A well-formed Sentinel-1 GRD product name: 2023-05-14T00:33:32Z, orbit 48456.
S1_NAME = "S1A_IW_GRDH_1SDV_20230514T003332_20230514T003357_048456_05D45D_1A2B"
S1_TIME = datetime(2023, 5, 14, 0, 33, 32, tzinfo=timezone.utc)

# Raster covers lon 70.00-70.64, lat 21.36-22.00 (offshore Gujarat).
ORIGIN_LON, ORIGIN_LAT = 70.0, 22.0
PIXEL_DEG = 0.01
RASTER_PX = 64


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def write_geotiff(
    path: Path,
    crs: str = "EPSG:4326",
    transform=None,
    size: int = RASTER_PX,
) -> Path:
    """Write a small single-band GeoTIFF standing in for a SAR scene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is None:
        transform = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG)
    data = np.linspace(0, 255, size * size, dtype=np.float64).reshape(size, size)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data.astype("uint8"), 1)
    return path


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A data/raw/ stand-in holding one synthetic Sentinel-1-named GeoTIFF."""
    root = tmp_path / "raw"
    write_geotiff(root / f"{S1_NAME}.tif")
    return root


@pytest.fixture
def source(raw_dir: Path) -> LocalSceneSource:
    return LocalSceneSource(root=raw_dir)


@pytest.fixture
def aoi_hit():
    """AOI overlapping the synthetic raster."""
    return box(70.1, 21.5, 70.3, 21.8)


@pytest.fixture
def aoi_miss():
    """AOI nowhere near the synthetic raster."""
    return box(0.0, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
def test_parse_sentinel1_name() -> None:
    meta = parse_sentinel1_name(f"{S1_NAME}.tif")
    assert meta["platform"] == "S1A"
    assert meta["sensor_mode"] == "IW"
    assert meta["product_type"] == "GRDH"
    assert meta["polarisations"] == ["VV", "VH"]
    assert meta["acquisition_time"] == S1_TIME
    assert meta["absolute_orbit"] == 48456


def test_parse_sentinel1_name_rejects_other_names() -> None:
    assert parse_sentinel1_name("some_random_raster.tif") == {}


def test_scene_normalises_naive_times_to_utc(aoi_hit) -> None:
    scene = Scene(
        scene_id="x",
        source=SceneSourceKind.LOCAL,
        acquisition_time=datetime(2023, 5, 14, 0, 33, 32),
        footprint=aoi_hit,
    )
    assert scene.acquisition_time == S1_TIME


def test_scene_accepts_geojson_footprint(aoi_hit) -> None:
    scene = Scene(
        scene_id="x",
        source=SceneSourceKind.LOCAL,
        acquisition_time=S1_TIME,
        footprint=mapping(aoi_hit),
    )
    assert scene.footprint.equals(aoi_hit)
    assert scene.bbox == pytest.approx(aoi_hit.bounds)


def test_scene_rejects_non_polygon_footprint() -> None:
    with pytest.raises(Exception):
        Scene(
            scene_id="x",
            source=SceneSourceKind.LOCAL,
            acquisition_time=S1_TIME,
            footprint={"type": "Point", "coordinates": [0, 0]},
        )


def test_scene_geojson_roundtrip(aoi_hit) -> None:
    scene = Scene(
        scene_id="x",
        source=SceneSourceKind.LOCAL,
        acquisition_time=S1_TIME,
        footprint=aoi_hit,
    )
    feature = scene.to_geojson_feature()
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["scene_id"] == "x"
    assert scenes_to_geojson([scene])["features"] == [feature]


def test_load_aoi_from_repo_config() -> None:
    from src.config import get_settings

    settings = get_settings()
    aoi = load_aoi(settings.paths.resolve(settings.paths.aoi_geojson))
    assert not aoi.is_empty
    assert aoi.geom_type in ("Polygon", "MultiPolygon")


def test_load_aoi_accepts_feature_and_geometry(aoi_hit) -> None:
    geometry = mapping(aoi_hit)
    assert load_aoi(geometry).equals(aoi_hit)
    assert load_aoi({"type": "Feature", "geometry": geometry}).equals(aoi_hit)
    assert load_aoi(aoi_hit) is aoi_hit


def test_load_aoi_unions_a_feature_collection(tmp_path: Path) -> None:
    left, right = box(0, 0, 1, 1), box(2, 0, 3, 1)
    path = tmp_path / "aoi.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": mapping(g), "properties": {}}
                    for g in (left, right)
                ],
            }
        ),
        encoding="utf-8",
    )
    merged = load_aoi(path)
    assert merged.geom_type == "MultiPolygon"
    assert merged.area == pytest.approx(left.area + right.area)


def test_filter_scenes_sorts_newest_first(aoi_hit) -> None:
    scenes = [
        Scene(
            scene_id=str(day),
            source=SceneSourceKind.LOCAL,
            acquisition_time=S1_TIME + timedelta(days=day),
            footprint=aoi_hit,
        )
        for day in (0, 2, 1)
    ]
    assert [s.scene_id for s in filter_scenes(scenes)] == ["2", "1", "0"]
    assert [s.scene_id for s in filter_scenes(scenes, limit=1)] == ["2"]


# --------------------------------------------------------------------------- #
# local_source
# --------------------------------------------------------------------------- #
def test_local_source_reads_scene_metadata(source: LocalSceneSource, raw_dir: Path) -> None:
    scenes = source.scenes()
    assert len(scenes) == 1

    scene = scenes[0]
    assert scene.scene_id == S1_NAME
    assert scene.source is SceneSourceKind.LOCAL
    assert scene.path == raw_dir / f"{S1_NAME}.tif"
    assert scene.acquisition_time == S1_TIME
    assert scene.acquisition_end == datetime(2023, 5, 14, 0, 33, 57, tzinfo=timezone.utc)
    assert scene.platform == "S1A"
    assert scene.sensor_mode == "IW"
    assert scene.product_type == "GRDH"
    assert scene.polarisations == ["VV", "VH"]
    assert scene.orbit.absolute_orbit == 48456
    assert scene.size_bytes and scene.size_bytes > 0
    assert scene.is_local


def test_local_source_footprint_matches_the_raster(source: LocalSceneSource) -> None:
    scene = source.scenes()[0]
    span = RASTER_PX * PIXEL_DEG
    assert scene.bbox == pytest.approx(
        (ORIGIN_LON, ORIGIN_LAT - span, ORIGIN_LON + span, ORIGIN_LAT), abs=1e-6
    )


def test_local_source_search_filters_by_aoi(
    source: LocalSceneSource, aoi_hit, aoi_miss
) -> None:
    assert len(source.search(aoi_hit)) == 1
    assert source.search(aoi_miss) == []


def test_local_source_search_filters_by_time(source: LocalSceneSource, aoi_hit) -> None:
    day = timedelta(days=1)
    assert len(source.search(aoi_hit, S1_TIME - day, S1_TIME + day)) == 1
    assert source.search(aoi_hit, S1_TIME + day, S1_TIME + 2 * day) == []


def test_local_source_search_respects_limit(raw_dir: Path, aoi_hit) -> None:
    write_geotiff(raw_dir / "second_scene.tif")
    source = LocalSceneSource(root=raw_dir)
    assert len(source.search(aoi_hit)) == 2
    assert len(source.search(aoi_hit, limit=1)) == 1


def test_local_source_fetch_returns_the_existing_path(source: LocalSceneSource) -> None:
    scene = source.scenes()[0]
    assert source.fetch(scene) == scene.path


def test_local_source_fetch_raises_when_the_file_is_gone(
    source: LocalSceneSource,
) -> None:
    scene = source.scenes()[0]
    Path(scene.path).unlink()
    with pytest.raises(LocalSourceError, match="missing from disk"):
        source.fetch(scene)


def test_local_source_get_by_id(source: LocalSceneSource) -> None:
    assert source.get(S1_NAME).scene_id == S1_NAME
    with pytest.raises(LocalSourceError, match="no local scene"):
        source.get("not-a-scene")


def test_local_source_falls_back_to_mtime_for_unnamed_rasters(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_geotiff(root / "plain_raster.tif")
    scene = LocalSceneSource(root=root).scenes()[0]
    assert scene.scene_id == "plain_raster"
    assert scene.acquisition_time.tzinfo is not None
    assert scene.platform is None
    assert scene.extra["metadata_source"] == "file"


def test_local_source_reprojects_a_utm_raster(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    # UTM 43N, 100 m pixels, somewhere off the Gujarat coast.
    write_geotiff(
        root / "utm_scene.tif",
        crs="EPSG:32643",
        transform=from_origin(200_000.0, 2_400_000.0, 100.0, 100.0),
    )
    scene = LocalSceneSource(root=root).scenes()[0]
    west, south, east, north = scene.bbox
    assert -180 <= west < east <= 180
    assert -90 <= south < north <= 90
    assert 60 < west < 75 and 15 < south < 30


def test_local_source_sidecar_overrides_derived_metadata(raw_dir: Path) -> None:
    override_time = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    (raw_dir / f"{S1_NAME}.json").write_text(
        json.dumps(
            {
                "scene_id": "overridden-id",
                "acquisition_time": override_time.isoformat(),
                "footprint": mapping(box(10.0, 10.0, 11.0, 11.0)),
                "orbit": {"direction": "descending", "relative_orbit": 34},
                "extra": {"note": "hand-checked"},
            }
        ),
        encoding="utf-8",
    )

    scene = LocalSceneSource(root=raw_dir).scenes()[0]
    assert scene.scene_id == "overridden-id"
    assert scene.acquisition_time == override_time
    assert scene.bbox == pytest.approx((10.0, 10.0, 11.0, 11.0))
    assert scene.orbit.direction == "DESCENDING"
    assert scene.orbit.relative_orbit == 34
    # values the sidecar did not mention still come from the product name
    assert scene.orbit.absolute_orbit == 48456
    assert scene.platform == "S1A"
    assert scene.extra["note"] == "hand-checked"


def test_local_source_sidecar_cannot_forge_source_or_path(raw_dir: Path) -> None:
    (raw_dir / f"{S1_NAME}.json").write_text(
        json.dumps({"source": "cdse", "path": "/somewhere/else.tif"}), encoding="utf-8"
    )
    scene = LocalSceneSource(root=raw_dir).scenes()[0]
    assert scene.source is SceneSourceKind.LOCAL
    assert scene.path == raw_dir / f"{S1_NAME}.tif"


def test_local_source_skips_unreadable_products(raw_dir: Path) -> None:
    (raw_dir / "broken.tif").write_bytes(b"not a geotiff")
    assert len(LocalSceneSource(root=raw_dir).scenes()) == 1

    with pytest.raises(Exception):
        LocalSceneSource(root=raw_dir, strict=True).scenes()


def test_local_source_finds_nested_products(raw_dir: Path) -> None:
    write_geotiff(raw_dir / "2023" / "05" / "nested.tif")
    assert len(LocalSceneSource(root=raw_dir).scenes()) == 2
    assert len(LocalSceneSource(root=raw_dir, recursive=False).scenes()) == 1


def test_local_source_handles_a_missing_root(tmp_path: Path) -> None:
    assert LocalSceneSource(root=tmp_path / "nope").scenes() == []


def test_load_local_scenes_helper(raw_dir: Path, aoi_hit) -> None:
    assert len(load_local_scenes(root=raw_dir, aoi=aoi_hit)) == 1


# --------------------------------------------------------------------------- #
# catalogue - all network calls are faked
# --------------------------------------------------------------------------- #
class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(
        self,
        payload: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        chunks: Optional[List[bytes]] = None,
    ) -> None:
        self._payload = payload or {}
        self.status_code = status_code
        self._chunks = chunks or []
        self.text = json.dumps(self._payload)

    def json(self) -> Dict[str, Any]:
        return self._payload

    def iter_content(self, chunk_size: int = 0):
        return iter(self._chunks)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeSession:
    """Records requests and replays queued responses."""

    def __init__(self, responses: Optional[List[FakeResponse]] = None) -> None:
        self.responses = responses or []
        self.get_calls: List[Dict[str, Any]] = []
        self.post_calls: List[Dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append({"url": url, **kwargs})
        return FakeResponse({"access_token": "fake-token", "expires_in": 600})

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def make_product(name: str = S1_NAME, product_id: str = "abc-123") -> Dict[str, Any]:
    """One OData product record, shaped like a real CDSE response."""
    return {
        "Id": product_id,
        "Name": f"{name}.SAFE",
        "ContentLength": 1_700_000_000,
        "Online": True,
        "ContentDate": {
            "Start": "2023-05-14T00:33:32.123456Z",
            "End": "2023-05-14T00:33:57.654321Z",
        },
        "GeoFootprint": mapping(box(70.0, 21.36, 70.64, 22.0)),
        "Attributes": [
            {"Name": "platformShortName", "Value": "SENTINEL-1"},
            {"Name": "productType", "Value": "IW_GRDH_1S"},
            {"Name": "operationalMode", "Value": "IW"},
            {"Name": "polarisationChannels", "Value": "VV&VH"},
            {"Name": "orbitDirection", "Value": "descending"},
            {"Name": "relativeOrbitNumber", "Value": 34},
            {"Name": "orbitNumber", "Value": 48456},
        ],
    }


@pytest.fixture
def client() -> CDSECatalogue:
    """A catalogue client with credentials injected - never reads the network."""
    return CDSECatalogue(
        client_id="id", client_secret="secret", session=FakeSession()
    )


def test_product_to_scene_maps_every_field() -> None:
    scene = product_to_scene(make_product())
    assert scene.source is SceneSourceKind.CDSE
    assert scene.scene_id == f"{S1_NAME}.SAFE"
    assert scene.acquisition_time == datetime(
        2023, 5, 14, 0, 33, 32, 123456, tzinfo=timezone.utc
    )
    assert scene.polarisations == ["VV", "VH"]
    assert scene.sensor_mode == "IW"
    assert scene.orbit.direction == "DESCENDING"
    assert scene.orbit.relative_orbit == 34
    assert scene.orbit.absolute_orbit == 48456
    assert scene.size_bytes == 1_700_000_000
    assert scene.download_url.endswith("Products(abc-123)/$value")
    assert scene.path is None and not scene.is_local


def test_product_to_scene_requires_a_footprint() -> None:
    product = make_product()
    del product["GeoFootprint"]
    with pytest.raises(CatalogueError, match="GeoFootprint"):
        product_to_scene(product)


def test_product_to_scene_parses_odd_fractional_seconds() -> None:
    product = make_product()
    product["ContentDate"]["Start"] = "2023-05-14T00:33:32.1Z"
    assert product_to_scene(product).acquisition_time == datetime(
        2023, 5, 14, 0, 33, 32, 100000, tzinfo=timezone.utc
    )


def test_build_filter_contains_the_expected_clauses(
    client: CDSECatalogue, aoi_hit
) -> None:
    end = S1_TIME
    odata_filter = client.build_filter(aoi_hit, end - timedelta(days=3), end)
    assert "Collection/Name eq 'SENTINEL-1'" in odata_filter
    assert "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((" in odata_filter
    assert "ContentDate/Start ge 2023-05-11T00:33:32.000Z" in odata_filter
    assert "ContentDate/Start le 2023-05-14T00:33:32.000Z" in odata_filter
    assert "'productType'" in odata_filter and "'GRD'" in odata_filter
    assert "'operationalMode'" in odata_filter and "'IW'" in odata_filter
    assert "polarisationChannels" in odata_filter


def test_search_parses_products_without_touching_the_network(aoi_hit) -> None:
    session = FakeSession([FakeResponse({"value": [make_product()]})])
    client = CDSECatalogue(client_id="id", client_secret="secret", session=session)

    scenes = client.search(aoi_hit, S1_TIME - timedelta(days=3), S1_TIME)

    assert len(scenes) == 1 and scenes[0].source is SceneSourceKind.CDSE
    assert session.post_calls[0]["url"] == catalogue_module.TOKEN_URL
    request = session.get_calls[0]
    assert request["url"].endswith("/Products")
    assert request["headers"]["Authorization"] == "Bearer fake-token"
    assert request["params"]["$expand"] == "Attributes"


def test_search_drops_products_outside_the_real_aoi(aoi_miss) -> None:
    session = FakeSession([FakeResponse({"value": [make_product()]})])
    client = CDSECatalogue(client_id="id", client_secret="secret", session=session)
    assert client.search(aoi_miss, S1_TIME - timedelta(days=3), S1_TIME) == []


def test_search_follows_pagination_until_the_limit(aoi_hit) -> None:
    page1 = FakeResponse(
        {
            "value": [make_product(product_id="p1")],
            "@odata.nextLink": "https://example.invalid/next",
        }
    )
    page2 = FakeResponse({"value": [make_product(product_id="p2")]})
    session = FakeSession([page1, page2])
    client = CDSECatalogue(client_id="id", client_secret="secret", session=session)

    scenes = client.search(aoi_hit, S1_TIME - timedelta(days=3), S1_TIME, limit=2)

    assert len(scenes) == 2
    assert session.get_calls[1]["url"] == "https://example.invalid/next"


def test_search_stops_at_the_limit(aoi_hit) -> None:
    session = FakeSession(
        [FakeResponse({"value": [make_product(product_id=f"p{i}") for i in range(5)]})]
    )
    client = CDSECatalogue(client_id="id", client_secret="secret", session=session)
    assert len(client.search(aoi_hit, S1_TIME - timedelta(days=3), S1_TIME, limit=2)) == 2


def test_token_is_reused_across_requests(aoi_hit) -> None:
    session = FakeSession(
        [FakeResponse({"value": []}), FakeResponse({"value": []})]
    )
    client = CDSECatalogue(client_id="id", client_secret="secret", session=session)
    window = (S1_TIME - timedelta(days=3), S1_TIME)
    client.search(aoi_hit, *window)
    client.search(aoi_hit, *window)
    assert len(session.post_calls) == 1


def test_missing_credentials_raise_before_any_request(
    aoi_hit, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    session = FakeSession()
    client = CDSECatalogue(session=session)

    with pytest.raises(CatalogueAuthError, match="CDSE_CLIENT_ID"):
        client.search(aoi_hit, S1_TIME - timedelta(days=3), S1_TIME)
    assert session.get_calls == [] and session.post_calls == []


def test_http_error_is_wrapped(aoi_hit) -> None:
    session = FakeSession([FakeResponse({"detail": "bad filter"}, status_code=400)])
    client = CDSECatalogue(client_id="id", client_secret="secret", session=session)
    with pytest.raises(CatalogueError, match="400"):
        client.search(aoi_hit, S1_TIME - timedelta(days=3), S1_TIME)


def test_fetch_streams_to_the_download_dir(tmp_path: Path) -> None:
    scene = product_to_scene(make_product())
    payload = b"fake-product-bytes"
    session = FakeSession([FakeResponse(chunks=[payload])])
    client = CDSECatalogue(
        client_id="id",
        client_secret="secret",
        session=session,
        download_dir=tmp_path,
    )

    path = client.fetch(scene)

    assert path == tmp_path / f"{scene.scene_id}.zip"
    assert path.read_bytes() == payload
    assert list(tmp_path.glob("*.part")) == []  # partial file cleaned up
    assert scene.path == path and scene.size_bytes == len(payload)


def test_fetch_skips_an_existing_download(tmp_path: Path) -> None:
    scene = product_to_scene(make_product())
    target = tmp_path / f"{scene.scene_id}.zip"
    target.write_bytes(b"already here")

    session = FakeSession()  # any GET would raise
    client = CDSECatalogue(
        client_id="id", client_secret="secret", session=session, download_dir=tmp_path
    )

    assert client.fetch(scene) == target
    assert session.get_calls == []


def test_fetch_without_a_download_url_raises(tmp_path: Path, aoi_hit) -> None:
    scene = Scene(
        scene_id="no-url",
        source=SceneSourceKind.CDSE,
        acquisition_time=S1_TIME,
        footprint=aoi_hit,
    )
    client = CDSECatalogue(
        client_id="id",
        client_secret="secret",
        session=FakeSession(),
        download_dir=tmp_path,
    )
    with pytest.raises(CatalogueError, match="no download URL"):
        client.fetch(scene)


# --------------------------------------------------------------------------- #
# the shared interface is what downstream code depends on
# --------------------------------------------------------------------------- #
def test_both_sources_expose_the_same_interface(raw_dir: Path) -> None:
    local = LocalSceneSource(root=raw_dir)
    remote = CDSECatalogue(client_id="id", client_secret="secret", session=FakeSession())
    for source in (local, remote):
        assert callable(source.search) and callable(source.fetch)


def test_both_sources_yield_indistinguishable_scenes(
    raw_dir: Path, aoi_hit
) -> None:
    """Downstream code sees one Scene type, whatever produced it."""
    session = FakeSession([FakeResponse({"value": [make_product()]})])
    remote = CDSECatalogue(client_id="id", client_secret="secret", session=session)

    local_scene = LocalSceneSource(root=raw_dir).search(aoi_hit)[0]
    remote_scene = remote.search(aoi_hit, S1_TIME - timedelta(days=3), S1_TIME)[0]

    for scene in (local_scene, remote_scene):
        assert isinstance(scene, Scene)
        assert scene.acquisition_time.tzinfo is not None
        assert scene.footprint.geom_type in ("Polygon", "MultiPolygon")
        assert scene.footprint.intersects(aoi_hit)
        assert set(scene.model_dump()) == set(Scene.model_fields)
