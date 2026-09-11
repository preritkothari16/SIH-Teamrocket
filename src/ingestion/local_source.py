"""Local scene reader: pre-downloaded SAR products from ``data/raw/``.

Produces exactly the same :class:`~src.ingestion.types.Scene` objects as
:mod:`src.ingestion.catalogue` and satisfies the same
:class:`~src.ingestion.types.SceneSource` protocol, so the rest of the pipeline
cannot tell - and must not care - which source it was handed. This is the source
the demo runs on, since it needs no credentials and no network.

Metadata is resolved per scene in this order, first hit wins per field:

1. a sidecar ``<name>.json`` next to the product (any Scene field, wins outright)
2. the Sentinel-1 product name (acquisition times, orbit, mode, polarisations)
3. the raster itself via rasterio (footprint from bounds + CRS) or a ``.SAFE``
   manifest
4. the file's modification time, as a last resort for acquisition time

Example::

    from src.ingestion.local_source import LocalSceneSource
    from src.ingestion.types import load_aoi

    source = LocalSceneSource()
    scenes = source.search(load_aoi("configs/aoi.geojson"), start, end)
    path = source.fetch(scenes[0])
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings
from src.ingestion.types import (
    WGS84,
    OrbitInfo,
    Scene,
    SceneSourceKind,
    filter_scenes,
    parse_sentinel1_name,
)

logger = logging.getLogger(__name__)

RASTER_SUFFIXES = (".tif", ".tiff", ".vrt", ".img", ".jp2")
ARCHIVE_SUFFIXES = (".zip",)
SAFE_SUFFIX = ".SAFE"
SIDECAR_SUFFIX = ".json"

# Corner coordinates in a Sentinel-1 manifest.safe, as "lat,lon lat,lon ...".
_GML_COORDS_RE = re.compile(
    r"<gml:coordinates>(?P<coords>[^<]+)</gml:coordinates>", re.IGNORECASE
)


class LocalSourceError(RuntimeError):
    """A local product could not be read or is missing required metadata."""


class LocalSceneSource:
    """Discover SAR scenes already sitting on disk.

    Args:
        root: directory to scan; defaults to ``paths.raw_dir`` from the config.
        recursive: descend into subdirectories.
        strict: raise instead of skipping when a product cannot be read.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        settings: Optional[Settings] = None,
        recursive: bool = True,
        strict: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        paths = self.settings.paths
        self.root = Path(root) if root is not None else paths.resolve(paths.raw_dir)
        self.recursive = recursive
        self.strict = strict

    # -- discovery -------------------------------------------------------- #
    def iter_products(self) -> Iterator[Path]:
        """Every candidate product path under ``root``, sorted for determinism.

        ``Path.glob("**/*")`` materialises the full recursive listing up
        front, so yielding a ``.SAFE`` directory does not stop it from also
        walking that directory's own contents - without the exclusion below,
        every measurement TIFF inside an already-yielded ``.SAFE`` directory
        would *also* be yielded as its own separate "product": a bare
        single-polarisation raster with no CRS/manifest, read as a bogus
        scene with a pixel-coordinate (not lat/lon) footprint. One real
        product must stay one product.
        """
        if not self.root.exists():
            logger.warning("local scene root does not exist: %s", self.root)
            return

        pattern = "**/*" if self.recursive else "*"
        safe_dirs: List[Path] = []
        for entry in sorted(self.root.glob(pattern)):
            if entry.name.startswith("."):
                continue
            if any(entry.is_relative_to(safe_dir) for safe_dir in safe_dirs):
                continue
            if entry.is_dir():
                if entry.name.upper().endswith(SAFE_SUFFIX):
                    safe_dirs.append(entry)
                    yield entry
                continue
            suffix = entry.suffix.lower()
            if suffix in RASTER_SUFFIXES or suffix in ARCHIVE_SUFFIXES:
                yield entry

    def scenes(self) -> List[Scene]:
        """Read every discoverable product into a Scene, newest first."""
        found: List[Scene] = []
        for path in self.iter_products():
            try:
                found.append(self.read_scene(path))
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
                if self.strict:
                    raise
                logger.warning("skipping unreadable product %s: %s", path, exc)
        found.sort(key=lambda s: s.acquisition_time, reverse=True)
        return found

    # -- SceneSource protocol --------------------------------------------- #
    def search(
        self,
        aoi: Optional[BaseGeometry] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Scene]:
        """Local scenes intersecting ``aoi`` acquired within [start, end].

        Every argument is optional here (unlike the network catalogue, scanning
        everything is cheap), so ``search()`` returns the whole local archive.
        """
        matched = filter_scenes(self.scenes(), aoi=aoi, start=start, end=end, limit=limit)
        logger.info("local search matched %d of the scenes under %s", len(matched), self.root)
        return matched

    def fetch(self, scene: Scene) -> Path:
        """Return the on-disk path. Local scenes are already 'downloaded'."""
        if scene.path is None:
            raise LocalSourceError(f"scene {scene.scene_id} has no local path")
        path = Path(scene.path)
        if not path.exists():
            raise LocalSourceError(f"scene {scene.scene_id} is missing from disk: {path}")
        return path

    def get(self, scene_id: str) -> Scene:
        """Look up one local scene by id."""
        for scene in self.scenes():
            if scene.scene_id == scene_id:
                return scene
        raise LocalSourceError(f"no local scene with id {scene_id!r} under {self.root}")

    # -- reading ---------------------------------------------------------- #
    def read_scene(self, path: Path) -> Scene:
        """Build a :class:`Scene` from a single product path."""
        path = Path(path)
        if not path.exists():
            raise LocalSourceError(f"product does not exist: {path}")

        sidecar = _read_sidecar(path)
        fields: Dict[str, Any] = {
            "scene_id": path.stem,
            "source": SceneSourceKind.LOCAL,
            "path": path,
        }

        name_meta = parse_sentinel1_name(path.name)
        orbit_kwargs: Dict[str, Any] = {}
        if name_meta:
            fields.update(
                {
                    "acquisition_time": name_meta["acquisition_time"],
                    "acquisition_end": name_meta["acquisition_end"],
                    "platform": name_meta["platform"],
                    "sensor_mode": name_meta["sensor_mode"],
                    "product_type": name_meta["product_type"],
                    "polarisations": name_meta["polarisations"],
                }
            )
            orbit_kwargs["absolute_orbit"] = name_meta["absolute_orbit"]

        footprint = _read_footprint(path)
        if footprint is not None:
            fields["footprint"] = footprint

        fields.setdefault("acquisition_time", _mtime(path))
        fields["orbit"] = OrbitInfo(**orbit_kwargs)
        fields["size_bytes"] = _size_bytes(path)
        fields["extra"] = {
            "product_name": path.name,
            "root": str(self.root),
            "metadata_source": "sidecar" if sidecar else ("name" if name_meta else "file"),
        }

        # The sidecar is authoritative, so it is applied last.
        if sidecar:
            orbit_override = sidecar.pop("orbit", None)
            extra_override = sidecar.pop("extra", None)
            if orbit_override:
                fields["orbit"] = OrbitInfo(**{**orbit_kwargs, **orbit_override})
            if extra_override:
                fields["extra"] = {**fields["extra"], **extra_override}
            fields.update(sidecar)

        if "footprint" not in fields:
            raise LocalSourceError(
                f"could not determine a footprint for {path}; add a sidecar "
                f"{path.stem}{SIDECAR_SUFFIX} with a GeoJSON 'footprint'"
            )

        return Scene(**fields)


# --------------------------------------------------------------------------- #
# metadata readers
# --------------------------------------------------------------------------- #
def _read_sidecar(path: Path) -> Dict[str, Any]:
    """Load ``<name>.json`` beside a product, if present."""
    sidecar = path.with_suffix(SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable sidecar %s: %s", sidecar, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("ignoring sidecar %s: expected a JSON object", sidecar)
        return {}
    # These are decided by the reader, never by the sidecar.
    for locked in ("source", "path"):
        data.pop(locked, None)
    return data


def _read_footprint(path: Path) -> Optional[BaseGeometry]:
    """Footprint in EPSG:4326, or None when it cannot be determined."""
    suffix = path.suffix.lower()
    if path.is_dir() or suffix in ARCHIVE_SUFFIXES:
        return _footprint_from_safe(path)
    if suffix in RASTER_SUFFIXES:
        return _footprint_from_raster(path)
    return None


def _footprint_from_raster(path: Path) -> Optional[BaseGeometry]:
    """Reproject a raster's bounds to lon/lat and box them."""
    try:
        with rasterio.open(path) as dataset:
            bounds = dataset.bounds
            crs = dataset.crs
    except rasterio.errors.RasterioError as exc:
        logger.warning("rasterio could not open %s: %s", path, exc)
        return None

    if crs is None:
        logger.warning("%s has no CRS; assuming %s", path, WGS84)
        return box(*bounds)

    if crs.to_string() == WGS84 or crs.to_epsg() == 4326:
        return box(*bounds)

    # densify so a curved projected edge is not cut short by its corners
    west, south, east, north = transform_bounds(crs, WGS84, *bounds, densify_pts=21)
    return box(west, south, east, north)


def _footprint_from_safe(path: Path) -> Optional[BaseGeometry]:
    """Footprint from a Sentinel-1 ``manifest.safe``, in a .SAFE dir or a .zip."""
    manifest = _read_manifest(path)
    if manifest is None:
        logger.warning("no manifest.safe found in %s", path)
        return None

    match = _GML_COORDS_RE.search(manifest)
    if match is None:
        logger.warning("no gml:coordinates in the manifest of %s", path)
        return None

    points = []
    for pair in match.group("coords").split():
        try:
            lat, lon = (float(v) for v in pair.split(","))
        except ValueError:
            logger.warning("malformed coordinate %r in %s", pair, path)
            return None
        points.append((lon, lat))

    if len(points) < 3:
        return None
    from shapely.geometry import Polygon  # local import keeps the module head light

    return Polygon(points)


def _read_manifest(path: Path) -> Optional[str]:
    """Text of ``manifest.safe`` from a .SAFE directory or a .zip archive."""
    if path.is_dir():
        manifest = path / "manifest.safe"
        return manifest.read_text(encoding="utf-8", errors="replace") if manifest.is_file() else None

    if path.suffix.lower() in ARCHIVE_SUFFIXES:
        try:
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.endswith("manifest.safe")]
                if not names:
                    return None
                return archive.read(names[0]).decode("utf-8", errors="replace")
        except (OSError, zipfile.BadZipFile) as exc:
            logger.warning("could not read archive %s: %s", path, exc)
    return None


def _mtime(path: Path) -> datetime:
    """File modification time as UTC - the fallback acquisition time."""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _size_bytes(path: Path) -> Optional[int]:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return None


def load_local_scenes(
    root: Optional[Path] = None,
    aoi: Optional[BaseGeometry] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Scene]:
    """One-shot convenience wrapper around :class:`LocalSceneSource`."""
    return LocalSceneSource(root=root).search(aoi=aoi, start=start, end=end, limit=limit)


__all__ = [
    "LocalSceneSource",
    "LocalSourceError",
    "load_local_scenes",
]
