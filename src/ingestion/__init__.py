"""Scene ingestion.

Two interchangeable sources, one output type:

* :class:`~src.ingestion.local_source.LocalSceneSource` reads products already
  in ``data/raw/`` - no credentials, no network. This is what the demo runs.
* :class:`~src.ingestion.catalogue.CDSECatalogue` searches and downloads from
  the Copernicus Data Space Ecosystem.

Both satisfy :class:`~src.ingestion.types.SceneSource` and return
:class:`~src.ingestion.types.Scene` objects, so downstream stages depend on the
shared interface rather than on where the pixels came from.

``catalogue`` is imported lazily so that offline runs never pull in the network
client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.ingestion.local_source import LocalSceneSource, LocalSourceError, load_local_scenes
from src.ingestion.safe import SAFEError, is_safe_archive, read_safe_bands, safe_measurement_members
from src.ingestion.types import (
    OrbitInfo,
    Scene,
    SceneSource,
    SceneSourceKind,
    filter_scenes,
    load_aoi,
    parse_sentinel1_name,
    scenes_to_geojson,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from src.ingestion.catalogue import CDSECatalogue, CatalogueError

_LAZY = {"CDSECatalogue", "CatalogueError", "CatalogueAuthError"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from src.ingestion import catalogue

        return getattr(catalogue, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Scene",
    "SceneSource",
    "SceneSourceKind",
    "OrbitInfo",
    "LocalSceneSource",
    "LocalSourceError",
    "load_local_scenes",
    "load_aoi",
    "filter_scenes",
    "parse_sentinel1_name",
    "scenes_to_geojson",
    "SAFEError",
    "is_safe_archive",
    "read_safe_bands",
    "safe_measurement_members",
    "CDSECatalogue",
    "CatalogueError",
    "CatalogueAuthError",
]
