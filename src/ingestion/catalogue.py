"""Copernicus Data Space Ecosystem (CDSE) catalogue client.

Searches the CDSE OData catalogue for Sentinel-1 GRD scenes intersecting an
area of interest within a time window, and downloads product archives.

Credentials come from the environment variables named in ``configs/config.yaml``
(see ``.env.example``); nothing is read from the YAML itself. A CDSE account is
free: https://dataspace.copernicus.eu

Example::

    from datetime import datetime, timedelta, timezone

    from src.ingestion.catalogue import CDSECatalogue
    from src.ingestion.types import load_aoi

    aoi = load_aoi("configs/aoi.geojson")
    end = datetime.now(timezone.utc)
    scenes = CDSECatalogue().search(aoi, end - timedelta(days=7), end)
    path = CDSECatalogue().fetch(scenes[0])

Every method here hits the network. Nothing else in the pipeline may import
this module directly - downstream stages take :class:`~src.ingestion.types.Scene`
objects, which :mod:`src.ingestion.local_source` produces offline.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings
from src.ingestion.types import (
    OrbitInfo,
    Scene,
    SceneSourceKind,
    as_utc,
)

logger = logging.getLogger(__name__)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"

DEFAULT_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 900
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 3
TOKEN_EXPIRY_MARGIN_S = 60


class CatalogueError(RuntimeError):
    """Any failure talking to the CDSE catalogue."""


class CatalogueAuthError(CatalogueError):
    """Missing or rejected credentials."""


def _odata_time(value: datetime) -> str:
    """Format a datetime the way the OData catalogue expects."""
    return as_utc(value).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _search_polygon_wkt(aoi: BaseGeometry, precision: int = 5) -> str:
    """WKT for the OData intersects filter.

    The filter takes a single polygon, so a MultiPolygon AOI is replaced by its
    convex hull. That over-selects rather than under-selects; results are
    re-checked against the true AOI before being returned.
    """
    geom = aoi if isinstance(aoi, Polygon) else aoi.convex_hull
    coords = ", ".join(
        f"{round(x, precision)} {round(y, precision)}"
        for x, y in geom.exterior.coords
    )
    return f"POLYGON(({coords}))"


def _attributes(product: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a product's expanded ``Attributes`` list into a plain dict."""
    flat: Dict[str, Any] = {}
    for attribute in product.get("Attributes") or []:
        name = attribute.get("Name")
        if name is not None:
            flat[name] = attribute.get("Value")
    return flat


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def product_to_scene(product: Dict[str, Any]) -> Scene:
    """Convert one OData product record into a :class:`Scene`."""
    attrs = _attributes(product)
    footprint = product.get("GeoFootprint") or product.get("Footprint")
    if footprint is None:
        raise CatalogueError(
            f"product {product.get('Name', '<unknown>')} has no GeoFootprint; "
            "the query needs GeoFootprint in its $select"
        )

    content_date = product.get("ContentDate") or {}
    start = content_date.get("Start") or attrs.get("beginningDateTime")
    end = content_date.get("End") or attrs.get("endingDateTime")
    if start is None:
        raise CatalogueError(
            f"product {product.get('Name', '<unknown>')} has no acquisition time"
        )

    # CDSE reports polarisation channels as e.g. "VV&VH".
    channels = (attrs.get("polarisationChannels") or "").replace(",", "&")
    polarisations = [p.strip() for p in channels.split("&") if p.strip()]

    product_id = product.get("Id")
    return Scene(
        scene_id=product.get("Name") or str(product_id),
        source=SceneSourceKind.CDSE,
        acquisition_time=_parse_odata_datetime(start),
        acquisition_end=_parse_odata_datetime(end) if end else None,
        footprint=footprint,
        download_url=f"{ODATA_URL}/Products({product_id})/$value" if product_id else None,
        platform=attrs.get("platformShortName"),
        product_type=attrs.get("productType"),
        sensor_mode=attrs.get("operationalMode"),
        polarisations=polarisations,
        orbit=OrbitInfo(
            direction=attrs.get("orbitDirection"),
            relative_orbit=_as_int(attrs.get("relativeOrbitNumber")),
            absolute_orbit=_as_int(attrs.get("orbitNumber")),
        ),
        size_bytes=_as_int(product.get("ContentLength")),
        extra={
            "product_id": product_id,
            "online": product.get("Online"),
            "origin_date": product.get("OriginDate"),
            "attributes": attrs,
        },
    )


def _parse_odata_datetime(value: str) -> datetime:
    """Parse the ISO timestamps OData returns, whose fractional part varies."""
    text = value.strip().replace("Z", "+00:00")
    # fromisoformat only accepts 3 or 6 fractional digits; CDSE emits others.
    match = re.match(r"^(?P<head>.*?)\.(?P<frac>\d+)(?P<tail>.*)$", text)
    if match:
        frac = match.group("frac")[:6].ljust(6, "0")
        text = f"{match.group('head')}.{frac}{match.group('tail')}"
    return as_utc(datetime.fromisoformat(text))


class CDSECatalogue:
    """Search and download Sentinel-1 products from the CDSE OData catalogue."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        settings: Optional[Settings] = None,
        session: Optional[requests.Session] = None,
        odata_url: str = ODATA_URL,
        token_url: str = TOKEN_URL,
        download_dir: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.odata_url = odata_url.rstrip("/")
        self.token_url = token_url
        self.session = session or requests.Session()

        credentials = self.settings.credentials
        self.client_id = client_id or _env(credentials.cdse_client_id_env)
        self.client_secret = client_secret or _env(credentials.cdse_client_secret_env)

        paths = self.settings.paths
        self.download_dir = Path(download_dir) if download_dir else paths.resolve(
            paths.raw_dir
        )

        self._token: Optional[str] = None
        self._token_expiry: datetime = datetime.min.replace(tzinfo=timezone.utc)

    # -- auth ------------------------------------------------------------- #
    def _access_token(self) -> str:
        """Current OAuth token, refreshed when missing or close to expiry."""
        now = datetime.now(timezone.utc)
        if self._token and now < self._token_expiry:
            return self._token

        if not self.client_id or not self.client_secret:
            creds = self.settings.credentials
            raise CatalogueAuthError(
                "CDSE credentials are not set. Populate "
                f"{creds.cdse_client_id_env} and {creds.cdse_client_secret_env} "
                "(copy .env.example to .env)."
            )

        response = self.session.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code >= 400:
            raise CatalogueAuthError(
                f"CDSE token request failed ({response.status_code}): {response.text[:300]}"
            )

        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise CatalogueAuthError("CDSE token response contained no access_token")

        lifetime = int(payload.get("expires_in", 600))
        self._token = token
        self._token_expiry = now + timedelta(
            seconds=max(lifetime - TOKEN_EXPIRY_MARGIN_S, 30)
        )
        return token

    # -- http ------------------------------------------------------------- #
    def _get(self, url: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """GET with a bearer token and retries on transient status codes."""
        last_error: Optional[str] = None
        for attempt in range(1, MAX_RETRIES + 1):
            headers = {"Authorization": f"Bearer {self._access_token()}"}
            headers.update(kwargs.pop("headers", {}) or {})
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=kwargs.pop("timeout", DEFAULT_TIMEOUT),
                **kwargs,
            )
            if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                last_error = f"{response.status_code}: {response.text[:200]}"
                backoff = 2**attempt
                logger.warning(
                    "CDSE request failed (%s), retrying in %ss", last_error, backoff
                )
                time.sleep(backoff)
                continue
            if response.status_code == 401:
                # Token may have been revoked early; drop it and retry once.
                self._token = None
                if attempt < MAX_RETRIES:
                    continue
            if response.status_code >= 400:
                raise CatalogueError(
                    f"CDSE request to {url} failed "
                    f"({response.status_code}): {response.text[:300]}"
                )
            return response
        raise CatalogueError(f"CDSE request to {url} failed after retries: {last_error}")

    # -- search ----------------------------------------------------------- #
    def build_filter(
        self,
        aoi: BaseGeometry,
        start: datetime,
        end: datetime,
        collection: Optional[str] = None,
        product_type: Optional[str] = None,
        sensor_mode: Optional[str] = None,
        polarisation: Optional[str] = None,
    ) -> str:
        """Assemble the OData ``$filter`` expression for a search."""
        cfg = self.settings.ingestion
        collection = collection or cfg.collection
        product_type = product_type if product_type is not None else cfg.product_type
        sensor_mode = sensor_mode if sensor_mode is not None else cfg.sensor_mode
        polarisation = polarisation if polarisation is not None else cfg.polarisation

        clauses = [
            f"Collection/Name eq '{collection}'",
            "OData.CSC.Intersects(area=geography'SRID=4326;"
            f"{_search_polygon_wkt(aoi)}')",
            f"ContentDate/Start ge {_odata_time(start)}",
            f"ContentDate/Start le {_odata_time(end)}",
        ]
        if product_type:
            clauses.append(_string_attribute("productType", product_type))
        if sensor_mode:
            clauses.append(_string_attribute("operationalMode", sensor_mode))
        if polarisation:
            clauses.append(_polarisation_clause(polarisation))
        return " and ".join(clauses)

    def search(
        self,
        aoi: BaseGeometry,
        start: datetime,
        end: datetime,
        limit: Optional[int] = None,
        **filters: Any,
    ) -> List[Scene]:
        """Sentinel-1 scenes intersecting ``aoi`` acquired within [start, end].

        Returns newest first. ``limit`` defaults to
        ``ingestion.max_products_per_query`` from the config.
        """
        limit = limit or self.settings.ingestion.max_products_per_query
        odata_filter = self.build_filter(aoi, start, end, **filters)

        scenes: List[Scene] = []
        for product in self._iter_products(odata_filter, limit):
            scene = product_to_scene(product)
            # The intersects filter runs against the AOI's convex hull, so
            # re-check against the real geometry.
            if scene.footprint.intersects(aoi):
                scenes.append(scene)
            if len(scenes) >= limit:
                break

        scenes.sort(key=lambda s: s.acquisition_time, reverse=True)
        logger.info("CDSE search returned %d scene(s)", len(scenes))
        return scenes

    def _iter_products(self, odata_filter: str, limit: int) -> Iterator[Dict[str, Any]]:
        """Page through OData results, following ``@odata.nextLink``."""
        url = f"{self.odata_url}/Products"
        params: Optional[Dict[str, Any]] = {
            "$filter": odata_filter,
            "$expand": "Attributes",
            "$orderby": "ContentDate/Start desc",
            "$top": min(max(limit, 1), 1000),
        }

        yielded = 0
        while url:
            payload = self._get(url, params=params).json()
            for product in payload.get("value", []):
                yield product
                yielded += 1
                if yielded >= limit:
                    return
            url = payload.get("@odata.nextLink") or ""
            params = None  # nextLink already carries the query string

    def get_product(self, product_id: str) -> Scene:
        """Fetch a single product record by its CDSE UUID."""
        payload = self._get(
            f"{self.odata_url}/Products({product_id})", params={"$expand": "Attributes"}
        ).json()
        return product_to_scene(payload)

    # -- download --------------------------------------------------------- #
    def fetch(
        self,
        scene: Scene,
        dest_dir: Optional[Path] = None,
        overwrite: bool = False,
    ) -> Path:
        """Download ``scene``'s product archive and return its local path.

        Satisfies the :class:`~src.ingestion.types.SceneSource` protocol. Already
        downloaded scenes are returned untouched unless ``overwrite`` is set.
        """
        if scene.is_local and not overwrite:
            return Path(scene.path)  # type: ignore[arg-type]
        if not scene.download_url:
            raise CatalogueError(f"scene {scene.scene_id} has no download URL")

        target_dir = Path(dest_dir) if dest_dir else self.download_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{scene.scene_id}.zip"

        if target.exists() and not overwrite:
            logger.info("scene %s already downloaded: %s", scene.scene_id, target)
            scene.path = target
            return target

        partial = target.with_suffix(target.suffix + ".part")
        logger.info("downloading %s -> %s", scene.scene_id, target)
        with self._get(
            scene.download_url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True,
        ) as response, partial.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if chunk:
                    fh.write(chunk)

        partial.replace(target)
        scene.path = target
        scene.size_bytes = target.stat().st_size
        return target

    # Kept as a readable alias; ``fetch`` is the protocol method.
    download = fetch


def _string_attribute(name: str, value: str) -> str:
    """OData clause matching a product string attribute."""
    return (
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
        f"'{name}' and att/OData.CSC.StringAttribute/Value eq '{value}')"
    )


def _polarisation_clause(polarisation: str) -> str:
    """Match products whose polarisation channels contain ``polarisation``."""
    return (
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq "
        "'polarisationChannels' and contains(att/OData.CSC.StringAttribute/Value, "
        f"'{polarisation}'))"
    )


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value or None


__all__ = [
    "CDSECatalogue",
    "CatalogueError",
    "CatalogueAuthError",
    "ODATA_URL",
    "TOKEN_URL",
    "product_to_scene",
]
