"""Wind vector lookups for a point in space and time.

``get_wind(lat, lon, time)`` reads a gridded wind field (10 m eastward/
northward components, ``u10``/``v10`` - ECMWF ERA5's own variable names) and
returns the wind at that exact point via nearest-neighbour time lookup and
spatial interpolation. The generic lookup shape (nearest-time, then spatial
interpolation) lives in :mod:`src.env_data.grid`, shared with
:mod:`src.env_data.currents` - this module only supplies ERA5's variable
names and its own fetch path.

Data source used in this environment
-------------------------------------
Real wind comes from ECMWF's ERA5 reanalysis via the Copernicus Climate Data
Store (CDS) API - see ``credentials.cds_api_key_env`` in ``config.yaml``.
**This environment has neither the ``cdsapi`` package installed nor a
``CDS_API_KEY``/``~/.cdsapirc`` configured**, so ``get_wind()`` here is built
and tested against a small local sample NetCDF fixture that carries the exact
same ``u10``/``v10``/``latitude``/``longitude``/``time`` layout ERA5 ships.
The lookup logic (:func:`src.env_data.grid.lookup_vector`) does not know or
care which of the two produced the file it is reading - swap in a real
download by pointing ``env_data.wind_dataset_path`` at one, or by configuring
CDS credentials so :func:`fetch_era5_wind` can fetch one; nothing in the
lookup itself changes. :func:`fetch_era5_wind` is written against the CDS
API's documented request shape but is **unverified in this environment** -
there is nothing here to run it against.

Currents
--------
Step 4.1 adds ocean currents (Copernicus Marine Service / HYCOM) in
:mod:`src.env_data.currents`, sharing :class:`EnvVector` and
:mod:`src.env_data.grid`'s lookup shape. :mod:`src.env_data.service` wraps
both behind one ``get_environment(lat, lon, time)`` call.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import xarray as xr

from src.config import Settings, get_settings
from src.env_data.grid import (
    DatasetLike,
    EnvDataError,
    EnvVector,
    as_utc_naive as _as_utc_naive,
    direction_from_deg as _direction_from_deg,
    lookup_vector,
    open_dataset as _open,
)

#: ERA5's own 10 m wind component variable names.
U_VAR = "u10"
V_VAR = "v10"

#: what error messages call this field - kept exactly as before the
#: grid.py extraction, so existing error-matching tests still pass.
_KIND = "wind dataset"


def get_wind(
    lat: float,
    lon: float,
    time: datetime,
    dataset: Optional[DatasetLike] = None,
    method: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> EnvVector:
    """Wind vector at ``(lat, lon, time)``.

    ``dataset`` may be a path to a NetCDF file, an already-open
    :class:`xarray.Dataset` (so a caller looking up many points can open it
    once), or omitted to use ``env_data.wind_dataset_path`` from the config.
    With neither a path nor an open dataset, falls back to fetching ERA5 via
    the CDS API if credentials are configured - see :func:`fetch_era5_wind`.

    Time uses nearest-neighbour lookup (ERA5 is hourly; blending across
    hours isn't physically meaningful the way spatial blending is).
    Space uses ``method`` (default ``env_data.interpolation_method``,
    ``"linear"`` - bilinear across the grid) or ``"nearest"``.
    """
    settings = settings or get_settings()
    cfg = settings.env_data
    method = method or cfg.interpolation_method

    if dataset is None:
        if cfg.wind_dataset_path is not None:
            dataset = settings.paths.resolve(cfg.wind_dataset_path)
        elif _cds_credentials_available(settings):
            cache_dir = settings.paths.resolve(settings.paths.env_dir)
            dataset = fetch_era5_wind(lat, lon, time, cache_dir, settings=settings)
        else:
            raise EnvDataError(
                "no wind data source available: pass dataset=, set "
                "env_data.wind_dataset_path, or configure a CDS API key "
                "(credentials.cds_api_key_env) and install cdsapi"
            )

    if isinstance(dataset, xr.Dataset):
        return lookup_vector(dataset, lat, lon, time, U_VAR, V_VAR, method=method, kind=_KIND)
    with _open(dataset, kind=_KIND) as ds:
        return lookup_vector(ds, lat, lon, time, U_VAR, V_VAR, method=method, kind=_KIND)


# --------------------------------------------------------------------------- #
# ERA5 via the Copernicus Climate Data Store (unverified in this environment
# - no cdsapi package and no CDS_API_KEY/~/.cdsapirc here; see module docstring)
# --------------------------------------------------------------------------- #
def _cds_credentials_available(settings: Settings) -> bool:
    try:
        import cdsapi  # noqa: F401
    except ImportError:
        return False
    return bool(settings.credentials.resolve().get("cds_api_key"))


def fetch_era5_wind(
    lat: float,
    lon: float,
    time: datetime,
    cache_dir: Path,
    settings: Optional[Settings] = None,
) -> Path:
    """Download hourly ERA5 10 m wind (u10, v10) for a small box around
    ``(lat, lon)`` on ``time``'s date, caching the result under ``cache_dir``.

    Requires the ``cdsapi`` package and a CDS API key (see
    ``credentials.cds_api_key_env``). Not exercised by anything in this repo
    yet - there is no CDS access in this environment to test it against.
    """
    import cdsapi

    settings = settings or get_settings()
    creds = settings.credentials.resolve(required=True)
    buffer = settings.env_data.cds_area_buffer_deg
    when = _as_utc_naive(time)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"era5_wind_{when:%Y%m%d}_{lat:.2f}_{lon:.2f}.nc"
    if target.is_file():
        return target

    client = cdsapi.Client(key=creds["cds_api_key"])
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": f"{when:%Y}",
            "month": f"{when:%m}",
            "day": f"{when:%d}",
            "time": f"{when:%H}:00",
            # CDS area order is North/West/South/East.
            "area": [lat + buffer, lon - buffer, lat - buffer, lon + buffer],
        },
        str(target),
    )
    return target


__all__ = [
    "EnvDataError",
    "EnvVector",
    "get_wind",
    "fetch_era5_wind",
]
