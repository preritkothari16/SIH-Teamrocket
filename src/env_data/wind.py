"""Wind vector lookups for a point in space and time.

``get_wind(lat, lon, time)`` reads a gridded wind field (10 m eastward/
northward components, ``u10``/``v10`` - ECMWF ERA5's own variable names) and
returns the wind at that exact point via nearest-neighbour time lookup and
spatial interpolation.

Data source used in this environment
-------------------------------------
Real wind comes from ECMWF's ERA5 reanalysis via the Copernicus Climate Data
Store (CDS) API - see ``credentials.cds_api_key_env`` in ``config.yaml``.
**This environment has neither the ``cdsapi`` package installed nor a
``CDS_API_KEY``/``~/.cdsapirc`` configured**, so ``get_wind()`` here is built
and tested against a small local sample NetCDF fixture that carries the exact
same ``u10``/``v10``/``latitude``/``longitude``/``time`` layout ERA5 ships.
The lookup logic (:func:`_lookup`) does not know or care which of the two
produced the file it is reading - swap in a real download by pointing
``env_data.wind_dataset_path`` at one, or by configuring CDS credentials so
:func:`fetch_era5_wind` can fetch one; nothing in the lookup itself changes.
:func:`fetch_era5_wind` is written against the CDS API's documented request
shape but is **unverified in this environment** - there is nothing here to
run it against.

Currents plug in later
-----------------------
Step 4.1 adds ocean currents (Copernicus Marine Service / HYCOM) with the
same interface: ``get_currents(lat, lon, time) -> EnvVector``, reusing
:class:`EnvVector` and the same nearest-time / spatial-interpolation lookup
shape as :func:`get_wind`. Not built yet - this is a marker, not a stub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import xarray as xr

from src.config import Settings, get_settings

#: ERA5's own 10 m wind component variable names.
U_VAR = "u10"
V_VAR = "v10"

DatasetLike = Union[Path, str, xr.Dataset]


class EnvDataError(RuntimeError):
    """A wind (or, later, current) lookup could not be completed."""


@dataclass
class EnvVector:
    """One environmental vector-field sample: wind now, currents later.

    ``direction_from_deg`` is the meteorological convention - the compass
    bearing (clockwise from north) the vector is blowing/flowing *from*, not
    the direction of travel - since that is what alerting and drift will
    both want to reason about ("wind out of the NW").
    """

    speed_ms: float
    direction_from_deg: float
    u: float
    v: float
    time: datetime  # the grid time actually used (nearest match), UTC


def _as_utc_naive(value: datetime) -> "datetime":
    """NetCDF time coordinates are naive (no tz); ERA5's own convention is
    UTC, so an aware input is converted to UTC and stripped, and a naive one
    is trusted to already be UTC - matching this project's convention
    everywhere else (see Scene's own timestamp handling)."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None)


def _direction_from_deg(u: float, v: float) -> float:
    """Meteorological wind direction: compass bearing it blows *from*.

    ``u``/``v`` point in the direction the air is moving *toward*; the
    standard conversion is ``(180 + atan2(u, v)) mod 360`` - e.g. a pure
    northerly (air moving south, u=0, v<0) gives 0 deg ("from the north").
    """
    if u == 0.0 and v == 0.0:
        return 0.0
    return (180.0 + math.degrees(math.atan2(u, v))) % 360.0


def _open(dataset: DatasetLike) -> xr.Dataset:
    if isinstance(dataset, xr.Dataset):
        return dataset
    path = Path(dataset)
    if not path.is_file():
        raise EnvDataError(f"no wind dataset at {path}")
    return xr.open_dataset(path)


def _lookup(
    ds: xr.Dataset, lat: float, lon: float, time: datetime, method: str = "linear",
) -> EnvVector:
    """The interface real ERA5 and the test fixture both satisfy: a Dataset
    with ``u10``/``v10`` over ``latitude``/``longitude``/``time``.
    """
    for dim in ("latitude", "longitude", "time"):
        if dim not in ds.dims and dim not in ds.coords:
            raise EnvDataError(f"wind dataset is missing the {dim!r} coordinate")

    at_time = ds.sel(time=_as_utc_naive(time), method="nearest")
    matched_time = at_time["time"].values

    lat_bounds = (float(ds["latitude"].min()), float(ds["latitude"].max()))
    lon_bounds = (float(ds["longitude"].min()), float(ds["longitude"].max()))
    if not (lat_bounds[0] <= lat <= lat_bounds[1]):
        raise EnvDataError(
            f"lat {lat} is outside the wind dataset's range {lat_bounds}"
        )
    if not (lon_bounds[0] <= lon <= lon_bounds[1]):
        raise EnvDataError(
            f"lon {lon} is outside the wind dataset's range {lon_bounds}"
        )

    point = at_time.interp(latitude=lat, longitude=lon, method=method)
    u = float(point[U_VAR].values)
    v = float(point[V_VAR].values)

    matched = matched_time.astype("datetime64[s]").astype(datetime).replace(
        tzinfo=timezone.utc
    )
    return EnvVector(
        speed_ms=math.hypot(u, v),
        direction_from_deg=_direction_from_deg(u, v),
        u=u,
        v=v,
        time=matched,
    )


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
        return _lookup(dataset, lat, lon, time, method=method)
    with _open(dataset) as ds:
        return _lookup(ds, lat, lon, time, method=method)


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


# --------------------------------------------------------------------------- #
# Step 4.1 marker - do not build yet
# --------------------------------------------------------------------------- #
# Ocean currents (Copernicus Marine Service / HYCOM) plug in here with the
# same shape: get_currents(lat, lon, time, dataset=None, method=None,
# settings=None) -> EnvVector, reading u/v current components instead of
# u10/v10 through the same _lookup() nearest-time/spatial-interpolation path.


__all__ = [
    "EnvDataError",
    "EnvVector",
    "get_wind",
    "fetch_era5_wind",
]
