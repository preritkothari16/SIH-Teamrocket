"""Ocean surface current lookups for a point in space and time.

Same interface pattern as :mod:`src.env_data.wind`: ``get_current(lat, lon,
time)`` reads a gridded current field (eastward/northward surface velocity
components) and returns the current at that exact point via nearest-neighbour
time lookup and spatial interpolation - the shared shape lives in
:mod:`src.env_data.grid`, this module only supplies the variable names and
its own fetch path.

Data source used in this environment
-------------------------------------
Real currents come from Copernicus Marine Service's global ocean analysis
(variables ``uo``/``vo``, eastward/northward sea water velocity) or HYCOM
(``water_u``/``water_v``) - see ``credentials.cmems_username_env`` /
``cmems_password_env`` in ``config.yaml``. **This environment has neither the
``copernicusmarine`` package installed nor CMEMS credentials configured**, so
``get_current()`` here is built and tested against a small local sample
NetCDF fixture carrying the same ``uo``/``vo``/``latitude``/``longitude``/
``time`` layout CMEMS ships. :func:`fetch_cmems_current` is written against
the Copernicus Marine Toolbox's documented request shape but is **unverified
in this environment** - there is nothing here to run it against.

Direction convention caveat
----------------------------
:class:`~src.env_data.grid.EnvVector` always reports
``direction_from_deg`` the meteorological way (compass bearing the vector
comes *from*). Oceanographers instead usually describe a current's direction
as where it flows *toward* - if you want that, it's
``(direction_from_deg + 180) % 360``. Nothing here does that flip
automatically, so the ``u``/``v`` components (unambiguous either way) are
what :mod:`src.drift.particle_model` actually consumes.
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
    open_dataset as _open,
    lookup_vector,
)

#: CMEMS global ocean analysis's own surface current variable names.
U_VAR = "uo"
V_VAR = "vo"

_KIND = "current dataset"


def get_current(
    lat: float,
    lon: float,
    time: datetime,
    dataset: Optional[DatasetLike] = None,
    method: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> EnvVector:
    """Surface current vector at ``(lat, lon, time)``.

    ``dataset`` may be a path to a NetCDF file, an already-open
    :class:`xarray.Dataset`, or omitted to use
    ``env_data.current_dataset_path`` from the config. With neither a path
    nor an open dataset, falls back to fetching via Copernicus Marine if
    credentials are configured - see :func:`fetch_cmems_current`.

    Time uses nearest-neighbour lookup; space uses ``method`` (default
    ``env_data.interpolation_method``) - identical lookup shape to
    :func:`src.env_data.wind.get_wind`.
    """
    settings = settings or get_settings()
    cfg = settings.env_data
    method = method or cfg.interpolation_method

    if dataset is None:
        if cfg.current_dataset_path is not None:
            dataset = settings.paths.resolve(cfg.current_dataset_path)
        elif _cmems_credentials_available(settings):
            cache_dir = settings.paths.resolve(settings.paths.env_dir)
            dataset = fetch_cmems_current(lat, lon, time, cache_dir, settings=settings)
        else:
            raise EnvDataError(
                "no current data source available: pass dataset=, set "
                "env_data.current_dataset_path, or configure CMEMS "
                "credentials (credentials.cmems_username_env/cmems_password_env) "
                "and install copernicusmarine"
            )

    if isinstance(dataset, xr.Dataset):
        return lookup_vector(dataset, lat, lon, time, U_VAR, V_VAR, method=method, kind=_KIND)
    with _open(dataset, kind=_KIND) as ds:
        return lookup_vector(ds, lat, lon, time, U_VAR, V_VAR, method=method, kind=_KIND)


# --------------------------------------------------------------------------- #
# Copernicus Marine Service (unverified in this environment - no
# copernicusmarine package and no CMEMS credentials here; see module docstring)
# --------------------------------------------------------------------------- #
def _cmems_credentials_available(settings: Settings) -> bool:
    try:
        import copernicusmarine  # noqa: F401
    except ImportError:
        return False
    creds = settings.credentials.resolve()
    return bool(creds.get("cmems_username") and creds.get("cmems_password"))


def fetch_cmems_current(
    lat: float,
    lon: float,
    time: datetime,
    cache_dir: Path,
    settings: Optional[Settings] = None,
) -> Path:
    """Download surface current (uo, vo) for a small box around
    ``(lat, lon)`` on ``time``'s date from Copernicus Marine Service, caching
    the result under ``cache_dir``.

    Requires the ``copernicusmarine`` package and CMEMS credentials (see
    ``credentials.cmems_username_env``/``cmems_password_env``). Not exercised
    by anything in this repo yet - there is no CMEMS access in this
    environment to test it against.
    """
    import copernicusmarine

    settings = settings or get_settings()
    creds = settings.credentials.resolve(required=True)
    buffer = settings.env_data.cmems_area_buffer_deg
    when = _as_utc_naive(time)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"cmems_current_{when:%Y%m%d}_{lat:.2f}_{lon:.2f}.nc"
    if target.is_file():
        return target

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=[U_VAR, V_VAR],
        minimum_longitude=lon - buffer, maximum_longitude=lon + buffer,
        minimum_latitude=lat - buffer, maximum_latitude=lat + buffer,
        start_datetime=when.isoformat(), end_datetime=when.isoformat(),
        output_directory=str(cache_dir), output_filename=target.name,
        username=creds["cmems_username"], password=creds["cmems_password"],
    )
    return target


__all__ = ["EnvDataError", "EnvVector", "get_current", "fetch_cmems_current"]
