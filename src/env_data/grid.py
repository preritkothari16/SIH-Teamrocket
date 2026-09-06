"""Shared machinery for reading one gridded vector field at a point in space
and time - nearest-neighbour in time, spatial interpolation in space.

Both :mod:`src.env_data.wind` (``u10``/``v10``) and
:mod:`src.env_data.currents` (``uo``/``vo``) are the same shape of problem -
a NetCDF with a ``time``/``latitude``/``longitude`` grid and two velocity
components - and differ only in which variables they read and which
real-world API would fetch the file. This module is that shared shape;
neither of the two source-specific modules re-implements it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import xarray as xr

DatasetLike = Union[Path, str, xr.Dataset]


class EnvDataError(RuntimeError):
    """A gridded vector-field lookup could not be completed."""


@dataclass
class EnvVector:
    """One environmental vector-field sample: wind or ocean current.

    ``direction_from_deg`` is always computed the same way (see
    :func:`direction_from_deg`) - the *meteorological* convention, the
    compass bearing the vector points/blows *from*. That is exactly the
    number wind wants. Oceanographic convention instead reports a current's
    direction as where it flows *toward* - if you need that, it is
    ``(direction_from_deg + 180) % 360``; this field is deliberately not
    pre-flipped, so the same formula and the same field name mean the same
    thing for every source that produces an ``EnvVector``.
    """

    speed_ms: float
    direction_from_deg: float
    u: float
    v: float
    time: datetime  # the grid time actually used (nearest match), UTC


def as_utc_naive(value: datetime) -> datetime:
    """NetCDF time coordinates are naive (no tz); ERA5/CMEMS/HYCOM all use
    UTC, so an aware input is converted to UTC and stripped, and a naive one
    is trusted to already be UTC - matching this project's convention
    everywhere else (see Scene's own timestamp handling)."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc)
    return value.replace(tzinfo=None)


def direction_from_deg(u: float, v: float) -> float:
    """Meteorological convention: compass bearing (deg clockwise from north)
    the vector points/blows *from*. See :class:`EnvVector` for the ocean-
    current caveat.

    ``u``/``v`` point in the direction the vector moves *toward*; the
    standard conversion is ``(180 + atan2(u, v)) mod 360`` - e.g. a pure
    northerly (moving south, u=0, v<0) gives 0 deg ("from the north").
    """
    if u == 0.0 and v == 0.0:
        return 0.0
    return (180.0 + math.degrees(math.atan2(u, v))) % 360.0


def open_dataset(dataset: DatasetLike, kind: str = "dataset") -> xr.Dataset:
    """Open ``dataset`` if it is a path, or pass an already-open one through.
    ``kind`` names the field in error messages (e.g. "wind dataset")."""
    if isinstance(dataset, xr.Dataset):
        return dataset
    path = Path(dataset)
    if not path.is_file():
        raise EnvDataError(f"no {kind} at {path}")
    return xr.open_dataset(path)


def lookup_vector(
    ds: xr.Dataset,
    lat: float,
    lon: float,
    time: datetime,
    u_var: str,
    v_var: str,
    method: str = "linear",
    kind: str = "dataset",
) -> EnvVector:
    """Nearest-time, spatially-interpolated ``(u_var, v_var)`` sample from
    ``ds`` at ``(lat, lon, time)``. ``kind`` names the field in error
    messages (e.g. "wind dataset", "current dataset").
    """
    for dim in ("latitude", "longitude", "time"):
        if dim not in ds.dims and dim not in ds.coords:
            raise EnvDataError(f"{kind} is missing the {dim!r} coordinate")

    at_time = ds.sel(time=as_utc_naive(time), method="nearest")
    matched_time = at_time["time"].values

    lat_bounds = (float(ds["latitude"].min()), float(ds["latitude"].max()))
    lon_bounds = (float(ds["longitude"].min()), float(ds["longitude"].max()))
    if not (lat_bounds[0] <= lat <= lat_bounds[1]):
        raise EnvDataError(f"lat {lat} is outside the {kind}'s range {lat_bounds}")
    if not (lon_bounds[0] <= lon <= lon_bounds[1]):
        raise EnvDataError(f"lon {lon} is outside the {kind}'s range {lon_bounds}")

    point = at_time.interp(latitude=lat, longitude=lon, method=method)
    u = float(point[u_var].values)
    v = float(point[v_var].values)

    matched = matched_time.astype("datetime64[s]").astype(datetime).replace(
        tzinfo=timezone.utc
    )
    return EnvVector(
        speed_ms=math.hypot(u, v),
        direction_from_deg=direction_from_deg(u, v),
        u=u,
        v=v,
        time=matched,
    )


__all__ = [
    "DatasetLike",
    "EnvDataError",
    "EnvVector",
    "as_utc_naive",
    "direction_from_deg",
    "open_dataset",
    "lookup_vector",
]
