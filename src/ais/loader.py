"""Load AIS position reports into one normalized DataFrame.

AIS is published by several agencies with different column names for the same
fields (MMSI, position, speed, course...). This module is deliberately
**source-agnostic**: it does not assume Danish Maritime Authority, NOAA
MarineCadastre, or aisstream.io - it matches whichever of their column names
shows up against a small alias table and normalizes the result to one schema.

That matters here specifically because **no openly available historical AIS
archive covers Indian coastal waters** - the AOI this project otherwise
targets. The demo instead replays one real, documented historical spill using
a public archive from elsewhere (Danish Maritime Authority or NOAA
MarineCadastre, or a live aisstream.io feed if one is captured for the
window). Nothing downstream of this module may assume a particular source's
column names or units; whichever demo dataset gets used at query time is
normalized on the way in through here.

Normalized schema (one row per position report)::

    mmsi          int64             vessel identifier
    timestamp     datetime64[ns, UTC]
    lat, lon      float64           degrees, WGS84
    sog           float64           speed over ground, knots (NaN if absent)
    cog           float64           course over ground, degrees (NaN if absent)
    heading       float64           true heading, degrees (NaN if absent)
    vessel_name   object            NaN if absent
    vessel_type   object            NaN if absent

Known real-world column names this recognizes out of the box (see
``COLUMN_ALIASES``): NOAA MarineCadastre's ``MMSI, BaseDateTime, LAT, LON,
SOG, COG, Heading, VesselName, VesselType`` and the Danish Maritime
Authority's ``MMSI, # Timestamp, Latitude, Longitude, SOG, COG, Heading,
Name, Ship type``. A different source's headers can still be read by passing
an explicit ``column_map``.

aisstream.io streams JSON over a websocket rather than CSV/parquet rows, so a
capture from it needs its own small flattening step before it reaches this
module - out of scope here, but the normalized schema above is exactly what
that step would need to produce.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NORMALIZED_COLUMNS: Tuple[str, ...] = (
    "mmsi", "timestamp", "lat", "lon", "sog", "cog", "heading",
    "vessel_name", "vessel_type",
)
REQUIRED_COLUMNS: Tuple[str, ...] = ("mmsi", "timestamp", "lat", "lon")

#: target column -> candidate source names, matched case/punctuation-insensitively.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "mmsi": ("mmsi",),
    "timestamp": ("timestamp", "basedatetime", "datetime", "time", "date_time"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "long", "longitude"),
    "sog": ("sog", "speed", "speedoverground", "speed_over_ground"),
    "cog": ("cog", "course", "courseoverground", "course_over_ground"),
    "heading": ("heading", "trueheading", "true_heading"),
    "vessel_name": ("vesselname", "name", "shipname", "ship_name"),
    "vessel_type": ("vesseltype", "shiptype", "ship_type", "type"),
}


class AISLoadError(RuntimeError):
    """AIS data on disk could not be read or is missing required fields."""


def _normalize_key(name: str) -> str:
    """Fold a column name to letters-and-digits only, lower-cased.

    Makes ``"# Timestamp"``, ``"BaseDateTime"`` and ``"base_date_time"`` all
    compare equal, which is the point: every source spells these differently.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_column_map(
    columns: pd.Index, column_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """target column -> the actual source column name found in ``columns``."""
    available = {_normalize_key(c): c for c in columns}
    resolved: Dict[str, str] = {}

    if column_map:
        for target, source in column_map.items():
            if source not in columns:
                raise AISLoadError(f"column_map names {source!r}, not found in the data")
            resolved[target] = source

    for target, candidates in COLUMN_ALIASES.items():
        if target in resolved:
            continue
        for candidate in candidates:
            match = available.get(_normalize_key(candidate))
            if match is not None:
                resolved[target] = match
                break

    missing = [c for c in REQUIRED_COLUMNS if c not in resolved]
    if missing:
        raise AISLoadError(
            f"could not find required column(s) {missing} among {list(columns)}; "
            "pass column_map={{'mmsi': '<your column>', ...}} for an unrecognized source"
        )
    return resolved


def normalize_ais_frame(
    raw: pd.DataFrame, column_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Rename, coerce and clean a raw AIS DataFrame to the normalized schema.

    Rows missing any of ``mmsi``/``timestamp``/``lat``/``lon`` - a position
    report is not one without all four - are dropped rather than kept with a
    hole in them.
    """
    resolved = _resolve_column_map(raw.columns, column_map)

    frame = pd.DataFrame(index=raw.index)
    for target in NORMALIZED_COLUMNS:
        source = resolved.get(target)
        frame[target] = raw[source] if source is not None else np.nan

    frame["mmsi"] = pd.to_numeric(frame["mmsi"], errors="coerce").astype("Int64")
    # Naive timestamps are assumed UTC, matching every public AIS archive this
    # loader targets and the convention used across the rest of the pipeline.
    #
    # NOAA MarineCadastre's BaseDateTime is ISO ("2023-05-13T12:00:00"), but
    # the Danish Maritime Authority's "# Timestamp" is day-first with slashes
    # ("31/12/2019 23:59:59"). Left to guess, pandas defaults to month-first
    # and silently swaps day/month for any DMA date whose day is <= 12 - a
    # wrong-but-plausible-looking timestamp is worse than a loud one, so the
    # separator decides which convention to use rather than leaving it to
    # pandas's default guess.
    dayfirst = frame["timestamp"].astype(str).str.contains("/", regex=False).any()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", utc=True, dayfirst=dayfirst
    )
    for col in ("lat", "lon", "sog", "cog", "heading"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    dropped = before - len(frame)
    if dropped:
        logger.warning("dropped %d AIS row(s) missing mmsi/timestamp/lat/lon", dropped)

    frame["mmsi"] = frame["mmsi"].astype("int64")
    frame = frame.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return frame[list(NORMALIZED_COLUMNS)]


def load_ais_csv(path: Path, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Read one AIS CSV export (Danish Maritime Authority, NOAA MarineCadastre,
    or anything else whose headers match ``COLUMN_ALIASES`` or ``column_map``)
    into the normalized schema.
    """
    path = Path(path)
    if not path.is_file():
        raise AISLoadError(f"no AIS file at {path}")
    return normalize_ais_frame(pd.read_csv(path), column_map=column_map)


def load_ais_parquet(path: Path, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Read one AIS parquet export into the normalized schema."""
    path = Path(path)
    if not path.is_file():
        raise AISLoadError(f"no AIS file at {path}")
    return normalize_ais_frame(pd.read_parquet(path), column_map=column_map)


def load_ais(path: Path, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Read a CSV or parquet AIS export by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_ais_csv(path, column_map=column_map)
    if suffix in (".parquet", ".pq"):
        return load_ais_parquet(path, column_map=column_map)
    raise AISLoadError(f"unrecognized AIS file extension {suffix!r} for {path}")


# --------------------------------------------------------------------------- #
# synthetic data (tests - no download needed)
# --------------------------------------------------------------------------- #
def synthetic_ais_sample(
    n_vessels: int = 3,
    points_per_vessel: int = 6,
    start_lon: float = 70.0,
    start_lat: float = 20.0,
    start_time: str = "2023-05-13T00:00:00+00:00",
    step_minutes: float = 30.0,
    speed_knots: float = 10.0,
    seed: int = 0,
) -> pd.DataFrame:
    """A small, deterministic AIS-shaped DataFrame - no download, no network.

    Each vessel sails a straight line at a fixed speed and heading from a
    randomized starting point near ``(start_lon, start_lat)``, spaced
    ``step_minutes`` apart - sparse and evenly spaced, the way real AIS is
    not, which is exactly why callers that need realistic gaps (query/track
    tests) build their own tighter fixtures rather than relying on this one.

    Round-trips through :func:`normalize_ais_frame` using source-agnostic
    column names (``MMSI``, ``BaseDateTime``, ...), so it also exercises the
    same alias resolution real data goes through.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_time)
    rows = []

    for i in range(n_vessels):
        mmsi = 200_000_000 + i
        lon0 = start_lon + rng.uniform(-0.5, 0.5)
        lat0 = start_lat + rng.uniform(-0.5, 0.5)
        heading = rng.uniform(0, 360)
        dlon = np.sin(np.radians(heading)) * (speed_knots * 1.852) / 111.32
        dlat = np.cos(np.radians(heading)) * (speed_knots * 1.852) / 110.57

        for j in range(points_per_vessel):
            hours = j * step_minutes / 60.0
            rows.append(
                {
                    "MMSI": mmsi,
                    "BaseDateTime": start + pd.Timedelta(hours=hours),
                    "LAT": lat0 + dlat * hours,
                    "LON": lon0 + dlon * hours,
                    "SOG": speed_knots,
                    "COG": heading,
                    "Heading": heading,
                    "VesselName": f"SYNTH_{mmsi}",
                    "VesselType": "Cargo",
                }
            )

    return normalize_ais_frame(pd.DataFrame(rows))


__all__ = [
    "NORMALIZED_COLUMNS",
    "REQUIRED_COLUMNS",
    "COLUMN_ALIASES",
    "AISLoadError",
    "normalize_ais_frame",
    "load_ais_csv",
    "load_ais_parquet",
    "load_ais",
    "synthetic_ais_sample",
]
