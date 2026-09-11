"""Generate demo fixtures: NetCDF (wind/current) + AIS CSV.

Converts the static JSON values into files the pipeline can load:
  - data/env/demo_wind.nc     (ERA5-shaped u10/v10)
  - data/env/demo_current.nc  (CMEMS-shaped uo/vo)
  - data/ais/demo_vessels.csv (NOAA-style AIS pings)

Usage:
    python scripts/generate_demo_fixtures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Grid for wind/current (Gulf of Mexico demo area)
# ---------------------------------------------------------------------------
LATS = np.arange(22.0, 27.0, 0.5)   # 22.0 to 26.5
LONS = np.arange(-93.0, -87.0, 0.5) # -93.0 to -87.5
TIMES = np.array(
    ["2026-09-11T06:00:00", "2026-09-11T18:00:00"],
    dtype="datetime64[ns]",
)


def _wind_to_uv(speed_ms: float, direction_from_deg: float) -> tuple[float, float]:
    toward_rad = math.radians((direction_from_deg + 180.0) % 360.0)
    return speed_ms * math.sin(toward_rad), speed_ms * math.cos(toward_rad)


def _current_to_uv(speed_ms: float, direction_toward_deg: float) -> tuple[float, float]:
    toward_rad = math.radians(direction_toward_deg)
    return speed_ms * math.sin(toward_rad), speed_ms * math.cos(toward_rad)


def generate_wind_fixture(out_path: Path) -> Path:
    u0, v0 = _wind_to_uv(5.8, 210)   # scene_01
    u1, v1 = _wind_to_uv(9.1, 320)   # scene_02
    nt, nlat, nlon = len(TIMES), len(LATS), len(LONS)
    u10 = np.full((nt, nlat, nlon), u0, dtype="float32")
    v10 = np.full((nt, nlat, nlon), v0, dtype="float32")
    u10[1, :, :] = u1
    v10[1, :, :] = v1
    ds = xr.Dataset(
        {"u10": (["time", "latitude", "longitude"], u10),
         "v10": (["time", "latitude", "longitude"], v10)},
        coords={"time": TIMES, "latitude": LATS, "longitude": LONS},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"wrote wind fixture: {out_path}  ({nt}x{nlat}x{nlon})")
    return out_path


def generate_current_fixture(out_path: Path) -> Path:
    u0, v0 = _current_to_uv(0.42, 95)   # scene_01
    u1, v1 = _current_to_uv(0.51, 280)  # scene_02
    nt, nlat, nlon = len(TIMES), len(LATS), len(LONS)
    uo = np.full((nt, nlat, nlon), u0, dtype="float32")
    vo = np.full((nt, nlat, nlon), v0, dtype="float32")
    uo[1, :, :] = u1
    vo[1, :, :] = v1
    ds = xr.Dataset(
        {"uo": (["time", "latitude", "longitude"], uo),
         "vo": (["time", "latitude", "longitude"], vo)},
        coords={"time": TIMES, "latitude": LATS, "longitude": LONS},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"wrote current fixture: {out_path}  ({nt}x{nlat}x{nlon})")
    return out_path


# ---------------------------------------------------------------------------
# AIS CSV — synthetic vessel tracks for both demo scenes
# ---------------------------------------------------------------------------
def _make_track(
    mmsi: int,
    name: str,
    vessel_type: str,
    start_lat: float,
    start_lon: float,
    start_time: str,
    speed_knots: float,
    course_deg: float,
    n_pings: int = 8,
    interval_min: float = 15.0,
) -> list[dict]:
    """Straight-line track from a start point at fixed speed/course."""
    start = pd.Timestamp(start_time)
    dlon = math.sin(math.radians(course_deg)) * (speed_knots * 1.852) / 111.32
    dlat = math.cos(math.radians(course_deg)) * (speed_knots * 1.852) / 110.57
    rows = []
    for i in range(n_pings):
        dt_hours = i * interval_min / 60.0
        rows.append({
            "MMSI": mmsi,
            "BaseDateTime": (start + pd.Timedelta(hours=dt_hours)).isoformat(),
            "LAT": round(start_lat + dlat * dt_hours, 6),
            "LON": round(start_lon + dlon * dt_hours, 6),
            "SOG": round(speed_knots + np.random.default_rng(i).uniform(-0.3, 0.3), 1),
            "COG": round(course_deg + np.random.default_rng(i + 100).uniform(-2, 2), 1),
            "Heading": round(course_deg, 1),
            "VesselName": name,
            "VesselType": vessel_type,
        })
    return rows


def generate_ais_csv(out_path: Path) -> Path:
    """Create a NOAA-style AIS CSV with 4 synthetic vessels across 2 demo scenes."""
    rows = []

    # --- demo_scene_01 (2026-09-11 ~06:00 UTC, Gulf of Mexico ~24N, -90W) ---
    # Vessel 1: TANKER ALPHA — close match (positioned near the slick)
    rows += _make_track(
        mmsi=900000001, name="DEMO TANKER ALPHA", vessel_type="tanker",
        start_lat=19.045, start_lon=72.831, start_time="2026-09-11T04:30:00Z",
        speed_knots=11.2, course_deg=245, n_pings=10, interval_min=12.0,
    )
    # Vessel 2: CARGO BETA — nearby but less likely track
    rows += _make_track(
        mmsi=900000002, name="DEMO CARGO BETA", vessel_type="cargo",
        start_lat=19.112, start_lon=72.905, start_time="2026-09-11T04:30:00Z",
        speed_knots=14.8, course_deg=190, n_pings=10, interval_min=12.0,
    )
    # Vessel 3: FISHING GAMMA — slow, further south
    rows += _make_track(
        mmsi=900000003, name="DEMO FISHING GAMMA", vessel_type="fishing",
        start_lat=18.980, start_lon=72.760, start_time="2026-09-11T04:30:00Z",
        speed_knots=6.1, course_deg=300, n_pings=10, interval_min=12.0,
    )

    # --- demo_scene_02 (2026-09-11 ~18:00 UTC, Gulf of Mexico ~26N, -91W) ---
    # Vessel 4: TANKER DELTA — along the scene_02 slick track
    rows += _make_track(
        mmsi=900000004, name="DEMO TANKER DELTA", vessel_type="tanker",
        start_lat=20.310, start_lon=73.150, start_time="2026-09-11T16:30:00Z",
        speed_knots=12.5, course_deg=340, n_pings=10, interval_min=12.0,
    )

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote AIS fixture: {out_path}  ({len(df)} rows, {df['MMSI'].nunique()} vessels)")
    return out_path


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    generate_wind_fixture(repo / "data" / "env" / "demo_wind.nc")
    generate_current_fixture(repo / "data" / "env" / "demo_current.nc")
    generate_ais_csv(repo / "data" / "ais" / "demo_vessels.csv")
