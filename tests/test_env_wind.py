"""Wind lookup tests.

Runs entirely against a small synthetic NetCDF fixture shaped like ERA5's own
u10/v10/latitude/longitude/time layout (descending latitude included, since
that's how ERA5 actually ships it) - no network, no CDS credentials needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.config import load_settings
from src.env_data.wind import EnvDataError, EnvVector, _direction_from_deg, get_wind

LATS = np.array([21.0, 20.0, 19.0])  # descending, matching real ERA5
LONS = np.array([69.0, 70.0, 71.0])
TIMES = np.array(["2023-05-13T00:00:00", "2023-05-13T01:00:00"], dtype="datetime64[ns]")


def write_wind_fixture(path: Path) -> Path:
    """u10 varies only with longitude, v10 only with latitude, and both step
    by +1 m/s at the second timestamp - simple enough that every lookup's
    expected value can be computed by hand in the assertions below.
    """
    u10 = np.zeros((2, 3, 3), dtype="float32")
    v10 = np.zeros((2, 3, 3), dtype="float32")
    for t in range(2):
        for i, lat in enumerate(LATS):
            for j, lon in enumerate(LONS):
                u10[t, i, j] = (lon - 70.0) * 2.0 + t
                v10[t, i, j] = (lat - 20.0) * 3.0 + t

    ds = xr.Dataset(
        {
            "u10": (["time", "latitude", "longitude"], u10),
            "v10": (["time", "latitude", "longitude"], v10),
        },
        coords={"time": TIMES, "latitude": LATS, "longitude": LONS},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def wind_fixture(tmp_path: Path) -> Path:
    return write_wind_fixture(tmp_path / "sample_wind.nc")


T0 = datetime(2023, 5, 13, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2023, 5, 13, 1, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# the core acceptance case: a grid node matches that node's own value
# --------------------------------------------------------------------------- #
def test_get_wind_at_a_grid_node_matches_that_nodes_value(wind_fixture: Path) -> None:
    wind = get_wind(20.0, 70.0, T0, dataset=wind_fixture)
    assert wind.u == pytest.approx(0.0)
    assert wind.v == pytest.approx(0.0)
    assert wind.speed_ms == pytest.approx(0.0)


def test_get_wind_at_a_non_zero_grid_node(wind_fixture: Path) -> None:
    """lat=21 (north edge), lon=71 (east edge): u=(71-70)*2=2, v=(21-20)*3=3."""
    wind = get_wind(21.0, 71.0, T0, dataset=wind_fixture)
    assert wind.u == pytest.approx(2.0)
    assert wind.v == pytest.approx(3.0)
    assert wind.speed_ms == pytest.approx((2.0**2 + 3.0**2) ** 0.5)


# --------------------------------------------------------------------------- #
# spatial interpolation
# --------------------------------------------------------------------------- #
def test_get_wind_interpolates_linearly_between_nodes(wind_fixture: Path) -> None:
    wind = get_wind(20.5, 70.5, T0, dataset=wind_fixture)
    assert wind.u == pytest.approx(1.0)  # (70.5-70)*2
    assert wind.v == pytest.approx(1.5)  # (20.5-20)*3


def test_get_wind_nearest_method_snaps_to_the_closer_node(wind_fixture: Path) -> None:
    wind = get_wind(20.5, 70.5, T0, dataset=wind_fixture, method="nearest")
    # nearest of (19,20,21) to 20.5 is 20 or 21 depending on tie-break; either
    # way the result must be one of the exact grid values, not an average.
    assert wind.u in (0.0, 2.0)
    assert wind.v in (0.0, 3.0)


def test_get_wind_rejects_a_latitude_outside_the_grid(wind_fixture: Path) -> None:
    with pytest.raises(EnvDataError, match="outside the wind dataset's range"):
        get_wind(50.0, 70.0, T0, dataset=wind_fixture)


def test_get_wind_rejects_a_longitude_outside_the_grid(wind_fixture: Path) -> None:
    with pytest.raises(EnvDataError, match="outside the wind dataset's range"):
        get_wind(20.0, 170.0, T0, dataset=wind_fixture)


# --------------------------------------------------------------------------- #
# time lookup - nearest neighbour, not interpolated
# --------------------------------------------------------------------------- #
def test_get_wind_snaps_to_the_nearest_hour(wind_fixture: Path) -> None:
    just_after_t0 = datetime(2023, 5, 13, 0, 10, 0, tzinfo=timezone.utc)
    wind = get_wind(20.0, 70.0, just_after_t0, dataset=wind_fixture)
    assert wind.time == T0
    assert wind.u == pytest.approx(0.0)  # t0's value, not blended with t1

    just_before_t1 = datetime(2023, 5, 13, 0, 50, 0, tzinfo=timezone.utc)
    wind = get_wind(20.0, 70.0, just_before_t1, dataset=wind_fixture)
    assert wind.time == T1
    assert wind.u == pytest.approx(1.0)  # t1's step, not t0's


def test_get_wind_accepts_a_naive_datetime_as_utc(wind_fixture: Path) -> None:
    naive = datetime(2023, 5, 13, 0, 0, 0)
    aware = get_wind(20.0, 70.0, T0, dataset=wind_fixture)
    naive_result = get_wind(20.0, 70.0, naive, dataset=wind_fixture)
    assert naive_result.u == pytest.approx(aware.u)
    assert naive_result.time == T0


# --------------------------------------------------------------------------- #
# direction convention: meteorological "blowing from", not "toward"
# --------------------------------------------------------------------------- #
def test_direction_from_a_pure_northerly() -> None:
    """Air moving due south (u=0, v<0) is reported as a wind FROM the north."""
    assert _direction_from_deg(0.0, -5.0) == pytest.approx(0.0)


def test_direction_from_a_pure_easterly() -> None:
    """Air moving due west (u<0, v=0) is reported as a wind FROM the east."""
    assert _direction_from_deg(-5.0, 0.0) == pytest.approx(90.0)


def test_direction_from_a_pure_southerly() -> None:
    assert _direction_from_deg(0.0, 5.0) == pytest.approx(180.0)


def test_direction_from_a_pure_westerly() -> None:
    assert _direction_from_deg(5.0, 0.0) == pytest.approx(270.0)


def test_direction_is_undefined_but_finite_for_zero_wind() -> None:
    assert _direction_from_deg(0.0, 0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# dataset source resolution
# --------------------------------------------------------------------------- #
def test_get_wind_accepts_an_already_open_dataset(wind_fixture: Path) -> None:
    with xr.open_dataset(wind_fixture) as ds:
        wind = get_wind(20.0, 70.0, T0, dataset=ds)
    assert isinstance(wind, EnvVector)
    assert wind.u == pytest.approx(0.0)


def test_get_wind_uses_the_configured_dataset_path_when_omitted(wind_fixture: Path) -> None:
    settings = load_settings()
    settings = settings.model_copy(
        update={"env_data": settings.env_data.model_copy(
            update={"wind_dataset_path": wind_fixture}
        )}
    )
    wind = get_wind(20.0, 70.0, T0, settings=settings)
    assert wind.u == pytest.approx(0.0)


def test_get_wind_raises_a_clear_error_with_no_source_configured() -> None:
    settings = load_settings()  # default: wind_dataset_path is None, no CDS creds
    with pytest.raises(EnvDataError, match="no wind data source available"):
        get_wind(20.0, 70.0, T0, settings=settings)


def test_get_wind_reports_a_missing_file_clearly(tmp_path: Path) -> None:
    with pytest.raises(EnvDataError, match="no wind dataset at"):
        get_wind(20.0, 70.0, T0, dataset=tmp_path / "missing.nc")
