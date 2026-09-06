"""Ocean current lookup tests.

Runs entirely against a small synthetic NetCDF fixture shaped like CMEMS's
own uo/vo/latitude/longitude/time layout - no network, no CMEMS credentials
needed. Mirrors tests/test_env_wind.py's structure, since get_current() and
get_wind() share the same lookup shape (src/env_data/grid.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.config import load_settings
from src.env_data.currents import EnvDataError, EnvVector, get_current

LATS = np.array([21.0, 20.0, 19.0])  # descending, matching real CMEMS
LONS = np.array([69.0, 70.0, 71.0])
TIMES = np.array(["2023-05-13T00:00:00", "2023-05-13T01:00:00"], dtype="datetime64[ns]")


def write_current_fixture(path: Path) -> Path:
    """uo varies only with longitude, vo only with latitude, and both step
    by +0.1 m/s at the second timestamp - simple enough that every lookup's
    expected value can be computed by hand in the assertions below.
    """
    uo = np.zeros((2, 3, 3), dtype="float32")
    vo = np.zeros((2, 3, 3), dtype="float32")
    for t in range(2):
        for i, lat in enumerate(LATS):
            for j, lon in enumerate(LONS):
                uo[t, i, j] = (lon - 70.0) * 0.5 + t * 0.1
                vo[t, i, j] = (lat - 20.0) * 0.3 + t * 0.1

    ds = xr.Dataset(
        {
            "uo": (["time", "latitude", "longitude"], uo),
            "vo": (["time", "latitude", "longitude"], vo),
        },
        coords={"time": TIMES, "latitude": LATS, "longitude": LONS},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def current_fixture(tmp_path: Path) -> Path:
    return write_current_fixture(tmp_path / "sample_current.nc")


T0 = datetime(2023, 5, 13, 0, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2023, 5, 13, 1, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# the core acceptance case: a grid node matches that node's own value
# --------------------------------------------------------------------------- #
def test_get_current_at_a_grid_node_matches_that_nodes_value(current_fixture: Path) -> None:
    current = get_current(20.0, 70.0, T0, dataset=current_fixture)
    assert current.u == pytest.approx(0.0)
    assert current.v == pytest.approx(0.0)
    assert current.speed_ms == pytest.approx(0.0)


def test_get_current_at_a_non_zero_grid_node(current_fixture: Path) -> None:
    """lat=21 (north edge), lon=71 (east edge): u=(71-70)*0.5=0.5, v=(21-20)*0.3=0.3."""
    current = get_current(21.0, 71.0, T0, dataset=current_fixture)
    assert current.u == pytest.approx(0.5)
    assert current.v == pytest.approx(0.3)
    assert current.speed_ms == pytest.approx((0.5**2 + 0.3**2) ** 0.5)


# --------------------------------------------------------------------------- #
# spatial interpolation
# --------------------------------------------------------------------------- #
def test_get_current_interpolates_linearly_between_nodes(current_fixture: Path) -> None:
    current = get_current(20.5, 70.5, T0, dataset=current_fixture)
    assert current.u == pytest.approx(0.25)  # (70.5-70)*0.5
    assert current.v == pytest.approx(0.15)  # (20.5-20)*0.3


def test_get_current_nearest_method_snaps_to_the_closer_node(current_fixture: Path) -> None:
    current = get_current(20.5, 70.5, T0, dataset=current_fixture, method="nearest")
    assert current.u in (0.0, 0.5)
    assert current.v in (0.0, 0.3)


def test_get_current_rejects_a_latitude_outside_the_grid(current_fixture: Path) -> None:
    with pytest.raises(EnvDataError, match="outside the current dataset's range"):
        get_current(50.0, 70.0, T0, dataset=current_fixture)


def test_get_current_rejects_a_longitude_outside_the_grid(current_fixture: Path) -> None:
    with pytest.raises(EnvDataError, match="outside the current dataset's range"):
        get_current(20.0, 170.0, T0, dataset=current_fixture)


# --------------------------------------------------------------------------- #
# time lookup - nearest neighbour, not interpolated
# --------------------------------------------------------------------------- #
def test_get_current_snaps_to_the_nearest_hour(current_fixture: Path) -> None:
    just_after_t0 = datetime(2023, 5, 13, 0, 10, 0, tzinfo=timezone.utc)
    current = get_current(20.0, 70.0, just_after_t0, dataset=current_fixture)
    assert current.time == T0
    assert current.u == pytest.approx(0.0)

    just_before_t1 = datetime(2023, 5, 13, 0, 50, 0, tzinfo=timezone.utc)
    current = get_current(20.0, 70.0, just_before_t1, dataset=current_fixture)
    assert current.time == T1
    assert current.u == pytest.approx(0.1)


def test_get_current_accepts_a_naive_datetime_as_utc(current_fixture: Path) -> None:
    naive = datetime(2023, 5, 13, 0, 0, 0)
    aware = get_current(20.0, 70.0, T0, dataset=current_fixture)
    naive_result = get_current(20.0, 70.0, naive, dataset=current_fixture)
    assert naive_result.u == pytest.approx(aware.u)
    assert naive_result.time == T0


# --------------------------------------------------------------------------- #
# dataset source resolution
# --------------------------------------------------------------------------- #
def test_get_current_accepts_an_already_open_dataset(current_fixture: Path) -> None:
    with xr.open_dataset(current_fixture) as ds:
        current = get_current(20.0, 70.0, T0, dataset=ds)
    assert isinstance(current, EnvVector)
    assert current.u == pytest.approx(0.0)


def test_get_current_uses_the_configured_dataset_path_when_omitted(current_fixture: Path) -> None:
    settings = load_settings()
    settings = settings.model_copy(
        update={"env_data": settings.env_data.model_copy(
            update={"current_dataset_path": current_fixture}
        )}
    )
    current = get_current(20.0, 70.0, T0, settings=settings)
    assert current.u == pytest.approx(0.0)


def test_get_current_raises_a_clear_error_with_no_source_configured() -> None:
    settings = load_settings()  # default: current_dataset_path is None, no CMEMS creds
    with pytest.raises(EnvDataError, match="no current data source available"):
        get_current(20.0, 70.0, T0, settings=settings)


def test_get_current_reports_a_missing_file_clearly(tmp_path: Path) -> None:
    with pytest.raises(EnvDataError, match="no current dataset at"):
        get_current(20.0, 70.0, T0, dataset=tmp_path / "missing.nc")
