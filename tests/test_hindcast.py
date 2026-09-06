"""Hindcast corridor tests - constant-environment stub (see
src/drift/forward.py's docstring for why), same pattern as
test_drift_forward.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from shapely.geometry import box, mapping

from src.config import load_settings
from src.drift.hindcast import hindcast_origin, nearest_snapshot
from src.env_data.grid import EnvVector

SPILL_POLYGON = box(70.0, 22.0, 70.05, 22.03)
ACQUISITION_TIME = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


def _spill_feature() -> dict:
    return {
        "type": "Feature",
        "geometry": mapping(SPILL_POLYGON),
        "properties": {"acquisition_timestamp": ACQUISITION_TIME.isoformat()},
    }


def _vector(u: float, v: float, time: datetime) -> EnvVector:
    return EnvVector(speed_ms=(u**2 + v**2) ** 0.5, direction_from_deg=0.0, u=u, v=v, time=time)


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def stub_environment(monkeypatch):
    def _install(wind=(5.0, 0.0), current=(0.5, 0.0)):
        def _fake_get_environment(lat, lon, time, settings=None):
            return {"wind": _vector(*wind, time), "current": _vector(*current, time)}

        monkeypatch.setattr("src.drift.forward.get_environment", _fake_get_environment)

    return _install


def test_hindcast_origin_starts_at_zero_hours_with_the_spills_own_polygon(
    stub_environment,
) -> None:
    stub_environment()
    corridor = hindcast_origin(_spill_feature(), rng=np.random.default_rng(0))
    assert corridor[0].hours_before_acquisition == 0.0
    assert corridor[0].time == ACQUISITION_TIME
    assert corridor[0].polygon.equals(SPILL_POLYGON)


def test_hindcast_origin_covers_hourly_steps_back_to_the_configured_horizon(
    stub_environment, settings,
) -> None:
    stub_environment()
    corridor = hindcast_origin(_spill_feature(), rng=np.random.default_rng(0))
    hours = [s.hours_before_acquisition for s in corridor]
    assert hours == [float(h) for h in range(0, int(settings.drift.forecast_hours) + 1)]


def test_hindcast_origin_times_move_backward_from_acquisition(stub_environment) -> None:
    stub_environment()
    corridor = hindcast_origin(_spill_feature(), max_hours_back=6.0, rng=np.random.default_rng(0))
    for snapshot in corridor:
        assert snapshot.time == ACQUISITION_TIME - timedelta(hours=snapshot.hours_before_acquisition)


def test_hindcast_origin_moves_the_swarm_upwind_for_an_eastward_wind(stub_environment) -> None:
    """An eastward-blowing wind forward-drifts oil east; hindcasting from the
    observed (eastern) position must walk the swarm back west (upwind)."""
    stub_environment(wind=(10.0, 0.0), current=(0.0, 0.0))
    corridor = hindcast_origin(_spill_feature(), max_hours_back=24.0, rng=np.random.default_rng(0))
    start_lon = corridor[0].positions[:, 0].mean()
    end_lon = corridor[-1].positions[:, 0].mean()
    assert end_lon < start_lon


def test_hindcast_origin_respects_custom_step_and_horizon(stub_environment) -> None:
    stub_environment()
    corridor = hindcast_origin(
        _spill_feature(), max_hours_back=12.0, step_hours=3.0, rng=np.random.default_rng(0),
    )
    assert [s.hours_before_acquisition for s in corridor] == [0.0, 3.0, 6.0, 9.0, 12.0]


def test_nearest_snapshot_picks_the_closest_time(stub_environment) -> None:
    stub_environment()
    corridor = hindcast_origin(_spill_feature(), max_hours_back=12.0, rng=np.random.default_rng(0))
    picked = nearest_snapshot(corridor, ACQUISITION_TIME - timedelta(hours=6.4))
    assert picked.hours_before_acquisition == 6.0


def test_nearest_snapshot_accepts_a_naive_utc_time(stub_environment) -> None:
    """movingpandas tracks carry naive-UTC timestamps internally - CPA times
    downstream may arrive without tzinfo, and nearest_snapshot must not choke
    comparing them against the corridor's own aware times."""
    stub_environment()
    corridor = hindcast_origin(_spill_feature(), max_hours_back=6.0, rng=np.random.default_rng(0))
    naive_target = (ACQUISITION_TIME - timedelta(hours=3)).replace(tzinfo=None)
    picked = nearest_snapshot(corridor, naive_target)
    assert picked.hours_before_acquisition == 3.0


def test_nearest_snapshot_rejects_an_empty_corridor() -> None:
    with pytest.raises(ValueError):
        nearest_snapshot([], ACQUISITION_TIME)
