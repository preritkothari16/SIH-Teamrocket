"""Forward drift forecast tests - constant-environment stub throughout (see
forward.py's own docstring for why a real forecast time series isn't used).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from shapely.geometry import Point, box, mapping

from src.config import load_settings
from src.drift.forward import DEFAULT_FORECAST_HOURS, forecast_drift
from src.drift.particle_model import scatter_particles
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
    return EnvVector(
        speed_ms=(u**2 + v**2) ** 0.5, direction_from_deg=0.0, u=u, v=v, time=time,
    )


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def stub_environment(monkeypatch):
    """Constant eastward wind + current, patched at forward.py's own import
    of get_environment (not env_data.service's) - that's the name the
    module actually calls."""

    def _install(wind=(5.0, 0.0), current=(0.5, 0.0)):
        def _fake_get_environment(lat, lon, time, settings=None):
            return {
                "wind": _vector(*wind, time) if wind is not None else None,
                "current": _vector(*current, time) if current is not None else None,
            }

        monkeypatch.setattr("src.drift.forward.get_environment", _fake_get_environment)

    return _install


def test_forecast_drift_returns_one_result_per_requested_hour(stub_environment) -> None:
    stub_environment()
    results = forecast_drift(_spill_feature(), rng=np.random.default_rng(0))
    assert [r.hours_elapsed for r in results] == sorted(DEFAULT_FORECAST_HOURS)


def test_forecast_drift_times_are_acquisition_time_plus_elapsed(stub_environment) -> None:
    stub_environment()
    results = forecast_drift(
        _spill_feature(), forecast_hours=(6.0, 24.0), rng=np.random.default_rng(0),
    )
    assert results[0].time == ACQUISITION_TIME + timedelta(hours=6.0)
    assert results[1].time == ACQUISITION_TIME + timedelta(hours=24.0)


def test_forecast_drift_moves_particles_east_for_eastward_wind(stub_environment) -> None:
    stub_environment(wind=(10.0, 0.0), current=(0.0, 0.0))
    results = forecast_drift(
        _spill_feature(), forecast_hours=(24.0,), rng=np.random.default_rng(0),
    )
    start_centroid_lon = SPILL_POLYGON.centroid.x
    moved = results[0].positions[:, 0]
    assert moved.mean() > start_centroid_lon


def test_forecast_drift_positions_advance_monotonically_with_horizon(stub_environment) -> None:
    """A constant eastward push must land the swarm further east at 48h than
    at 6h - later snapshots are not overwriting earlier ones."""
    stub_environment(wind=(10.0, 0.0), current=(0.0, 0.0))
    results = forecast_drift(
        _spill_feature(), forecast_hours=(6.0, 48.0), rng=np.random.default_rng(0),
    )
    assert results[0].positions[:, 0].mean() < results[1].positions[:, 0].mean()


def test_forecast_drift_polygon_is_a_convex_hull_of_the_positions(stub_environment) -> None:
    stub_environment()
    results = forecast_drift(
        _spill_feature(), forecast_hours=(6.0,), n_particles=50, rng=np.random.default_rng(0),
    )
    forecast = results[0]
    assert forecast.polygon.area > 0
    assert all(
        forecast.polygon.buffer(1e-9).contains(Point(x, y)) for x, y in forecast.positions
    )


def test_forecast_drift_supports_a_custom_dt_that_does_not_evenly_divide_the_horizon(
    stub_environment,
) -> None:
    stub_environment()
    results = forecast_drift(
        _spill_feature(), forecast_hours=(6.0,), dt_hours=4.0, rng=np.random.default_rng(0),
    )
    assert results[0].hours_elapsed == pytest.approx(6.0)


def test_forecast_drift_treats_missing_wind_and_current_as_zero(
    stub_environment, caplog, settings,
) -> None:
    """Zero wind, zero current, zero diffusion -> particles do not move at
    all, so the result must match a bare scatter with the same seed exactly."""
    stub_environment(wind=None, current=None)
    no_diffusion = settings.model_copy(
        update={"drift": settings.drift.model_copy(update={"diffusion_std_ms": 0.0})}
    )
    expected_start = scatter_particles(
        SPILL_POLYGON, rng=np.random.default_rng(0), settings=no_diffusion,
    )
    with caplog.at_level("WARNING"):
        results = forecast_drift(
            _spill_feature(), forecast_hours=(6.0,),
            rng=np.random.default_rng(0), settings=no_diffusion,
        )
    assert np.allclose(results[0].positions, expected_start)
    assert "no wind available" in caplog.text
    assert "no current available" in caplog.text


def test_forecast_drift_rejects_non_positive_forecast_hours(stub_environment) -> None:
    stub_environment()
    with pytest.raises(ValueError):
        forecast_drift(_spill_feature(), forecast_hours=(0.0, 6.0))


def test_forecast_drift_accepts_a_bare_geometry_timestamp_tuple(stub_environment) -> None:
    stub_environment()
    results = forecast_drift(
        (SPILL_POLYGON, ACQUISITION_TIME), forecast_hours=(6.0,), rng=np.random.default_rng(0),
    )
    assert len(results) == 1
