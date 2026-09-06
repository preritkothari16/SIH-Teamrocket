"""Forward drift forecast: where a spill's oil is likely to be in the future.

:func:`forecast_drift` wraps :mod:`src.drift.particle_model` around one
spill: scatters particles across its polygon, steps them forward with wind
and current pulled from :mod:`src.env_data.service`, and returns the particle
swarm (plus its convex-hull footprint) at each requested forecast horizon -
6h/12h/24h/48h by default.

**Known limitation, by design, not an oversight**: this environment has no
real forecast time series for wind or current (see
``src/env_data/{wind,currents}.py`` - no CDS/CMEMS access here), only a
current-instant field at best. ``forecast_drift`` therefore samples wind and
current **once**, at the spill's centroid and acquisition time, and holds
that single vector pair constant for the entire forecast horizon. A 48h
forecast under a constant wind is measurably wrong the moment the real wind
actually shifts - this is the correct, honest thing to do with only an
instantaneous field, not a bug to fix here. Swap in a real forecast time
series by resampling ``wind_uv``/``current_uv`` per step once one is
available; nothing about the stepping itself would need to change.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry

from src.ais.query import SpillLike, spill_geometry_and_time
from src.config import Settings, get_settings
from src.drift.particle_model import scatter_particles, step_particles
from src.env_data.service import get_environment

logger = logging.getLogger(__name__)

#: the four forecast horizons requested for the demo.
DEFAULT_FORECAST_HOURS: Tuple[float, ...] = (6.0, 12.0, 24.0, 48.0)


@dataclass
class DriftForecast:
    """The particle swarm - and its convex-hull footprint - at one forecast
    horizon."""

    hours_elapsed: float  # always positive: this is a forward forecast
    time: datetime  # acquisition_time + hours_elapsed, UTC
    positions: np.ndarray  # (N, 2) [lon, lat]
    polygon: BaseGeometry  # convex hull of positions - the estimated footprint


def sample_environment_vectors(
    lat: float, lon: float, time: datetime, settings: Settings,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """(wind_uv, current_uv) at one point/time, each (0, 0) - loudly logged -
    if that field is unavailable. See module docstring for why this is
    sampled once rather than per step. Shared with :mod:`src.drift.hindcast`,
    which samples the same way for the same reason."""
    environment = get_environment(lat, lon, time, settings=settings)

    wind = environment["wind"]
    if wind is None:
        logger.warning(
            "no wind available for drift forecast at (%s, %s, %s); treating as zero",
            lat, lon, time,
        )
    current = environment["current"]
    if current is None:
        logger.warning(
            "no current available for drift forecast at (%s, %s, %s); treating as zero",
            lat, lon, time,
        )

    wind_uv = (wind.u, wind.v) if wind is not None else (0.0, 0.0)
    current_uv = (current.u, current.v) if current is not None else (0.0, 0.0)
    return wind_uv, current_uv


def forecast_drift(
    spill: SpillLike,
    forecast_hours: Sequence[float] = DEFAULT_FORECAST_HOURS,
    n_particles: Optional[int] = None,
    dt_hours: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    settings: Optional[Settings] = None,
) -> List[DriftForecast]:
    """Forward drift forecast for ``spill`` at each hour in ``forecast_hours``.

    Wind and current are sampled once, at the spill's centroid and
    acquisition time, and held constant for the whole run (see module
    docstring). ``forecast_hours`` need not share a common step size with
    ``dt_hours`` (default ``drift.timestep_hours``) - the last step before
    each requested hour is shortened to land on it exactly.
    """
    settings = settings or get_settings()
    dt_hours = dt_hours if dt_hours is not None else settings.drift.timestep_hours
    rng = rng or np.random.default_rng()

    targets = sorted({float(h) for h in forecast_hours})
    if not targets or targets[0] <= 0:
        raise ValueError("forecast_hours must be one or more positive hour values")

    geometry, acquisition_time = spill_geometry_and_time(spill)
    centroid = geometry.centroid
    wind_uv, current_uv = sample_environment_vectors(centroid.y, centroid.x, acquisition_time, settings)

    positions = scatter_particles(geometry, n_particles=n_particles, rng=rng, settings=settings)

    results: List[DriftForecast] = []
    elapsed = 0.0
    target_index = 0
    while target_index < len(targets):
        step = min(dt_hours, targets[target_index] - elapsed)
        positions = step_particles(
            positions, step, wind_uv, current_uv, rng=rng, settings=settings,
        )
        elapsed += step
        if math.isclose(elapsed, targets[target_index], abs_tol=1e-6):
            results.append(
                DriftForecast(
                    hours_elapsed=elapsed,
                    time=acquisition_time + timedelta(hours=elapsed),
                    positions=positions.copy(),
                    polygon=MultiPoint(positions).convex_hull,
                )
            )
            target_index += 1

    return results


__all__ = [
    "DriftForecast",
    "DEFAULT_FORECAST_HOURS",
    "forecast_drift",
    "sample_environment_vectors",
]
