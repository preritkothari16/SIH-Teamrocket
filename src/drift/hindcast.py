"""Hindcast: run the drift stepper backward from a spill's acquisition time
to build a time-indexed "origin corridor" - where the oil most likely was at
each hour before it was observed.

This is :mod:`src.drift.forward`'s exact mirror image, not a separate
implementation: same stepper, same constant-environment sampling (see that
module's docstring for why), only the sign of ``dt`` differs - per
:mod:`src.drift.particle_model`'s own design, that is the whole hindcast.

The corridor feeds :mod:`src.attribution.scoring`'s ``use_hindcasting`` path:
instead of scoring a candidate vessel's CPA against the spill's one, final
polygon, that path compares against the corridor entry nearest the vessel's
own CPA time - the oil's estimated position back when the vessel was
actually there, not where it ended up by the time the image was taken.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence

import numpy as np
from shapely.geometry import MultiPoint
from shapely.geometry.base import BaseGeometry

from src.ais.query import SpillLike, spill_geometry_and_time
from src.config import Settings, get_settings
from src.drift.forward import sample_environment_vectors
from src.drift.particle_model import scatter_particles, step_particles
from src.env_data.grid import as_utc_naive


@dataclass
class OriginSnapshot:
    """Where the spill's oil is estimated to have been at one past time."""

    time: datetime
    hours_before_acquisition: float  # >= 0; 0 is the spill's own observed polygon
    positions: np.ndarray  # (N, 2) [lon, lat]
    polygon: BaseGeometry  # convex hull of positions (the spill's own geometry at 0h)


def hindcast_origin(
    spill: SpillLike,
    max_hours_back: Optional[float] = None,
    step_hours: Optional[float] = None,
    n_particles: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    settings: Optional[Settings] = None,
) -> List[OriginSnapshot]:
    """The origin corridor for ``spill``: an :class:`OriginSnapshot` every
    ``step_hours`` (default ``drift.timestep_hours``) back to
    ``max_hours_back`` (default ``drift.forecast_hours``) before acquisition.

    Always starts with an ``hours_before_acquisition=0.0`` entry holding the
    spill's own observed polygon (not a particle hull - the actual
    footprint), so a vessel whose CPA landed at or after acquisition time
    still has a sensible corridor entry to match against.
    """
    settings = settings or get_settings()
    cfg = settings.drift
    max_hours_back = max_hours_back if max_hours_back is not None else cfg.forecast_hours
    step_hours = step_hours if step_hours is not None else cfg.timestep_hours
    rng = rng or np.random.default_rng()

    geometry, acquisition_time = spill_geometry_and_time(spill)
    centroid = geometry.centroid
    wind_uv, current_uv = sample_environment_vectors(
        centroid.y, centroid.x, acquisition_time, settings,
    )

    positions = scatter_particles(geometry, n_particles=n_particles, rng=rng, settings=settings)
    corridor: List[OriginSnapshot] = [
        OriginSnapshot(
            time=acquisition_time, hours_before_acquisition=0.0,
            positions=positions.copy(), polygon=geometry,
        )
    ]

    n_steps = max(1, round(max_hours_back / step_hours))
    elapsed = 0.0
    for _ in range(n_steps):
        positions = step_particles(
            positions, -step_hours, wind_uv, current_uv, rng=rng, settings=settings,
        )
        elapsed += step_hours
        corridor.append(
            OriginSnapshot(
                time=acquisition_time - timedelta(hours=elapsed),
                hours_before_acquisition=elapsed,
                positions=positions.copy(),
                polygon=MultiPoint(positions).convex_hull,
            )
        )

    return corridor


def nearest_snapshot(corridor: Sequence[OriginSnapshot], time: datetime) -> OriginSnapshot:
    """The corridor entry whose own time is closest to ``time`` - the lookup
    a candidate vessel's CPA time uses against the corridor.
    """
    if not corridor:
        raise ValueError("hindcast corridor is empty")
    target = as_utc_naive(time)
    return min(corridor, key=lambda s: abs((as_utc_naive(s.time) - target).total_seconds()))


__all__ = ["OriginSnapshot", "hindcast_origin", "nearest_snapshot"]
