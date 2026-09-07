"""Lagrangian particle drift stepper - pure numpy, no environmental-data
coupling of its own (see module note at the bottom).

1. :func:`scatter_particles` - N points spread across a spill polygon.
2. :func:`step_particles` - one step of ``dt_hours``:
   ``velocity = wind_drift_factor * deflect(wind) + current + diffusion``.
3. **The same function hindcasts**: pass a negative ``dt_hours`` and the
   displacement direction flips cleanly (``displacement = velocity *
   dt_seconds``, and ``dt_seconds`` carries the sign) - there is no separate
   backward code path to keep in sync.
4. :func:`simulate_drift` - repeats step 2 for a run, recording a
   :class:`DriftSnapshot` every ``snapshot_interval_hours`` of elapsed time.

Positions are always ``(N, 2)`` arrays of ``[lon, lat]`` in degrees, WGS84,
matching every other geometry in this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings

#: metres per degree of latitude - a flat-earth approximation, appropriate
#: here (a single short drift step) but NOT for area, which this project
#: always computes in a proper equal-area projection instead (see
#: src/characterization/spill_object.py) - a different kind of calculation
#: at a different scale, not a relaxation of that rule.
METRES_PER_DEGREE_LAT = 111_320.0

VectorLike = Union[Tuple[float, float], np.ndarray]


@dataclass
class DriftSnapshot:
    """Particle positions at one point in a simulation.

    ``hours_elapsed`` carries the run's own sign - negative throughout a
    hindcast - so a caller can tell "6h before start" from "6h after" without
    inspecting anything else.
    """

    hours_elapsed: float
    positions: np.ndarray  # (N, 2) [lon, lat]


def scatter_particles(
    polygon: BaseGeometry,
    n_particles: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    settings: Optional[Settings] = None,
) -> np.ndarray:
    """``n_particles`` points (``[lon, lat]`` rows) uniformly scattered
    inside ``polygon``, via rejection sampling against its bounding box.

    ``n_particles`` defaults to ``drift.n_particles`` - the "configurable
    density" knob: more particles per polygon is a denser scatter.
    """
    settings = settings or get_settings()
    n_particles = n_particles if n_particles is not None else settings.drift.n_particles
    rng = rng or np.random.default_rng()

    if polygon.is_empty or polygon.area == 0:
        raise ValueError(
            "scatter_particles requires a non-empty polygon with area > 0; "
            f"got {polygon.geom_type} with area {polygon.area}"
        )

    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    accepted = np.empty((0, 2))
    # A thin/elongated slick polygon can fill only a small fraction of its
    # own bounding box, so oversample each round rather than looping one
    # candidate point at a time.
    while len(accepted) < n_particles:
        batch = rng.uniform(
            [min_lon, min_lat], [max_lon, max_lat],
            size=(max(n_particles * 2, 200), 2),
        )
        inside = shapely.contains_xy(polygon, batch[:, 0], batch[:, 1])
        accepted = np.vstack([accepted, batch[inside]])

    return accepted[:n_particles]


def _deflect(uv: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``(u, v)`` vectors clockwise by ``angle_deg`` - positive angles
    steer to the right of the vector's own direction, the Northern
    Hemisphere Ekman-deflection convention.
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    u, v = uv[..., 0], uv[..., 1]
    return np.stack([u * cos_t + v * sin_t, -u * sin_t + v * cos_t], axis=-1)


def _displacement_deg(
    u_ms: np.ndarray, v_ms: np.ndarray, dt_seconds: float, lat_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Metres-per-second velocity, over ``dt_seconds``, as a (dlon, dlat)
    displacement in degrees - longitude scaled by each particle's own
    latitude, since a degree of longitude narrows away from the equator.
    """
    dlat = (v_ms * dt_seconds) / METRES_PER_DEGREE_LAT
    dlon = (u_ms * dt_seconds) / (METRES_PER_DEGREE_LAT * np.cos(np.radians(lat_deg)))
    return dlon, dlat


def step_particles(
    positions: np.ndarray,
    dt_hours: float,
    wind_uv: VectorLike,
    current_uv: VectorLike,
    wind_drift_factor: Optional[float] = None,
    wind_deflection_deg: Optional[float] = None,
    diffusion_std_ms: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    settings: Optional[Settings] = None,
) -> np.ndarray:
    """One step of ``dt_hours`` for every particle in ``positions``.

    ``velocity = wind_drift_factor * deflect(wind_uv) + current_uv +
    diffusion``, then ``displacement = velocity * dt_seconds``.
    ``dt_hours`` negative hindcasts - the displacement direction flips
    because ``dt_seconds`` is negative, nothing else changes. The Ekman
    deflection itself does **not** flip with hindcasting: it is a fixed
    transform of the (always forward-sense) wind vector, only the final
    displacement's sign follows ``dt_hours``.

    ``wind_uv``/``current_uv`` are ``(u, v)`` m/s - either one pair applied
    to every particle, or an ``(N, 2)`` array, one pair per particle.
    Fetching real, time-varying wind/current for a moving swarm of particles
    is the forward/hindcast wrapper this step explicitly defers; this
    function only knows how to apply whatever vectors it's given.

    The diffusion term is a small, independent Gaussian velocity draw per
    particle per step (std ``diffusion_std_ms``) - a deliberately simple
    random-walk stand-in, not a formal diffusivity. It does not cancel when
    ``dt_hours`` flips sign (it is redrawn every call), which is exactly why
    a forward-then-backward round trip lands close to, not exactly on, the
    start.
    """
    settings = settings or get_settings()
    cfg = settings.drift
    wind_drift_factor = wind_drift_factor if wind_drift_factor is not None else cfg.wind_drift_factor
    wind_deflection_deg = (
        wind_deflection_deg if wind_deflection_deg is not None else cfg.wind_deflection_deg
    )
    diffusion_std_ms = diffusion_std_ms if diffusion_std_ms is not None else cfg.diffusion_std_ms
    rng = rng or np.random.default_rng()

    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    wind_uv = np.broadcast_to(np.asarray(wind_uv, dtype=float), (n, 2))
    current_uv = np.broadcast_to(np.asarray(current_uv, dtype=float), (n, 2))

    velocity = wind_drift_factor * _deflect(wind_uv, wind_deflection_deg) + current_uv
    if diffusion_std_ms > 0:
        velocity = velocity + rng.normal(0.0, diffusion_std_ms, size=(n, 2))

    dt_seconds = dt_hours * 3600.0
    dlon, dlat = _displacement_deg(velocity[:, 0], velocity[:, 1], dt_seconds, positions[:, 1])
    return np.column_stack([positions[:, 0] + dlon, positions[:, 1] + dlat])


def simulate_drift(
    positions: np.ndarray,
    n_steps: int,
    dt_hours: float,
    wind_uv: VectorLike,
    current_uv: VectorLike,
    snapshot_interval_hours: Optional[float] = None,
    wind_drift_factor: Optional[float] = None,
    wind_deflection_deg: Optional[float] = None,
    diffusion_std_ms: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    settings: Optional[Settings] = None,
) -> List[DriftSnapshot]:
    """Run :func:`step_particles` ``n_steps`` times, recording a snapshot
    every ``snapshot_interval_hours`` of elapsed (signed) time.

    ``dt_hours`` negative hindcasts the whole run (see :func:`step_particles`).
    ``wind_uv``/``current_uv`` are held constant for the run - see
    :func:`step_particles` for why. Always includes the starting positions
    (``hours_elapsed=0.0``) and a final snapshot at the run's own end, even
    if that does not land exactly on the snapshot cadence.
    """
    settings = settings or get_settings()
    cfg = settings.drift
    snapshot_interval_hours = (
        snapshot_interval_hours if snapshot_interval_hours is not None
        else cfg.snapshot_interval_hours
    )
    rng = rng or np.random.default_rng()

    positions = np.asarray(positions, dtype=float)
    snapshots = [DriftSnapshot(hours_elapsed=0.0, positions=positions.copy())]

    steps_per_snapshot = max(1, round(snapshot_interval_hours / abs(dt_hours)))
    elapsed = 0.0
    for step in range(1, n_steps + 1):
        positions = step_particles(
            positions, dt_hours, wind_uv, current_uv,
            wind_drift_factor=wind_drift_factor, wind_deflection_deg=wind_deflection_deg,
            diffusion_std_ms=diffusion_std_ms, rng=rng, settings=settings,
        )
        elapsed += dt_hours
        if step % steps_per_snapshot == 0 or step == n_steps:
            snapshots.append(DriftSnapshot(hours_elapsed=elapsed, positions=positions.copy()))

    return snapshots


__all__ = [
    "DriftSnapshot",
    "scatter_particles",
    "step_particles",
    "simulate_drift",
]
