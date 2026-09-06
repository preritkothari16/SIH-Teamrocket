"""Particle drift stepper tests.

No environmental-data coupling here by design (see particle_model.py's own
module docstring) - wind/current are plain (u, v) m/s tuples throughout.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, box

from src.config import load_settings
from src.drift.particle_model import (
    DriftSnapshot,
    _deflect,
    scatter_particles,
    simulate_drift,
    step_particles,
)

SPILL_POLYGON = box(70.0, 22.0, 70.05, 22.03)  # ~5km x 3km near 22N


@pytest.fixture
def settings():
    return load_settings()


# --------------------------------------------------------------------------- #
# scatter_particles
# --------------------------------------------------------------------------- #
def test_scatter_particles_returns_the_requested_count() -> None:
    points = scatter_particles(SPILL_POLYGON, n_particles=200, rng=np.random.default_rng(0))
    assert points.shape == (200, 2)


def test_scatter_particles_are_all_inside_the_polygon() -> None:
    points = scatter_particles(SPILL_POLYGON, n_particles=300, rng=np.random.default_rng(1))
    assert all(SPILL_POLYGON.contains(Point(x, y)) for x, y in points)


def test_scatter_particles_works_for_an_elongated_thin_polygon() -> None:
    """A thin sliver fills only a small fraction of its own bounding box -
    the rejection-sampling loop must still terminate and stay accurate."""
    thin = box(70.0, 22.0, 70.10, 22.001)
    points = scatter_particles(thin, n_particles=100, rng=np.random.default_rng(2))
    assert points.shape == (100, 2)
    assert all(thin.contains(Point(x, y)) for x, y in points)


def test_scatter_particles_uses_the_configured_density_when_omitted(settings) -> None:
    settings = settings.model_copy(
        update={"drift": settings.drift.model_copy(update={"n_particles": 42})}
    )
    points = scatter_particles(SPILL_POLYGON, rng=np.random.default_rng(0), settings=settings)
    assert points.shape == (42, 2)


# --------------------------------------------------------------------------- #
# _deflect - the Ekman rotation in isolation
# --------------------------------------------------------------------------- #
def test_deflect_a_pure_eastward_vector_turns_toward_the_south() -> None:
    """Northern Hemisphere default: deflection is to the right of the vector -
    facing east, turning right faces south."""
    deflected = _deflect(np.array([1.0, 0.0]), angle_deg=90.0)
    assert deflected[0] == pytest.approx(0.0, abs=1e-9)
    assert deflected[1] == pytest.approx(-1.0)


def test_deflect_by_zero_degrees_is_a_no_op() -> None:
    deflected = _deflect(np.array([3.0, -2.0]), angle_deg=0.0)
    assert deflected[0] == pytest.approx(3.0)
    assert deflected[1] == pytest.approx(-2.0)


def test_deflect_preserves_magnitude() -> None:
    deflected = _deflect(np.array([3.0, 4.0]), angle_deg=17.0)
    assert np.hypot(*deflected) == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# step_particles - deterministic behaviour (diffusion off)
# --------------------------------------------------------------------------- #
def test_step_particles_moves_east_for_an_eastward_wind() -> None:
    start = np.array([[70.0, 22.0]])
    after = step_particles(
        start, dt_hours=1.0, wind_uv=(5.0, 0.0), current_uv=(0.0, 0.0),
        wind_drift_factor=0.03, wind_deflection_deg=15.0, diffusion_std_ms=0.0,
    )
    assert after[0, 0] > start[0, 0]  # moved east (lon increased)


def test_step_particles_deflects_an_eastward_wind_southward() -> None:
    """The whole point of the deflection term: a due-east wind (Northern
    Hemisphere default deflection) must push particles slightly south of
    due east, not straight along the wind."""
    start = np.array([[70.0, 22.0]])
    after = step_particles(
        start, dt_hours=1.0, wind_uv=(5.0, 0.0), current_uv=(0.0, 0.0),
        wind_drift_factor=0.03, wind_deflection_deg=15.0, diffusion_std_ms=0.0,
    )
    assert after[0, 1] < start[0, 1]  # moved south (lat decreased)


def test_step_particles_with_zero_deflection_moves_due_east_for_due_east_wind() -> None:
    start = np.array([[70.0, 22.0]])
    after = step_particles(
        start, dt_hours=1.0, wind_uv=(5.0, 0.0), current_uv=(0.0, 0.0),
        wind_drift_factor=0.03, wind_deflection_deg=0.0, diffusion_std_ms=0.0,
    )
    assert after[0, 1] == pytest.approx(start[0, 1], abs=1e-12)  # no north/south drift
    assert after[0, 0] > start[0, 0]


def test_step_particles_current_is_not_deflected() -> None:
    """Only the wind term goes through the Ekman rotation - current is added
    directly, matching velocity = factor*deflect(wind) + current."""
    start = np.array([[70.0, 22.0]])
    wind_only = step_particles(
        start, dt_hours=1.0, wind_uv=(0.0, 0.0), current_uv=(1.0, 0.0),
        wind_drift_factor=0.03, wind_deflection_deg=45.0, diffusion_std_ms=0.0,
    )
    assert wind_only[0, 1] == pytest.approx(start[0, 1], abs=1e-12)  # current alone: due east only
    assert wind_only[0, 0] > start[0, 0]


def test_step_particles_broadcasts_a_single_vector_to_every_particle() -> None:
    """Same latitude for all three, so the longitude-scaling factor (which
    legitimately varies with latitude) can't mask a broadcast bug."""
    start = np.array([[70.0, 22.0], [70.01, 22.0], [70.02, 22.0]])
    after = step_particles(
        start, dt_hours=1.0, wind_uv=(5.0, 0.0), current_uv=(0.0, 0.0), diffusion_std_ms=0.0,
    )
    displacements = after - start
    # every particle moved by the same amount - one shared wind vector
    assert np.allclose(displacements[0], displacements[1])
    assert np.allclose(displacements[1], displacements[2])


def test_step_particles_longitude_displacement_accounts_for_latitude() -> None:
    """A degree of longitude is shorter away from the equator - the same
    eastward wind must displace a higher-latitude particle by more degrees
    of longitude than one nearer the equator."""
    start = np.array([[70.0, 5.0], [70.0, 60.0]])
    after = step_particles(
        start, dt_hours=1.0, wind_uv=(5.0, 0.0), current_uv=(0.0, 0.0), diffusion_std_ms=0.0,
    )
    dlon_low_lat, dlon_high_lat = (after - start)[:, 0]
    assert dlon_high_lat > dlon_low_lat


def test_step_particles_accepts_a_per_particle_vector_array() -> None:
    start = np.array([[70.0, 22.0], [70.0, 22.0]])
    wind_per_particle = np.array([[5.0, 0.0], [-5.0, 0.0]])
    after = step_particles(
        start, dt_hours=1.0, wind_uv=wind_per_particle, current_uv=(0.0, 0.0),
        diffusion_std_ms=0.0,
    )
    assert after[0, 0] > start[0, 0]  # first particle pushed east
    assert after[1, 0] < start[1, 0]  # second particle pushed west


# --------------------------------------------------------------------------- #
# dt sign flip - the SAME function hindcasts
# --------------------------------------------------------------------------- #
def test_forward_then_backward_with_no_diffusion_returns_exactly_to_start() -> None:
    """With the random term off, the deterministic part must cancel exactly -
    proof that hindcasting is nothing but dt's sign, not a separate path."""
    start = np.array([[70.0, 22.0], [70.02, 22.01]])
    forward = step_particles(
        start, dt_hours=2.0, wind_uv=(4.0, 2.0), current_uv=(0.3, -0.2), diffusion_std_ms=0.0,
    )
    back = step_particles(
        forward, dt_hours=-2.0, wind_uv=(4.0, 2.0), current_uv=(0.3, -0.2), diffusion_std_ms=0.0,
    )
    assert np.allclose(back, start, atol=1e-10)


def test_forward_then_backward_with_diffusion_lands_close_to_start() -> None:
    """The acceptance case: diffusion noise means the round trip does not
    cancel exactly, but must stay small - not drift arbitrarily far away.
    """
    rng = np.random.default_rng(42)
    start = scatter_particles(SPILL_POLYGON, n_particles=30, rng=rng)

    forward = simulate_drift(
        start, n_steps=24, dt_hours=1.0, wind_uv=(3.0, 1.0), current_uv=(0.2, -0.1),
        snapshot_interval_hours=6.0, rng=rng,
    )
    back = simulate_drift(
        forward[-1].positions, n_steps=24, dt_hours=-1.0, wind_uv=(3.0, 1.0),
        current_uv=(0.2, -0.1), snapshot_interval_hours=6.0, rng=rng,
    )

    displacement_deg = np.linalg.norm(back[-1].positions - start, axis=1)
    # ~111 km/deg at this latitude - 0.05 deg is ~5.5 km, generous for 48h of
    # small (0.05 m/s std) diffusion noise over 48 one-hour steps.
    assert displacement_deg.max() < 0.05


# --------------------------------------------------------------------------- #
# simulate_drift - snapshot cadence
# --------------------------------------------------------------------------- #
def test_simulate_drift_snapshots_at_the_configured_interval() -> None:
    start = np.array([[70.0, 22.0]])
    snapshots = simulate_drift(
        start, n_steps=24, dt_hours=1.0, wind_uv=(2.0, 0.0), current_uv=(0.0, 0.0),
        snapshot_interval_hours=6.0, diffusion_std_ms=0.0,
    )
    hours = [s.hours_elapsed for s in snapshots]
    assert hours == [0.0, 6.0, 12.0, 18.0, 24.0]
    assert all(isinstance(s, DriftSnapshot) for s in snapshots)


def test_simulate_drift_hindcast_snapshots_carry_negative_hours() -> None:
    start = np.array([[70.0, 22.0]])
    snapshots = simulate_drift(
        start, n_steps=12, dt_hours=-1.0, wind_uv=(2.0, 0.0), current_uv=(0.0, 0.0),
        snapshot_interval_hours=6.0, diffusion_std_ms=0.0,
    )
    hours = [s.hours_elapsed for s in snapshots]
    assert hours == [0.0, -6.0, -12.0]


def test_simulate_drift_always_includes_a_final_snapshot_off_cadence() -> None:
    start = np.array([[70.0, 22.0]])
    snapshots = simulate_drift(
        start, n_steps=10, dt_hours=1.0, wind_uv=(2.0, 0.0), current_uv=(0.0, 0.0),
        snapshot_interval_hours=6.0, diffusion_std_ms=0.0,
    )
    assert snapshots[-1].hours_elapsed == pytest.approx(10.0)


def test_simulate_drift_uses_configured_defaults(settings) -> None:
    settings = settings.model_copy(
        update={"drift": settings.drift.model_copy(update={"snapshot_interval_hours": 3.0})}
    )
    start = np.array([[70.0, 22.0]])
    snapshots = simulate_drift(
        start, n_steps=6, dt_hours=1.0, wind_uv=(1.0, 0.0), current_uv=(0.0, 0.0),
        diffusion_std_ms=0.0, settings=settings,
    )
    assert [s.hours_elapsed for s in snapshots] == [0.0, 3.0, 6.0]
