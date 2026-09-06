"""Attribution scoring tests.

Each sub-score is tested for its documented shape in isolation (monotonic
where a factor claims to be, peaked where it claims to be), then one combined
case checks that a candidate with better evidence on every factor actually
outranks one with worse evidence on every factor - the point of scoring
candidates at all instead of just ranking by CPA distance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.ais.filter import CandidateVessel
from src.attribution.scoring import (
    _distance_to_polygon_km,
    alignment_score,
    build_explanation,
    score_candidate,
    score_candidates,
    slick_axis_bearing,
    spatial_score,
    temporal_score,
    type_prior,
)
from src.config import load_settings
from src.drift.hindcast import OriginSnapshot

ACQUIRED = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings():
    return load_settings()


def make_candidate(
    mmsi=1, cpa_km=1.0, lag_hours=6.0, heading=None, vessel_type="Tanker", name="V",
) -> CandidateVessel:
    return CandidateVessel(
        mmsi=mmsi, cpa_distance_km=cpa_km, cpa_time=ACQUIRED - timedelta(hours=lag_hours),
        track=pd.DataFrame(), vessel_name=name, vessel_type=vessel_type,
        heading_at_cpa=heading,
    )


# --------------------------------------------------------------------------- #
# 1. spatial_score - closer is higher
# --------------------------------------------------------------------------- #
def test_spatial_score_is_maximal_at_zero_distance() -> None:
    assert spatial_score(0.0, scale_km=10.0) == pytest.approx(1.0)


def test_spatial_score_is_strictly_monotonic_decreasing() -> None:
    distances = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]
    scores = [spatial_score(d, scale_km=10.0) for d in distances]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)  # strictly, not just non-increasing


def test_spatial_score_is_always_between_zero_and_one() -> None:
    for d in [0.0, 1e-6, 100.0, 1e6]:
        assert 0.0 <= spatial_score(d, scale_km=10.0) <= 1.0


# --------------------------------------------------------------------------- #
# 2. temporal_score - a peaked curve, not a naive linear one
# --------------------------------------------------------------------------- #
def test_temporal_score_peaks_at_the_optimal_lag() -> None:
    optimal, scale = 6.0, 8.0
    at_optimum = temporal_score(ACQUIRED - timedelta(hours=optimal), ACQUIRED, optimal, scale)
    assert at_optimum == pytest.approx(1.0)


def test_temporal_score_penalizes_very_recent_lag() -> None:
    """The whole point of this factor: lag near zero must score below the
    optimum, because oil has not had time to spread into a visible slick.
    """
    optimal, scale = 6.0, 8.0
    near_zero = temporal_score(ACQUIRED, ACQUIRED, optimal, scale)
    at_optimum = temporal_score(ACQUIRED - timedelta(hours=optimal), ACQUIRED, optimal, scale)
    assert near_zero < at_optimum


def test_temporal_score_also_penalizes_lag_well_past_the_optimum() -> None:
    """Falls off on *both* sides - a naive "more recent is better" line could
    not do this."""
    optimal, scale = 6.0, 8.0
    at_optimum = temporal_score(ACQUIRED - timedelta(hours=optimal), ACQUIRED, optimal, scale)
    much_earlier = temporal_score(ACQUIRED - timedelta(hours=48), ACQUIRED, optimal, scale)
    assert much_earlier < at_optimum


def test_temporal_score_is_zero_for_a_cpa_after_acquisition() -> None:
    """A vessel whose closest approach happened after the image cannot have
    caused what the image already shows."""
    score = temporal_score(ACQUIRED + timedelta(hours=1), ACQUIRED, 6.0, 8.0)
    assert score == 0.0


def test_temporal_score_is_symmetric_in_log_lag_space_around_the_shape() -> None:
    """Monotonic decreasing on each side of the optimum, independently."""
    optimal, scale = 6.0, 8.0
    before = [temporal_score(ACQUIRED - timedelta(hours=h), ACQUIRED, optimal, scale)
              for h in [6, 10, 20, 40]]
    assert before == sorted(before, reverse=True)
    after = [temporal_score(ACQUIRED - timedelta(hours=h), ACQUIRED, optimal, scale)
             for h in [0, 2, 4, 6]]
    assert after == sorted(after)


# --------------------------------------------------------------------------- #
# 3. alignment_score - the strongest single discriminator
# --------------------------------------------------------------------------- #
def test_alignment_score_is_one_for_a_perfectly_aligned_heading() -> None:
    assert alignment_score(45.0, 45.0) == pytest.approx(1.0)


def test_alignment_score_is_zero_when_perpendicular() -> None:
    assert alignment_score(90.0, 0.0) == pytest.approx(0.0)
    assert alignment_score(0.0, 90.0) == pytest.approx(0.0)


def test_alignment_score_treats_the_axis_as_undirected() -> None:
    """A heading running the *other* way along the same line scores the same -
    a slick's axis has no "forward"."""
    assert alignment_score(10.0, 10.0) == pytest.approx(alignment_score(190.0, 10.0))


def test_alignment_score_is_monotonic_in_angular_difference() -> None:
    diffs = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
    scores = [alignment_score(d, 0.0) for d in diffs]
    assert scores == sorted(scores, reverse=True)


def test_alignment_score_is_neutral_when_heading_is_unknown() -> None:
    assert alignment_score(None, 45.0) == pytest.approx(0.5)
    assert alignment_score(45.0, None) == pytest.approx(0.5)


def test_slick_axis_bearing_converts_math_orientation_to_compass_bearing() -> None:
    """orientation_and_elongation() reports degrees CCW from east; a compass
    bearing is degrees CW from north - a 20 deg CCW-from-east line points
    70 deg CW-from-north (verified against the actual function's convention).
    """
    assert slick_axis_bearing(20.0) == pytest.approx(70.0)
    assert slick_axis_bearing(0.0) == pytest.approx(90.0)
    assert slick_axis_bearing(None) is None


# --------------------------------------------------------------------------- #
# 4. type_prior - configurable weight table
# --------------------------------------------------------------------------- #
def test_type_prior_ranks_tanker_and_cargo_above_fishing_and_leisure(settings) -> None:
    cfg = settings.attribution
    assert type_prior("tanker", cfg) > type_prior("fishing", cfg)
    assert type_prior("cargo", cfg) > type_prior("leisure", cfg)


def test_type_prior_is_case_insensitive(settings) -> None:
    cfg = settings.attribution
    assert type_prior("Tanker", cfg) == type_prior("tanker", cfg)


def test_type_prior_falls_back_to_default_for_unknown_or_missing_type(settings) -> None:
    cfg = settings.attribution
    assert type_prior("blimp", cfg) == cfg.default_type_prior
    assert type_prior(None, cfg) == cfg.default_type_prior
    assert type_prior("", cfg) == cfg.default_type_prior


# --------------------------------------------------------------------------- #
# explanation string
# --------------------------------------------------------------------------- #
def test_explanation_names_all_four_factors() -> None:
    candidate = make_candidate(cpa_km=1.2, lag_hours=5.0, heading=68.0, vessel_type="Tanker")
    explanation = build_explanation(candidate, ACQUIRED, slick_bearing=70.0)
    assert "1.2 km CPA" in explanation
    assert "5h before image" in explanation
    assert "8° of slick axis" in explanation or "2° of slick axis" in explanation
    assert "tanker" in explanation


def test_explanation_reports_unknown_heading_honestly() -> None:
    candidate = make_candidate(heading=None)
    explanation = build_explanation(candidate, ACQUIRED, slick_bearing=70.0)
    assert "heading unknown" in explanation


# --------------------------------------------------------------------------- #
# combined score - better evidence must outrank worse evidence
# --------------------------------------------------------------------------- #
def test_combined_score_ranks_stronger_evidence_above_weaker(settings) -> None:
    strong = make_candidate(
        mmsi=1, cpa_km=0.5, lag_hours=6.0, heading=71.0, vessel_type="Tanker", name="STRONG",
    )
    weak = make_candidate(
        mmsi=2, cpa_km=24.0, lag_hours=47.0, heading=161.0, vessel_type="Leisure", name="WEAK",
    )
    # slick axis bearing 70 (orientation_deg 20): strong is 1 deg off, weak is
    # perpendicular (161 - 70 = 91, i.e. ~89 deg off after the undirected fold).
    scored = score_candidates([strong, weak], ACQUIRED, slick_orientation_deg=20.0, settings=settings)

    assert [s.candidate.mmsi for s in scored] == [1, 2]
    assert scored[0].score > scored[1].score
    assert scored[0].spatial > scored[1].spatial
    assert scored[0].temporal > scored[1].temporal
    assert scored[0].alignment > scored[1].alignment
    assert scored[0].type_prior > scored[1].type_prior


def test_score_candidate_score_is_bounded_zero_to_one(settings) -> None:
    for candidate in (
        make_candidate(cpa_km=0.0, lag_hours=6.0, heading=90.0, vessel_type="Tanker"),
        make_candidate(cpa_km=1000.0, lag_hours=-5.0, heading=None, vessel_type=None),
    ):
        scored = score_candidate(candidate, ACQUIRED, 20.0, settings=settings)
        assert 0.0 <= scored.score <= 1.0


def test_score_candidates_on_an_empty_list_yields_nothing(settings) -> None:
    assert score_candidates([], ACQUIRED, 20.0, settings=settings) == []


# --------------------------------------------------------------------------- #
# use_hindcasting - step 4.4: score against the drift corridor, not the fixed
# spill polygon, when a candidate's CPA happened while the oil was elsewhere
# --------------------------------------------------------------------------- #
FINAL_POLYGON = box(10.0, 0.0, 10.01, 0.01)  # the spill as observed at acquisition
EARLY_POLYGON = box(10.30, 0.0, 10.31, 0.01)  # ~33 km east: oil's estimated position 5h earlier
GUILTY_POSITION = (10.305, 0.005)  # inside EARLY_POLYGON, where the oil actually was at CPA
INNOCENT_POSITION = (10.005, 0.005)  # inside FINAL_POLYGON


def test_use_hindcasting_improves_rank_for_a_vessel_that_left_before_the_oil_drifted(
    settings,
) -> None:
    """GUILTY was directly under the oil 5h before acquisition, then left; by
    acquisition time the oil had drifted ~33 km away, so scoring GUILTY
    against the spill's *final* polygon (the static path) makes it look like
    it was nowhere near the slick. INNOCENT merely ended up near where the
    oil ended up, at a less plausible lag. Hindcasting must correct this: it
    should not leave GUILTY ranked below INNOCENT once corrected for where
    the oil actually was at CPA time.
    """
    static_guilty_distance = _distance_to_polygon_km(*GUILTY_POSITION, FINAL_POLYGON)
    static_innocent_distance = _distance_to_polygon_km(*INNOCENT_POSITION, FINAL_POLYGON)

    guilty = CandidateVessel(
        mmsi=1, cpa_distance_km=static_guilty_distance, cpa_time=ACQUIRED - timedelta(hours=5),
        track=pd.DataFrame(), vessel_name="GUILTY", vessel_type="Tanker",
        heading_at_cpa=70.0, position_at_cpa=GUILTY_POSITION,
    )
    innocent = CandidateVessel(
        mmsi=2, cpa_distance_km=static_innocent_distance, cpa_time=ACQUIRED - timedelta(hours=1),
        track=pd.DataFrame(), vessel_name="INNOCENT", vessel_type="Tanker",
        heading_at_cpa=70.0, position_at_cpa=INNOCENT_POSITION,
    )
    corridor = [
        OriginSnapshot(
            time=ACQUIRED, hours_before_acquisition=0.0,
            positions=np.empty((0, 2)), polygon=FINAL_POLYGON,
        ),
        OriginSnapshot(
            time=ACQUIRED - timedelta(hours=5), hours_before_acquisition=5.0,
            positions=np.empty((0, 2)), polygon=EARLY_POLYGON,
        ),
    ]

    static_scores = score_candidates([guilty, innocent], ACQUIRED, slick_orientation_deg=20.0, settings=settings)
    assert [s.candidate.mmsi for s in static_scores] == [2, 1]  # INNOCENT unfairly ranked first

    hindcast_settings = settings.model_copy(
        update={"attribution": settings.attribution.model_copy(update={"use_hindcasting": True})}
    )
    hindcast_scores = score_candidates(
        [guilty, innocent], ACQUIRED, slick_orientation_deg=20.0,
        settings=hindcast_settings, hindcast_corridor=corridor,
    )
    assert hindcast_scores[0].candidate.mmsi == 1  # GUILTY now ranks first
    # GUILTY's rank strictly improved (2nd -> 1st); it was never worsened.
    guilty_static_rank = [s.candidate.mmsi for s in static_scores].index(1)
    guilty_hindcast_rank = [s.candidate.mmsi for s in hindcast_scores].index(1)
    assert guilty_hindcast_rank <= guilty_static_rank


def test_use_hindcasting_off_ignores_a_supplied_corridor(settings) -> None:
    """The corridor is inert unless use_hindcasting is on - callers can pass
    it unconditionally without it changing anything by default."""
    guilty = CandidateVessel(
        mmsi=1, cpa_distance_km=_distance_to_polygon_km(*GUILTY_POSITION, FINAL_POLYGON),
        cpa_time=ACQUIRED - timedelta(hours=5), track=pd.DataFrame(),
        vessel_name="GUILTY", vessel_type="Tanker", heading_at_cpa=70.0,
        position_at_cpa=GUILTY_POSITION,
    )
    corridor = [
        OriginSnapshot(
            time=ACQUIRED - timedelta(hours=5), hours_before_acquisition=5.0,
            positions=np.empty((0, 2)), polygon=EARLY_POLYGON,
        ),
    ]
    without_corridor = score_candidate(guilty, ACQUIRED, 20.0, settings=settings)
    with_corridor = score_candidate(guilty, ACQUIRED, 20.0, settings=settings, hindcast_corridor=corridor)
    assert without_corridor.score == pytest.approx(with_corridor.score)
    assert without_corridor.cpa_distance_km == pytest.approx(with_corridor.cpa_distance_km)
