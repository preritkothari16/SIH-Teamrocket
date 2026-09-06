"""AIS anomaly feature tests: synthetic tracks with an injected reporting gap
and an injected slow-down, asserting each factor moves in the expected
direction - and that neither factor guesses when there is not enough
evidence to say anything (0.0, not a penalty).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.attribution.anomaly_features import (
    gap_explanation,
    gap_score,
    largest_gap_hours,
    speed_change_explanation,
    speed_change_score,
)

START = datetime(2023, 5, 14, 0, 0, 0, tzinfo=timezone.utc)


def make_points(timestamps, sog=None) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(list(timestamps), utc=True),
        "lat": [20.0] * n,
        "lon": [70.0] * n,
        "sog": sog if sog is not None else [None] * n,
    })


def evenly_spaced_points(n=6, step_minutes=20, sog=10.0):
    timestamps = [START + timedelta(minutes=step_minutes * i) for i in range(n)]
    return make_points(timestamps, sog=[sog] * n)


# --------------------------------------------------------------------------- #
# largest_gap_hours / gap_score
# --------------------------------------------------------------------------- #
def test_largest_gap_hours_is_zero_for_fewer_than_two_pings() -> None:
    assert largest_gap_hours(make_points([START])) == 0.0
    assert largest_gap_hours(make_points([])) == 0.0


def test_largest_gap_hours_finds_the_biggest_consecutive_interval() -> None:
    timestamps = [
        START, START + timedelta(minutes=10), START + timedelta(hours=8),
        START + timedelta(hours=8, minutes=10),
    ]
    assert largest_gap_hours(make_points(timestamps)) == pytest.approx(7.833333, rel=1e-4)


def test_gap_score_is_zero_at_or_below_the_threshold() -> None:
    """Routine reporting - a normal track with no unusual gap - must not be
    flagged just because pings are not perfectly continuous."""
    points = evenly_spaced_points(step_minutes=20)
    score, gap_hours = gap_score(points, min_gap_hours=1.0, scale_hours=6.0)
    assert score == 0.0
    assert gap_hours < 1.0


def test_gap_score_rises_for_an_injected_transponder_gap() -> None:
    """The core acceptance case: a vessel that goes dark for hours during an
    otherwise normal track must score higher than one that never does."""
    normal = evenly_spaced_points(step_minutes=20)
    with_gap = make_points([
        START, START + timedelta(minutes=20), START + timedelta(minutes=40),
        START + timedelta(hours=10),  # a 9h+ silence injected here
        START + timedelta(hours=10, minutes=20),
    ])

    normal_score, _ = gap_score(normal, min_gap_hours=1.0, scale_hours=6.0)
    gap_injected_score, gap_hours = gap_score(with_gap, min_gap_hours=1.0, scale_hours=6.0)

    assert normal_score == 0.0
    assert gap_injected_score > normal_score
    assert gap_hours > 9.0


def test_gap_score_increases_monotonically_with_gap_size() -> None:
    scores = []
    for hours in (1.0, 2.0, 6.0, 12.0, 24.0):
        points = make_points([START, START + timedelta(hours=hours)])
        score, _ = gap_score(points, min_gap_hours=1.0, scale_hours=6.0)
        scores.append(score)
    assert scores == sorted(scores)
    assert scores[0] == 0.0  # exactly at the threshold
    assert 0.0 < scores[-1] <= 1.0


def test_gap_explanation_names_the_gap_or_says_theres_none() -> None:
    assert gap_explanation(0.2, min_gap_hours=1.0) == "no unusual AIS gap"
    assert "9.5h" in gap_explanation(9.5, min_gap_hours=1.0)


# --------------------------------------------------------------------------- #
# speed_change_score
# --------------------------------------------------------------------------- #
def test_speed_change_score_is_zero_for_a_steady_track() -> None:
    points = evenly_spaced_points(sog=10.0)
    cpa_time = START + timedelta(minutes=40)
    score, baseline, at_cpa = speed_change_score(points, cpa_time, min_baseline_speed_knots=3.0)
    assert score == 0.0
    assert baseline == pytest.approx(10.0)
    assert at_cpa == pytest.approx(10.0)


def test_speed_change_score_rises_for_an_injected_slowdown_near_cpa() -> None:
    """The core acceptance case: a vessel cruising normally but slowing
    sharply right at its closest approach must score higher than steady."""
    timestamps = [START + timedelta(minutes=20 * i) for i in range(6)]
    cpa_time = timestamps[3]
    sog = [10.0, 10.0, 10.0, 0.5, 10.0, 10.0]  # slows to a near-stop at CPA
    points = make_points(timestamps, sog=sog)

    score, baseline, at_cpa = speed_change_score(points, cpa_time, min_baseline_speed_knots=3.0)

    assert baseline == pytest.approx(10.0)
    assert at_cpa == pytest.approx(0.5)
    assert score > 0.9  # near-total slowdown relative to baseline


def test_speed_change_score_is_zero_for_speeding_up_near_cpa() -> None:
    """A bonus factor, not a penalty for the opposite direction: speeding up
    must not score negatively, just 0."""
    timestamps = [START + timedelta(minutes=20 * i) for i in range(4)]
    cpa_time = timestamps[2]
    sog = [5.0, 5.0, 15.0, 5.0]
    points = make_points(timestamps, sog=sog)
    score, _, _ = speed_change_score(points, cpa_time, min_baseline_speed_knots=3.0)
    assert score == 0.0


def test_speed_change_score_is_zero_with_no_sog_reported() -> None:
    points = evenly_spaced_points(sog=None)
    points["sog"] = None
    score, baseline, at_cpa = speed_change_score(
        points, START + timedelta(minutes=40), min_baseline_speed_knots=3.0,
    )
    assert (score, baseline, at_cpa) == (0.0, None, None)


def test_speed_change_score_is_zero_when_baseline_is_already_too_slow_to_assess() -> None:
    """A vessel that was always near-stationary (below min_baseline_speed_knots)
    - is_stationary() upstream would normally have dropped it already, but
    this factor must not fabricate a "slowdown" out of noise either way."""
    points = evenly_spaced_points(sog=1.0)
    score, baseline, at_cpa = speed_change_score(
        points, START + timedelta(minutes=40), min_baseline_speed_knots=3.0,
    )
    assert score == 0.0
    assert baseline == pytest.approx(1.0)
    assert at_cpa is None


def test_speed_change_explanation_covers_every_case() -> None:
    assert speed_change_explanation(None, None) == "speed unknown"
    assert "too low" in speed_change_explanation(1.0, None)
    assert "slowed from 10.0 to 0.5" in speed_change_explanation(10.0, 0.5)
    assert "steady speed" in speed_change_explanation(10.0, 10.0)
