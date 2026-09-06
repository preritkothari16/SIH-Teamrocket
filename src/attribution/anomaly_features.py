"""Two extra, configurable-weight attribution factors on top of the four MVP
ones in :mod:`src.attribution.scoring` (spatial/temporal/alignment/type):

1. :func:`gap_score` - an unusually long AIS reporting gap anywhere in the
   vessel's own space-time-scoped track. Routine AIS reporting is every few
   seconds to a few minutes; a multi-hour silence is not routine, and a
   vessel that wanted to spill unseen has an obvious reason to have one.
2. :func:`speed_change_score` - the vessel running noticeably slower at its
   closest approach than its own baseline speed elsewhere in the track.
   Slowing or stopping near the slick's location/time is consistent with
   discharging or an equipment stop; cruising through at an unchanged speed
   is not.

Both operate on a :class:`~src.ais.filter.CandidateVessel`'s own ``track``
(raw pings - the same DataFrame :func:`src.ais.filter.filter_candidates`
already screened) and ``cpa_time`` - no new AIS query, no new track
reconstruction. Both score 0.0 (no bonus, not a penalty) whenever there is
not enough evidence to tell one way or the other - missing SOG, too few
pings, or a vessel whose baseline speed is already too low to call a further
slowdown meaningful - matching this project's existing "absent data is
neutral, never penalised" pattern (see ``alignment_score``'s 0.5 default,
``type_prior``'s config fallback). A bonus factor's neutral value is 0, not
0.5, since there is nothing to be neutral *between*: an unremarkable track is
not suspicious, it just isn't evidence either way.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd


def largest_gap_hours(points: pd.DataFrame) -> float:
    """The longest interval between two consecutive raw pings, in hours.

    0.0 for a track with fewer than two pings - there is no gap to measure.
    """
    if len(points) < 2:
        return 0.0
    gaps = points["timestamp"].sort_values().diff().dropna()
    if gaps.empty:
        return 0.0
    return float(gaps.dt.total_seconds().max() / 3600.0)


def gap_score(
    points: pd.DataFrame, min_gap_hours: float, scale_hours: float,
) -> Tuple[float, float]:
    """(score, largest_gap_hours) for one vessel's reporting gaps.

    0.0 at or below ``min_gap_hours`` (routine reporting gaps happen and are
    not evidence of anything), then rises smoothly - ``1 - exp(-(gap -
    min_gap_hours) / scale_hours)`` - the same saturating-exponential shape
    :func:`src.attribution.scoring.spatial_score` uses, just increasing
    instead of decaying: more silence is more suspicious, without a single
    threshold making one extra minute the difference between 0 and full
    score.
    """
    gap_hours = largest_gap_hours(points)
    if gap_hours <= min_gap_hours:
        return 0.0, gap_hours
    score = 1.0 - math.exp(-(gap_hours - min_gap_hours) / scale_hours)
    return score, gap_hours


def gap_explanation(gap_hours: float, min_gap_hours: float) -> str:
    if gap_hours <= min_gap_hours:
        return "no unusual AIS gap"
    return f"{gap_hours:.1f}h AIS reporting gap"


def speed_change_score(
    points: pd.DataFrame, cpa_time: datetime, min_baseline_speed_knots: float,
) -> Tuple[float, Optional[float], Optional[float]]:
    """(score, baseline_speed_knots, speed_at_cpa_knots) for one vessel.

    Baseline is the median of every reported SOG value in ``points``; the
    "current" speed is the SOG of the raw ping nearest ``cpa_time``. Score is
    the fractional slowdown, ``max(0, 1 - speed_at_cpa / baseline)``, clipped
    to 1.0 - 0.0 for unchanged or faster, 1.0 for a full stop. Returns
    ``(0.0, None, None)`` when no ping reports SOG at all, and
    ``(0.0, baseline, None)`` when the vessel's own baseline is already below
    ``min_baseline_speed_knots`` - too slow overall to tell a further
    slowdown from noise (this is deliberately about a *change*, not just
    "vessel was slow", which :func:`src.ais.filter.is_stationary` already
    screens for upstream).
    """
    if "sog" not in points.columns or points.empty:
        return 0.0, None, None
    reported = points.dropna(subset=["sog"])
    if reported.empty:
        return 0.0, None, None

    baseline = float(reported["sog"].median())
    if baseline < min_baseline_speed_knots:
        return 0.0, baseline, None

    cpa_ts = pd.Timestamp(cpa_time)
    nearest_index = (reported["timestamp"] - cpa_ts).abs().idxmin()
    speed_at_cpa = float(reported.loc[nearest_index, "sog"])

    score = min(1.0, max(0.0, 1.0 - speed_at_cpa / baseline))
    return score, baseline, speed_at_cpa


def speed_change_explanation(baseline: Optional[float], speed_at_cpa: Optional[float]) -> str:
    if baseline is None:
        return "speed unknown"
    if speed_at_cpa is None:
        return f"baseline speed too low to assess ({baseline:.1f} kn)"
    if speed_at_cpa < baseline:
        return f"slowed from {baseline:.1f} to {speed_at_cpa:.1f} kn near spill"
    return f"steady speed near spill ({speed_at_cpa:.1f} kn)"


__all__ = [
    "largest_gap_hours",
    "gap_score",
    "gap_explanation",
    "speed_change_score",
    "speed_change_explanation",
]
