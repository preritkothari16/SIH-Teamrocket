"""filter.py tests: CPA correctness, and the keep/drop decision it drives.

Three synthetic tracks, one per behaviour the filter must tell apart:
a vessel that crosses close to the spill polygon, one that stays far away,
and one that sits still near the polygon (moored, not a spiller).

The spill polygon sits on the equator so 1 degree of longitude and latitude
are both ~111.32 km, which keeps the expected CPA distances hand-computable
without a latitude correction.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import box

from src.ais.filter import closest_point_of_approach, filter_candidates, is_stationary
from src.ais.tracks import build_tracks
from src.config import load_settings

DEG_KM = 111.32
SPILL_POLYGON = box(10.0, 0.0, 10.01, 0.01)  # ~1.1 km square, at the equator
ACQUIRED = pd.Timestamp("2023-05-14T00:00:00Z")
T0 = ACQUIRED - pd.Timedelta(hours=2)


def ais_row(mmsi, when, lat, lon, sog, name):
    return {
        "mmsi": mmsi, "timestamp": when, "lat": lat, "lon": lon,
        "sog": sog, "cog": 90.0, "heading": 90.0,
        "vessel_name": name, "vessel_type": "Cargo",
    }


def straight_line_track(mmsi, lons, lats, sog, name, n=4, step_minutes=40):
    """n evenly-timed pings moving linearly from the first to the last (lon, lat)."""
    rows = []
    for i in range(n):
        f = i / (n - 1)
        lon = lons[0] + (lons[1] - lons[0]) * f
        lat = lats[0] + (lats[1] - lats[0]) * f
        rows.append(ais_row(mmsi, T0 + pd.Timedelta(minutes=step_minutes * i), lat, lon, sog, name))
    return rows


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def three_vessel_ais() -> pd.DataFrame:
    rows = []
    # 1: CROSSER - sails straight through the polygon
    rows += straight_line_track(1, (9.9, 10.11), (0.005, 0.005), sog=10.0, name="CROSSER")
    # 2: FAR - stays roughly 2200 km away the whole time
    rows += straight_line_track(2, (30.0, 30.01), (30.0, 30.01), sog=10.0, name="FAR")
    # 3: MOORED - sits still just east of the polygon
    rows += straight_line_track(3, (10.02, 10.02), (0.005, 0.005), sog=0.05, name="MOORED")
    return pd.DataFrame(rows)


def test_cpa_of_a_crossing_track_is_near_zero(three_vessel_ais, settings) -> None:
    track = build_tracks(three_vessel_ais, settings=settings)[1]
    cpa_km, cpa_time = closest_point_of_approach(track, SPILL_POLYGON)
    assert cpa_km == pytest.approx(0.0, abs=0.1)
    assert T0 <= cpa_time <= T0 + pd.Timedelta(minutes=120)


def test_cpa_of_a_distant_track_is_large(three_vessel_ais, settings) -> None:
    track = build_tracks(three_vessel_ais, settings=settings)[2]
    cpa_km, _ = closest_point_of_approach(track, SPILL_POLYGON)
    assert cpa_km > 1000.0


def test_cpa_of_a_moored_track_matches_the_hand_computed_distance(
    three_vessel_ais, settings
) -> None:
    """The vessel sits at lon 10.02, 0.01 deg (~1.11 km) east of the polygon's
    east edge (lon 10.01), at a latitude inside the polygon's own range - so
    the closest point on the polygon is directly due west, at exactly that
    longitude gap.
    """
    track = build_tracks(three_vessel_ais, settings=settings)[3]
    cpa_km, _ = closest_point_of_approach(track, SPILL_POLYGON)
    expected_km = (10.02 - 10.01) * DEG_KM
    assert cpa_km == pytest.approx(expected_km, rel=0.02)


def test_is_stationary_flags_near_zero_speed_for_the_whole_window(
    three_vessel_ais, settings
) -> None:
    tracks = build_tracks(three_vessel_ais, settings=settings)
    assert is_stationary(tracks[3], settings.ais.min_moving_speed_knots) is True
    assert is_stationary(tracks[1], settings.ais.min_moving_speed_knots) is False
    assert is_stationary(tracks[2], settings.ais.min_moving_speed_knots) is False


def test_filter_candidates_keeps_the_crosser_drops_far_and_moored(
    three_vessel_ais, settings
) -> None:
    tracks = build_tracks(three_vessel_ais, settings=settings)
    candidates = filter_candidates(tracks, (SPILL_POLYGON, ACQUIRED), settings=settings)

    assert [c.mmsi for c in candidates] == [1]
    assert candidates[0].vessel_name == "CROSSER"
    assert candidates[0].cpa_distance_km == pytest.approx(0.0, abs=0.1)
    assert not candidates[0].track.empty


def test_filter_candidates_reports_why_implicitly_via_absence(
    three_vessel_ais, settings
) -> None:
    """Not a reasons-list like the look-alike filter - but the two dropped
    vessels must still be individually explainable via CPA/is_stationary.
    """
    tracks = build_tracks(three_vessel_ais, settings=settings)
    far_cpa, _ = closest_point_of_approach(tracks[2], SPILL_POLYGON)
    assert far_cpa > settings.ais.search_radius_km
    assert is_stationary(tracks[3], settings.ais.min_moving_speed_knots)


def test_filter_candidates_sorts_nearest_first_and_caps_the_list(settings) -> None:
    rows = []
    rows += straight_line_track(10, (9.9, 10.11), (0.005, 0.005), sog=10.0, name="NEAR")
    # ~15.6 km north of the polygon - inside the 25 km search buffer, but
    # clearly farther than NEAR's near-zero CPA.
    rows += straight_line_track(20, (10.005, 10.006), (0.15, 0.151), sog=10.0, name="FARTHER")
    ais = pd.DataFrame(rows)
    tracks = build_tracks(ais, settings=settings)

    all_candidates = filter_candidates(tracks, (SPILL_POLYGON, ACQUIRED), settings=settings)
    assert [c.mmsi for c in all_candidates] == [10, 20]
    assert all_candidates[0].cpa_distance_km <= all_candidates[1].cpa_distance_km

    capped_settings = settings.model_copy(
        update={"ais": settings.ais.model_copy(update={"max_candidate_vessels": 1})}
    )
    capped = filter_candidates(tracks, (SPILL_POLYGON, ACQUIRED), settings=capped_settings)
    assert [c.mmsi for c in capped] == [10]


def test_filter_candidates_on_no_tracks_yields_no_candidates(settings) -> None:
    assert filter_candidates({}, (SPILL_POLYGON, ACQUIRED), settings=settings) == []
