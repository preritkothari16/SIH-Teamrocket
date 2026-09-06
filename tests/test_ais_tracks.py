"""tracks.py tests: sparse pings become a continuous, interpolated track."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ais.tracks import VesselTrack, build_tracks
from src.config import load_settings

T0 = pd.Timestamp("2023-05-13T00:00:00Z")


def ais_row(mmsi, when, lat, lon, sog=8.0, name="V", vtype="Cargo"):
    return {
        "mmsi": mmsi, "timestamp": when, "lat": lat, "lon": lon,
        "sog": sog, "cog": 90.0, "heading": 90.0,
        "vessel_name": name, "vessel_type": vtype,
    }


@pytest.fixture
def settings():
    return load_settings()


def test_build_tracks_groups_by_mmsi(settings) -> None:
    df = pd.DataFrame([
        ais_row(1, T0, 20.0, 70.0),
        ais_row(1, T0 + pd.Timedelta(hours=1), 20.1, 70.1),
        ais_row(2, T0, 10.0, 60.0),
    ])
    tracks = build_tracks(df, settings=settings)
    assert set(tracks.keys()) == {1, 2}
    assert len(tracks[1].points) == 2
    assert len(tracks[2].points) == 1


def test_a_two_hour_gap_is_filled_at_the_configured_cadence(settings) -> None:
    df = pd.DataFrame([
        ais_row(1, T0, 20.0, 70.0),
        ais_row(1, T0 + pd.Timedelta(hours=2), 20.2, 70.2),
    ])
    track = build_tracks(df, settings=settings)[1]

    assert track.has_trajectory
    step = pd.Timedelta(minutes=settings.ais.interpolation_minutes)
    expected_points = int(pd.Timedelta(hours=2) / step) + 1
    assert len(track.resampled) == expected_points


def test_interpolation_is_linear_between_two_sparse_pings(settings) -> None:
    df = pd.DataFrame([
        ais_row(1, T0, 20.0, 70.0),
        ais_row(1, T0 + pd.Timedelta(hours=1), 21.0, 71.0),
    ])
    track = build_tracks(df, settings=settings)[1]

    midpoint = track.resampled.iloc[len(track.resampled) // 2]
    # halfway in time must be (close to) halfway in space for a straight line
    assert midpoint["lat"] == pytest.approx(20.5, abs=0.05)
    assert midpoint["lon"] == pytest.approx(70.5, abs=0.05)


def test_resampled_timestamps_stay_tz_aware_utc(settings) -> None:
    df = pd.DataFrame([
        ais_row(1, T0, 20.0, 70.0),
        ais_row(1, T0 + pd.Timedelta(hours=1), 20.1, 70.1),
    ])
    track = build_tracks(df, settings=settings)[1]
    assert track.resampled["timestamp"].dt.tz is not None
    assert str(track.resampled["timestamp"].dt.tz) == "UTC"


def test_a_single_ping_vessel_gets_a_degenerate_one_row_track(settings) -> None:
    df = pd.DataFrame([ais_row(1, T0, 20.0, 70.0)])
    track = build_tracks(df, settings=settings)[1]

    assert not track.has_trajectory
    assert len(track.resampled) == 1
    assert track.resampled.iloc[0]["lat"] == pytest.approx(20.0)


def test_duplicate_timestamps_are_deduplicated(settings) -> None:
    df = pd.DataFrame([
        ais_row(1, T0, 20.0, 70.0),
        ais_row(1, T0, 20.0, 70.0),  # exact duplicate ping
        ais_row(1, T0 + pd.Timedelta(hours=1), 20.1, 70.1),
    ])
    track = build_tracks(df, settings=settings)[1]
    assert len(track.points) == 2


def test_vessel_name_and_type_are_carried_through(settings) -> None:
    df = pd.DataFrame([ais_row(1, T0, 20.0, 70.0, name="MV EXAMPLE", vtype="Tanker")])
    track = build_tracks(df, settings=settings)[1]
    assert track.vessel_name == "MV EXAMPLE"
    assert track.vessel_type == "Tanker"


def test_missing_vessel_metadata_becomes_none(settings) -> None:
    row = ais_row(1, T0, 20.0, 70.0)
    row["vessel_name"] = None
    row["vessel_type"] = None
    track = build_tracks(pd.DataFrame([row]), settings=settings)[1]
    assert track.vessel_name is None
    assert track.vessel_type is None
