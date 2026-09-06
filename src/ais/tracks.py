"""Reconstruct one continuous track per vessel from its raw AIS pings.

AIS reporting is sparse and uneven - a vessel might ping every few seconds
while maneuvering and then not again for half an hour - so connecting raw
points with straight lines would understate how close a vessel actually came
to a spill between two pings. This module uses movingpandas to build a proper
:class:`movingpandas.Trajectory` per MMSI and resamples it onto a regular
cadence (``ais.interpolation_minutes``) by linear interpolation, so a closest
point of approach computed downstream (:mod:`src.ais.filter`) is not blind to
whatever happened between two sparse pings.

A vessel with only one ping in the window cannot be interpolated - there is
nothing to interpolate between - so it is carried through as a single-point
"track" rather than dropped; :mod:`src.ais.filter` treats that position as
its own closest point of approach.

movingpandas trajectories are built and queried in **naive UTC** timestamps -
mixing tz-aware and tz-naive datetimes inside movingpandas raises, and it
silently drops tz info on construction if given aware ones. Everything that
crosses this module's boundary (``points``, and the ``timestamp`` column of
``resampled``) is tz-aware UTC, matching the rest of the pipeline; the naive
conversion is an internal detail of talking to movingpandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import movingpandas as mpd
import pandas as pd

from src.config import Settings, get_settings

POSITION_COLUMNS = ("timestamp", "lat", "lon")


@dataclass
class VesselTrack:
    """One vessel's reconstructed track across the AIS query window.

    ``points`` is the raw, deduplicated pings as returned by
    :func:`src.ais.query.query_ais_for_spill`. ``resampled`` is the same track
    at a regular cadence, interpolated where AIS reporting left gaps - this is
    what CPA should be measured against, not ``points`` directly.
    """

    mmsi: int
    points: pd.DataFrame
    resampled: pd.DataFrame
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None

    @property
    def has_trajectory(self) -> bool:
        """False when there was only one ping - nothing to interpolate."""
        return len(self.points) >= 2


def _first_non_null(frame: pd.DataFrame, column: str) -> Optional[str]:
    values = frame[column].dropna()
    return str(values.iloc[0]) if not values.empty else None


def _resample_track(points: pd.DataFrame, step_minutes: float, mmsi: int) -> pd.DataFrame:
    """Regularly-spaced positions across ``points``' own observed time span.

    Resampling is bounded by the vessel's own first and last ping, not the
    full query window - a vessel seen for one hour of a 48 hour window has no
    business being assumed stationary for the other 47.
    """
    if len(points) < 2:
        return points[list(POSITION_COLUMNS)].reset_index(drop=True)

    naive = points.assign(
        timestamp=points["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    ).set_index("timestamp")
    trajectory = mpd.Trajectory(naive, traj_id=mmsi, x="lon", y="lat", crs="EPSG:4326")

    start, end = naive.index.min(), naive.index.max()
    times = pd.date_range(start, end, freq=pd.Timedelta(minutes=step_minutes))
    if times.empty or times[-1] != end:
        times = times.append(pd.DatetimeIndex([end]))

    positions = [trajectory.interpolate_position_at(t) for t in times]
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex(times, tz="UTC"),
        "lat": [p.y for p in positions],
        "lon": [p.x for p in positions],
    })


def build_tracks(
    ais_points: pd.DataFrame, settings: Optional[Settings] = None
) -> Dict[int, VesselTrack]:
    """One :class:`VesselTrack` per MMSI in ``ais_points``.

    ``ais_points`` is expected to already be scoped to one spill's space-time
    box, e.g. the output of :func:`src.ais.query.query_ais_for_spill`.
    """
    settings = settings or get_settings()
    step_minutes = settings.ais.interpolation_minutes

    tracks: Dict[int, VesselTrack] = {}
    for mmsi, group in ais_points.groupby("mmsi"):
        points = (
            group.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="first")
            .reset_index(drop=True)
        )
        tracks[int(mmsi)] = VesselTrack(
            mmsi=int(mmsi),
            points=points,
            resampled=_resample_track(points, step_minutes, int(mmsi)),
            vessel_name=_first_non_null(points, "vessel_name"),
            vessel_type=_first_non_null(points, "vessel_type"),
        )
    return tracks


__all__ = ["VesselTrack", "build_tracks"]
