"""AIS loader tests: column-alias resolution and normalization, no network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ais.loader import (
    AISLoadError,
    NORMALIZED_COLUMNS,
    load_ais,
    load_ais_csv,
    normalize_ais_frame,
    synthetic_ais_sample,
)


def test_normalize_recognizes_noaa_marinecadastre_headers() -> None:
    raw = pd.DataFrame({
        "MMSI": [123456789],
        "BaseDateTime": ["2023-05-13T12:00:00"],
        "LAT": [22.0],
        "LON": [70.0],
        "SOG": [8.5],
        "COG": [90.0],
        "Heading": [91.0],
        "VesselName": ["MV TEST"],
        "VesselType": ["Cargo"],
    })
    frame = normalize_ais_frame(raw)
    assert list(frame.columns) == list(NORMALIZED_COLUMNS)
    assert frame.loc[0, "mmsi"] == 123456789
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2023-05-13T12:00:00", tz="UTC")
    assert frame.loc[0, "vessel_name"] == "MV TEST"


def test_normalize_recognizes_danish_maritime_authority_headers() -> None:
    """DMA's own column names, including the odd '# Timestamp' spelling."""
    raw = pd.DataFrame({
        "MMSI": [219000001],
        "# Timestamp": ["31/12/2019 23:59:59"],
        "Latitude": [56.0],
        "Longitude": [11.0],
        "SOG": [12.0],
        "COG": [180.0],
        "Name": ["NORDIC STAR"],
        "Ship type": ["Tanker"],
    })
    frame = normalize_ais_frame(raw)
    assert frame.loc[0, "mmsi"] == 219000001
    assert frame.loc[0, "vessel_name"] == "NORDIC STAR"
    assert frame.loc[0, "vessel_type"] == "Tanker"


def test_normalize_parses_dma_day_first_dates_correctly() -> None:
    """05/06/2019 must be 5 June, not 6 May - the ambiguous case pandas would
    otherwise guess wrong on a DMA export.
    """
    raw = pd.DataFrame({
        "MMSI": [1], "# Timestamp": ["05/06/2019 10:00:00"],
        "Latitude": [1.0], "Longitude": [1.0],
    })
    frame = normalize_ais_frame(raw)
    ts = frame.loc[0, "timestamp"]
    assert (ts.day, ts.month, ts.year) == (5, 6, 2019)


def test_normalize_accepts_an_explicit_column_map_for_an_unknown_source() -> None:
    raw = pd.DataFrame({
        "vessel_id": [1], "seen_at": ["2023-05-13T00:00:00Z"],
        "y": [10.0], "x": [20.0],
    })
    frame = normalize_ais_frame(
        raw, column_map={"mmsi": "vessel_id", "timestamp": "seen_at", "lat": "y", "lon": "x"}
    )
    assert frame.loc[0, ["mmsi", "lat", "lon"]].tolist() == [1, 10.0, 20.0]


def test_normalize_requires_the_core_fields() -> None:
    with pytest.raises(AISLoadError, match="required column"):
        normalize_ais_frame(pd.DataFrame({"MMSI": [1], "LAT": [1.0]}))


def test_normalize_drops_rows_missing_a_required_field() -> None:
    raw = pd.DataFrame({
        "MMSI": [1, 2], "BaseDateTime": ["2023-05-13T00:00:00", None],
        "LAT": [1.0, 2.0], "LON": [1.0, 2.0],
    })
    frame = normalize_ais_frame(raw)
    assert len(frame) == 1
    assert frame.loc[0, "mmsi"] == 1


def test_normalize_sorts_by_mmsi_then_time() -> None:
    raw = pd.DataFrame({
        "MMSI": [2, 1, 1],
        "BaseDateTime": ["2023-05-13T02:00:00", "2023-05-13T01:00:00", "2023-05-13T00:00:00"],
        "LAT": [1.0, 1.0, 1.0], "LON": [1.0, 1.0, 1.0],
    })
    frame = normalize_ais_frame(raw)
    assert frame["mmsi"].tolist() == [1, 1, 2]
    assert frame["timestamp"].is_monotonic_increasing or (
        frame.groupby("mmsi")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()
    )


def test_load_ais_csv_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "ais.csv"
    pd.DataFrame({
        "MMSI": [1], "BaseDateTime": ["2023-05-13T00:00:00"], "LAT": [1.0], "LON": [1.0],
    }).to_csv(path, index=False)
    frame = load_ais_csv(path)
    assert len(frame) == 1
    assert load_ais(path) is not None  # extension dispatch works too


def test_load_ais_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(AISLoadError, match="no AIS file"):
        load_ais_csv(tmp_path / "missing.csv")


def test_load_ais_rejects_an_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "ais.txt"
    path.write_text("mmsi\n1\n")
    with pytest.raises(AISLoadError, match="unrecognized"):
        load_ais(path)


def test_synthetic_ais_sample_needs_no_download() -> None:
    frame = synthetic_ais_sample(n_vessels=3, points_per_vessel=4)
    assert list(frame.columns) == list(NORMALIZED_COLUMNS)
    assert frame["mmsi"].nunique() == 3
    assert len(frame) == 12
    assert frame["timestamp"].dt.tz is not None


def test_synthetic_ais_sample_vessels_move() -> None:
    """A straight-line track, not a stack of identical points."""
    frame = synthetic_ais_sample(n_vessels=1, points_per_vessel=5)
    assert frame["lat"].nunique() > 1 or frame["lon"].nunique() > 1
