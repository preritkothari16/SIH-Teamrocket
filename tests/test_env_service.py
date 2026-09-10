"""get_environment() tests: wind + current in one call, each degrading to
None independently rather than the whole call raising.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.config import load_settings
from src.env_data.service import get_environment
from tests.test_env_currents import write_current_fixture
from tests.test_env_wind import write_wind_fixture

T0 = datetime(2023, 5, 13, 0, 0, 0, tzinfo=timezone.utc)


def _settings_with(tmp_path: Path, wind: bool, current: bool):
    settings = load_settings()
    updates = {}
    if wind:
        updates["wind_dataset_path"] = write_wind_fixture(tmp_path / "wind.nc")
    if current:
        updates["current_dataset_path"] = write_current_fixture(tmp_path / "current.nc")
    return settings.model_copy(update={"env_data": settings.env_data.model_copy(update=updates)})


def test_get_environment_returns_both_when_both_are_configured(tmp_path: Path) -> None:
    settings = _settings_with(tmp_path, wind=True, current=True)
    env = get_environment(20.0, 70.0, T0, settings=settings)
    assert env["wind"] is not None
    assert env["current"] is not None
    assert env["wind"].vector.u == pytest.approx(0.0)
    assert env["current"].vector.u == pytest.approx(0.0)
    assert env["wind"].source == "era5_fixture"
    assert env["current"].source == "cmems_fixture"


def test_get_environment_degrades_wind_alone_when_only_current_is_configured(
    tmp_path: Path,
) -> None:
    settings = _settings_with(tmp_path, wind=False, current=True)
    env = get_environment(20.0, 70.0, T0, settings=settings)
    assert env["wind"] is None
    assert env["current"] is not None


def test_get_environment_degrades_current_alone_when_only_wind_is_configured(
    tmp_path: Path,
) -> None:
    settings = _settings_with(tmp_path, wind=True, current=False)
    env = get_environment(20.0, 70.0, T0, settings=settings)
    assert env["wind"] is not None
    assert env["current"] is None


def test_get_environment_returns_both_none_with_nothing_configured() -> None:
    settings = load_settings()  # defaults: neither source configured, no creds
    env = get_environment(20.0, 70.0, T0, settings=settings)
    assert env == {"wind": None, "current": None}
