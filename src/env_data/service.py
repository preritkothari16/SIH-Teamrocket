"""Unified environmental-vector lookup: wind + ocean current in one call.

``get_environment(lat, lon, time)`` returns ``{"wind": EnvVector | None,
"current": EnvVector | None}``. Each field independently degrades to
``None`` rather than raising if its own lookup is unavailable - no dataset
configured, point outside the grid, no credentials - matching how
:mod:`src.alerts.manager` already treated a missing wind reading before this
module existed: unavailable is a fact for the caller to act on, not a reason
to fail the whole call. Call :func:`src.env_data.wind.get_wind` or
:func:`src.env_data.currents.get_current` directly if you need to know *why*
a field came back empty.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

from src.config import Settings, get_settings
from src.env_data.currents import get_current
from src.env_data.grid import EnvDataError, EnvVector
from src.env_data.wind import get_wind

logger = logging.getLogger(__name__)


def get_environment(
    lat: float, lon: float, time: datetime, settings: Optional[Settings] = None,
) -> Dict[str, Optional[EnvVector]]:
    """Wind and current at ``(lat, lon, time)``, each ``None`` if unavailable."""
    settings = settings or get_settings()

    wind: Optional[EnvVector] = None
    try:
        wind = get_wind(lat, lon, time, settings=settings)
    except EnvDataError as exc:
        logger.info("wind lookup unavailable at (%s, %s, %s): %s", lat, lon, time, exc)

    current: Optional[EnvVector] = None
    try:
        current = get_current(lat, lon, time, settings=settings)
    except EnvDataError as exc:
        logger.info("current lookup unavailable at (%s, %s, %s): %s", lat, lon, time, exc)

    return {"wind": wind, "current": current}


__all__ = ["get_environment"]
