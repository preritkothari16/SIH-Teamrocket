"""Check the Sentinel-1 catalogue for new scenes over an AOI, download them,
and hand each one back ready to run through the pipeline.

This is the library half of step 5.1; :mod:`scripts.run_live` is the loop
that calls it repeatedly. Kept separate from
:mod:`src.ingestion.catalogue`/:mod:`src.ingestion.local_source` themselves -
this module only *drives* a :class:`~src.ingestion.types.SceneSource`, it
does not implement one, so :func:`poll_once` works identically against the
real network catalogue or a test stub satisfying the same protocol.

Nothing here is imported by the local pre-downloaded-scene demo path
(``scripts/run_pipeline.py`` against ``data/raw/``); that keeps working
completely unchanged whether or not live polling is ever used.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from shapely.geometry.base import BaseGeometry

from src.config import Settings, get_settings
from src.ingestion.catalogue import CDSECatalogue
from src.ingestion.types import Scene, SceneSource, as_utc

logger = logging.getLogger(__name__)


class PollerError(RuntimeError):
    """A poll cycle could not complete."""


@dataclass
class PollResult:
    """One poll cycle's outcome."""

    checked_at: datetime
    new_scenes: List[Scene]


def load_last_checked(state_path: Path, default_lookback_hours: float) -> datetime:
    """The last successful poll's timestamp, from a small on-disk state file.

    On the very first run (no state file yet), falls back to
    ``now - default_lookback_hours`` rather than the catalogue's entire
    history - the same lookback-window idea ``ingestion.search_lookback_days``
    already uses for a one-shot search.
    """
    path = Path(state_path)
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        return as_utc(datetime.fromisoformat(data["last_checked"]))
    return datetime.now(timezone.utc) - timedelta(hours=default_lookback_hours)


def save_last_checked(state_path: Path, checked_at: datetime) -> None:
    """Persist ``checked_at`` so the next process start resumes from here."""
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_checked": as_utc(checked_at).isoformat()}), encoding="utf-8")


def poll_once(
    aoi: BaseGeometry,
    since: datetime,
    catalogue: Optional[SceneSource] = None,
    settings: Optional[Settings] = None,
    download: bool = True,
) -> PollResult:
    """One check: scenes acquired after ``since``, downloaded and ready to run.

    ``catalogue`` defaults to a real, network-hitting
    :class:`~src.ingestion.catalogue.CDSECatalogue`; tests pass a stub
    satisfying :class:`~src.ingestion.types.SceneSource` (``search()`` /
    ``fetch()``) instead, so this function - and everything built on it -
    never needs live Copernicus access to be exercised.

    Filters strictly newer than ``since`` (not ``>=``): the catalogue's own
    ``search(start, end)`` is inclusive of ``start``, and ``since`` is
    normally the previous cycle's own ``checked_at`` - without the strict
    inequality here, a scene sitting exactly on that boundary would be
    re-fetched and re-run every cycle.
    """
    settings = settings or get_settings()
    catalogue = catalogue or CDSECatalogue(settings=settings)
    since = as_utc(since)
    now = datetime.now(timezone.utc)

    scenes = catalogue.search(aoi, since, now)
    new_scenes = [s for s in scenes if as_utc(s.acquisition_time) > since]
    logger.info("poll: %d new scene(s) since %s", len(new_scenes), since.isoformat())

    if download:
        for scene in new_scenes:
            scene.path = catalogue.fetch(scene)

    return PollResult(checked_at=now, new_scenes=new_scenes)


__all__ = ["PollerError", "PollResult", "load_last_checked", "save_last_checked", "poll_once"]
