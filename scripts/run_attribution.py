"""Phase 2 end to end: a spill GeoJSON + an AIS export in, ranked candidates out.

Chains every AIS/attribution step built so far, for one or more spills in the
input GeoJSON:

    2.1 query    space-time box around the spill (config: ``ais.*``)
    2.2 tracks   reconstruct + interpolate each candidate vessel's track
    2.2 filter   drop tracks whose CPA is outside the buffer, or moored
    2.3 scoring  weighted 0-1 score per survivor, plus a plain-English reason

**No AIS-gap/speed-anomaly features and no hindcasting-based matching yet** -
this is the static buffered space-time search described in the current
design, nothing more. Those are explicitly future work, not an oversight.

Usage::

    python scripts/run_attribution.py \\
        --spill data/processed/spills/00000.geojson \\
        --ais data/ais/some_export.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/run_attribution.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.ais.filter import filter_candidates  # noqa: E402
from src.ais.loader import load_ais  # noqa: E402
from src.ais.query import query_ais_for_spill, spill_geometry_and_time  # noqa: E402
from src.ais.tracks import build_tracks  # noqa: E402
from src.attribution.scoring import ScoredCandidate, score_candidates  # noqa: E402
from src.config import Settings, get_settings  # noqa: E402
from src.drift.hindcast import OriginSnapshot, hindcast_origin  # noqa: E402

logger = logging.getLogger(__name__)


class AttributionRunError(RuntimeError):
    """The Phase 2 chain could not run for one or more spills."""


def load_spills(path: Path) -> List[Dict[str, Any]]:
    """One or more spill GeoJSON Features from a Feature or FeatureCollection."""
    path = Path(path)
    if not path.is_file():
        raise AttributionRunError(f"no spill file at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("type") == "FeatureCollection":
        features = data.get("features") or []
        if not features:
            return []
        return features
    if data.get("type") == "Feature":
        return [data]
    raise AttributionRunError(f"{path} is not a GeoJSON Feature or FeatureCollection")


def _candidate_payload(scored: ScoredCandidate) -> Dict[str, Any]:
    """A scored candidate's numbers plus its track, ready for json.dumps."""
    payload = scored.to_dict()
    payload["track"] = [
        {
            "timestamp": row.timestamp.isoformat(),
            "lat": float(row.lat),
            "lon": float(row.lon),
        }
        for row in scored.candidate.track.itertuples()
    ]
    return payload


def attribute_spill(
    spill: Dict[str, Any], ais: "Any", settings: Optional[Settings] = None,
    hindcast_corridor: Optional[Sequence[OriginSnapshot]] = None,
) -> Dict[str, Any]:
    """Run query -> tracks -> filter -> scoring for one spill Feature.

    ``hindcast_corridor`` (see :func:`src.drift.hindcast.hindcast_origin`) is
    passed straight through to :func:`src.attribution.scoring.score_candidates`
    - only consulted there when ``settings.attribution.use_hindcasting`` is
    true; harmless to pass unconditionally otherwise.
    """
    settings = settings or get_settings()
    properties = spill.get("properties") or {}
    geometry, acquisition_time = spill_geometry_and_time(spill)

    ais_points = query_ais_for_spill(spill, ais, settings=settings)
    vessel_count = int(ais_points["mmsi"].nunique()) if not ais_points.empty else 0
    print(f"[2.1] {vessel_count} vessel(s) in the space-time box")

    tracks = build_tracks(ais_points, settings=settings)
    print(f"[2.2] reconstructed {len(tracks)} track(s)")

    candidates = filter_candidates(tracks, spill, settings=settings)
    print(f"[2.2] {len(candidates)} candidate(s) survive CPA/stationarity")

    scored = score_candidates(
        candidates, acquisition_time, properties.get("orientation_deg"), settings=settings,
        hindcast_corridor=hindcast_corridor,
    )
    print(f"[2.3] {len(scored)} scored candidate(s), best first:")
    for s in scored[:5]:
        print(f"      {s.candidate.mmsi}  {s.score:.3f}  {s.explanation}")

    return {
        "spill_id": properties.get("spill_id"),
        "scene_id": properties.get("scene_id"),
        "acquisition_timestamp": properties.get("acquisition_timestamp"),
        "candidate_count": len(scored),
        "candidates": [_candidate_payload(s) for s in scored],
    }


def run(
    spill_path: Path, ais_path: Path, output: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Run Phase 2 attribution for every spill in ``spill_path`` and return
    (and write) the ranked result."""
    settings = settings or get_settings()

    spills = load_spills(spill_path)
    print(f"\n[2.0] {len(spills)} spill(s) in {spill_path}")
    ais = load_ais(ais_path)
    print(f"      {ais['mmsi'].nunique()} vessel(s), {len(ais)} position report(s) in {ais_path}\n")

    results = []
    for spill in spills:
        corridor = hindcast_origin(spill, settings=settings) if settings.attribution.use_hindcasting else None
        results.append(attribute_spill(spill, ais, settings=settings, hindcast_corridor=corridor))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spill_source": str(spill_path),
        "ais_source": str(ais_path),
        "spills": results,
    }

    target = Path(output) if output else Path(spill_path).with_name(
        f"{Path(spill_path).stem}_attribution.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[2.3] wrote {target}\n")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 AIS attribution chain over a spill GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--spill", type=Path, required=True,
                        help="spill GeoJSON Feature or FeatureCollection (Phase 1 output)")
    parser.add_argument("--ais", dest="ais_path", type=Path, required=True,
                        help="AIS export (.csv or .parquet); source-agnostic, see src/ais/loader.py")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run(spill_path=args.spill, ais_path=args.ais_path, output=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
