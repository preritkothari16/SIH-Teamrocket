"""Step 5.4: a minimal FastAPI dashboard for one pipeline run's output.

Serves a single page: the layer-toggle map from :mod:`src.output.map`
(slick / vessel tracks / drift forecast / hindcast corridor - the toggle
itself lives there, this just supplies the hindcast layer's data), the
ranked vessel table across every alerted spill, and basic run metadata
(scene ID, acquisition time, alert status).

A demo tool, not a production UI: one process serves one already-computed
``scripts/run_pipeline.py`` result, loaded once at startup from its JSON
file - no auth, no multi-run browser, no live refresh, no database.

FastAPI + a single HTML response, not Streamlit: FastAPI and uvicorn are
already part of this project's stack (``src.config.APIConfig`` exists for
a future real API) and need no new install to run reliably here; Streamlit
is not installed in this environment, and pulling it in for one demo page
would be a new, heavier dependency for no reliability gain.

Run it::

    .venv/Scripts/python.exe -m src.output.dashboard --result data/processed/00000/pipeline_result.json
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src.config import Settings, get_settings
from src.drift.hindcast import OriginSnapshot, hindcast_origin
from src.output.map import build_map

STYLE = """
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 1.5rem;
       color: #1a1a1a; }
h1 { margin-bottom: 0.2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }
th { background: #f2f2f2; }
.meta { display: flex; gap: 2rem; flex-wrap: wrap; margin: 0.5rem 0 1.5rem;
        font-size: 0.95rem; }
.meta div { background: #f7f7f7; border-radius: 6px; padding: 0.5rem 1rem; }
.meta b { display: block; font-size: 0.75rem; color: #666; text-transform: uppercase; }
.status-active { color: #b00000; font-weight: bold; }
.status-possible { color: #a06000; font-weight: bold; }
.status-rejected { color: #555; }
"""


def _fmt(value: Optional[float], digits: int = 3, suffix: str = "") -> str:
    return "unknown" if value is None else f"{value:.{digits}f}{suffix}"


def _status_class(status: Optional[str]) -> str:
    return f"status-{status}" if status in ("active", "possible", "rejected") else ""


def _metadata_html(result: Dict[str, Any]) -> str:
    spills = result.get("spills") or []
    alerted = sum(1 for e in spills if (e.get("alert") or {}).get("alert"))
    first_props = (spills[0]["spill"].get("properties") or {}) if spills else {}
    cards = [
        ("Scene ID", html_lib.escape(str(result.get("scene_id", "unknown")))),
        ("Acquisition time", html_lib.escape(str(first_props.get("acquisition_timestamp", "unknown")))),
        ("Spills detected", str(len(spills))),
        ("Spills alerted", str(alerted)),
        ("AIS source", html_lib.escape(str(result.get("ais_source") or "none"))),
        ("Stub model", "yes" if result.get("stub_model") else "no"),
    ]
    return "".join(f'<div><b>{k}</b>{v}</div>' for k, v in cards)


def _vessel_rows(result: Dict[str, Any]) -> str:
    rows = []
    for entry in result.get("spills") or []:
        alert = entry.get("alert") or {}
        spill_id = alert.get("spill_id") or "unknown"
        for rank, vessel in enumerate(entry.get("vessels") or []):
            rows.append(
                "<tr>"
                f"<td>{html_lib.escape(str(spill_id))}</td>"
                f"<td>{rank + 1}</td>"
                f"<td>{html_lib.escape(str(vessel.get('mmsi', 'unknown')))}</td>"
                f"<td>{html_lib.escape(str(vessel.get('vessel_name') or 'unknown'))}</td>"
                f"<td>{html_lib.escape(str(vessel.get('vessel_type') or 'unknown'))}</td>"
                f"<td>{_fmt(vessel.get('score'), 3)}</td>"
                f"<td>{_fmt(vessel.get('cpa_distance_km'), 2, ' km')}</td>"
                f"<td>{html_lib.escape(str(vessel.get('explanation', '')))}</td>"
                "</tr>"
            )
    if not rows:
        return "<p>No candidate vessels attributed in this run.</p>"
    header = (
        "<tr><th>Spill</th><th>Rank</th><th>MMSI</th><th>Name</th><th>Type</th>"
        "<th>Score</th><th>CPA</th><th>Explanation</th></tr>"
    )
    return f"<table>{header}{''.join(rows)}</table>"


def _alert_status_html(result: Dict[str, Any]) -> str:
    rows = []
    for entry in result.get("spills") or []:
        alert = entry.get("alert") or {}
        if not alert.get("spill_id"):
            continue  # rejected, never-registered spills clutter this list without adding anything
        status = alert.get("status", "unknown")
        rows.append(
            f'<tr><td>{html_lib.escape(str(alert["spill_id"]))}</td>'
            f'<td class="{_status_class(status)}">{html_lib.escape(str(status))}</td></tr>'
        )
    if not rows:
        return "<p>No registered spills in this run.</p>"
    return f"<table><tr><th>Spill</th><th>Status</th></tr>{''.join(rows)}</table>"


def hindcast_corridors_for(
    result: Dict[str, Any], settings: Settings,
) -> Dict[str, List[OriginSnapshot]]:
    """A hindcast corridor per alerted spill, keyed by spill_id - computed on
    demand (see :mod:`src.drift.hindcast`), never persisted in the combined
    JSON. Shown as its own toggleable map layer regardless of whether
    ``attribution.use_hindcasting`` is on for scoring - visualizing it is
    useful either way; only the scoring behaviour is gated by that flag.
    """
    corridors: Dict[str, List[OriginSnapshot]] = {}
    for entry in result.get("spills") or []:
        alert = entry.get("alert") or {}
        if not alert.get("alert"):
            continue
        spill_id = alert.get("spill_id")
        if not spill_id:
            continue
        corridors[spill_id] = hindcast_origin(entry["spill"], settings=settings)
    return corridors


def render_dashboard_html(result: Dict[str, Any], settings: Optional[Settings] = None) -> str:
    """The dashboard's one page, as a complete HTML string."""
    settings = settings or get_settings()
    corridors = hindcast_corridors_for(result, settings)
    map_embed = build_map(result, hindcast_corridors=corridors)._repr_html_()

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SAR oil-spill dashboard</title>
<style>{STYLE}</style>
</head>
<body>
<h1>SAR oil-spill dashboard</h1>
<div class="meta">{_metadata_html(result)}</div>
<h2>Map</h2>
{map_embed}
<h2>Alert status</h2>
{_alert_status_html(result)}
<h2>Ranked candidate vessels</h2>
{_vessel_rows(result)}
</body>
</html>
"""


def create_app(result_path: Path, settings: Optional[Settings] = None) -> FastAPI:
    """A FastAPI app serving ``result_path``'s pipeline output at ``/``.

    The result is loaded once, here, at app-construction time - reloading it
    per request would let a stale/partial file being rewritten by a
    concurrent pipeline run corrupt an in-flight page load, and this is a
    single-run demo tool, not a live-refreshing one (see step 5.1's poller
    for the live side; wiring that up to this dashboard is future work).
    """
    settings = settings or get_settings()
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))

    app = FastAPI(title="SAR oil-spill dashboard")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard_html(result, settings=settings)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a scripts/run_pipeline.py result as a local dashboard.",
    )
    parser.add_argument("--result", type=Path, required=True,
                        help="path to a pipeline_result.json from scripts/run_pipeline.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None,
                        help="defaults to api.port from config.yaml")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    settings = get_settings()
    app = create_app(args.result, settings=settings)
    port = args.port if args.port is not None else settings.api.port
    uvicorn.run(app, host=args.host, port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["create_app", "render_dashboard_html", "hindcast_corridors_for"]
