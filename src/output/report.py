"""Step 5.3: exportable incident report - one static HTML file per pipeline
run, combining everything ``scripts/run_pipeline.py`` already computed (spill
summary, the folium map, ranked vessels with their explanations, drift
forecast) into a single document a non-technical reader can open and share.

HTML, not PDF: no extra rendering dependency (wkhtmltopdf/weasyprint) to get
working reliably in this environment, no pagination/font shape to get right
for a demo tool, and it reuses :func:`src.output.map.build_map`'s folium
output directly via its own ``_repr_html_()`` embed (a self-contained
``<iframe srcdoc="...">``, the same thing Jupyter uses to render a folium map
inline) - no separate rendering path to maintain. If a PDF ever becomes a
real requirement, this is the one file that would need a
``weasyprint.HTML(string=...).write_pdf()`` call added at the bottom; none of
the content-building functions below would change.
"""

from __future__ import annotations

import html as html_lib
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.output.map import build_map

STYLE = """
body { font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem;
       color: #1a1a1a; max-width: 960px; }
h1 { border-bottom: 3px solid #1a1a1a; padding-bottom: 0.3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }
h3 { margin-top: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.92rem; }
th { background: #f2f2f2; }
.meta { color: #555; font-size: 0.9rem; }
.status-active { color: #b00000; font-weight: bold; }
.status-possible { color: #a06000; font-weight: bold; }
.status-rejected { color: #555; }
.spill-section { border: 1px solid #ddd; border-radius: 6px; padding: 1rem 1.25rem; margin: 1.5rem 0; }
.explanation { color: #333; font-size: 0.88rem; }
"""


class ReportBuildError(RuntimeError):
    """A combined pipeline result could not be turned into a report."""


def _fmt(value: Optional[float], digits: int = 3, suffix: str = "") -> str:
    return "unknown" if value is None else f"{value:.{digits}f}{suffix}"


def _status_class(status: Optional[str]) -> str:
    return f"status-{status}" if status in ("active", "possible", "rejected") else ""


def _spill_summary_table(spill: Dict[str, Any], alert: Dict[str, Any]) -> str:
    props = spill.get("properties") or {}
    rows = [
        ("Spill ID", props.get("spill_id", "unknown")),
        ("Scene ID", props.get("scene_id", "unknown")),
        ("Acquisition time", props.get("acquisition_timestamp") or "unknown"),
        ("Location (lon, lat)", f"{_fmt(props.get('centroid_lon'), 5)}, {_fmt(props.get('centroid_lat'), 5)}"),
        ("Area", _fmt(props.get("area_km2"), 3, " km²")),
        ("Mean confidence", _fmt(props.get("mean_confidence"), 3)),
        ("Elongation", _fmt(props.get("elongation"), 2)),
        ("Estimated volume", _fmt(props.get("estimated_volume_m3"), 1, " m³")),
        (
            "Alert status",
            f'<span class="{_status_class(alert.get("status"))}">{html_lib.escape(str(alert.get("status", "unknown")))}</span>',
        ),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _alert_rules_list(alert: Dict[str, Any]) -> str:
    rules = alert.get("rules") or []
    if not rules:
        return "<p>No rule evaluations recorded.</p>"
    items = "".join(
        f"<li>{'✓' if r.get('passed') else '✗'} <b>{html_lib.escape(str(r.get('name')))}</b>"
        f" - {html_lib.escape(str(r.get('reason', '')))}</li>"
        for r in rules
    )
    return f"<ul>{items}</ul>"


def _vessel_table(vessels: List[Dict[str, Any]]) -> str:
    if not vessels:
        return "<p>No candidate vessels attributed to this spill.</p>"
    header = (
        "<tr><th>Rank</th><th>MMSI</th><th>Name</th><th>Type</th>"
        "<th>Score</th><th>CPA</th><th>Explanation</th></tr>"
    )
    rows = []
    for rank, v in enumerate(vessels):
        rows.append(
            "<tr>"
            f"<td>{rank + 1}</td>"
            f"<td>{html_lib.escape(str(v.get('mmsi', 'unknown')))}</td>"
            f"<td>{html_lib.escape(str(v.get('vessel_name') or 'unknown'))}</td>"
            f"<td>{html_lib.escape(str(v.get('vessel_type') or 'unknown'))}</td>"
            f"<td>{_fmt(v.get('score'), 3)}</td>"
            f"<td>{_fmt(v.get('cpa_distance_km'), 2, ' km')}</td>"
            f"<td class=\"explanation\">{html_lib.escape(str(v.get('explanation', '')))}</td>"
            "</tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def _drift_forecast_table(forecast: List[Dict[str, Any]]) -> str:
    if not forecast:
        return "<p>No drift forecast available for this spill.</p>"
    header = "<tr><th>Horizon</th><th>Estimated time</th></tr>"
    rows = "".join(
        f"<tr><td>+{f.get('hours_elapsed', 0):.0f}h</td><td>{html_lib.escape(str(f.get('time', 'unknown')))}</td></tr>"
        for f in forecast
    )
    return f"<table>{header}{rows}</table>"


def _spill_section(entry: Dict[str, Any], index: int) -> str:
    spill = entry["spill"]
    alert = entry.get("alert") or {}
    vessels = entry.get("vessels") or []
    forecast = entry.get("drift_forecast") or []
    props = spill.get("properties") or {}

    return f"""
    <section class="spill-section">
      <h2>Spill {index + 1}: {html_lib.escape(str(props.get('spill_id', 'unknown')))}</h2>
      {_spill_summary_table(spill, alert)}
      <h3>Alert rules</h3>
      {_alert_rules_list(alert)}
      <h3>Ranked candidate vessels</h3>
      {_vessel_table(vessels)}
      <h3>Drift forecast</h3>
      {_drift_forecast_table(forecast)}
    </section>
    """


def build_report_html(result: Dict[str, Any]) -> str:
    """A complete, standalone HTML incident report for one combined
    ``scripts/run_pipeline.py`` result.

    Every spill in ``result["spills"]`` gets its own section (alerted or
    not, matching :func:`src.output.map.build_map`'s own "show everything,
    explain why" stance); the map embedded at the top covers all of them.
    """
    spills = result.get("spills") or []
    if not spills:
        raise ReportBuildError("pipeline result has no spills to report")

    map_embed = build_map(result)._repr_html_()
    sections = "".join(_spill_section(entry, i) for i, entry in enumerate(spills))
    scene_id = html_lib.escape(str(result.get("scene_id", "unknown")))
    generated_at = html_lib.escape(str(result.get("generated_at", "unknown")))
    alerted = sum(1 for e in spills if (e.get("alert") or {}).get("alert"))

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Incident report - {scene_id}</title>
<style>{STYLE}</style>
</head>
<body>
<h1>SAR oil-spill incident report</h1>
<p class="meta">
  Scene <b>{scene_id}</b> &middot; generated {generated_at} &middot;
  {len(spills)} spill(s) detected, {alerted} alerted
</p>
<h2>Map</h2>
{map_embed}
{sections}
</body>
</html>
"""


def save_report(result: Dict[str, Any], output_path: Path) -> Path:
    """Build the report and save it as a standalone HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report_html(result), encoding="utf-8")
    return output_path


__all__ = ["ReportBuildError", "build_report_html", "save_report"]
