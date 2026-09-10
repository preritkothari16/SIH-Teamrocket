"""Step 5.3: exportable incident report - one static HTML file per pipeline
run, combining everything ``scripts/run_pipeline.py`` already computed (spill
summary, the folium map, ranked vessels with their explanations, drift
forecast) into a single document a non-technical reader can open and share.

HTML, not PDF: no extra rendering dependency (wkhtmltopdf/weasyprint) to get
working reliably in this environment, no pagination/font shape to get right
for a demo tool, and it reuses :func:`src.output.map.build_map`'s folium
output directly via its own ``_repr_html_()`` embed (a self-contained
``<iframe srcdoc="..."``), the same thing Jupyter uses to render a folium map
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
:root {
  --bg: #0f0f13;
  --surface: #1a1a22;
  --surface-alt: #222230;
  --border: #2a2a38;
  --text: #e4e4ec;
  --text-muted: #8888a0;
  --accent-red: #ef4444;
  --accent-orange: #f97316;
  --accent-yellow: #eab308;
  --accent-green: #22c55e;
  --accent-blue: #3b82f6;
}
* { box-sizing: border-box; }
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  margin: 0; padding: 0;
  background: var(--bg); color: var(--text);
  line-height: 1.6;
}
.page { max-width: 960px; margin: 0 auto; padding: 2.5rem 2rem; }

/* Header */
.header { border-bottom: 2px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem; }
.header h1 { font-size: 1.6rem; font-weight: 800; margin: 0 0 0.4rem 0; letter-spacing: -0.02em; }
.header h1 span { color: var(--accent-yellow); }
.header .meta { color: var(--text-muted); font-size: 0.85rem; }
.header .meta b { color: var(--text); }

/* Stats row */
.stats { display: flex; gap: 1rem; margin: 1.2rem 0 0 0; }
.stat-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 0.8rem 1.2rem; flex: 1; text-align: center;
}
.stat-card .value { font-size: 1.5rem; font-weight: 700; }
.stat-card .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-top: 0.15rem; }
.stat-red .value { color: var(--accent-red); }
.stat-green .value { color: var(--accent-green); }
.stat-blue .value { color: var(--accent-blue); }

/* Map */
.map-section { margin: 2rem 0; }
.map-section h2 { font-size: 1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin: 0 0 0.8rem 0; }

/* Spill section */
.spill-section {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.5rem; margin: 1.5rem 0;
}
.spill-section h2 {
  font-size: 1.1rem; font-weight: 700; margin: 0 0 1rem 0;
  padding-bottom: 0.6rem; border-bottom: 1px solid var(--border);
}
.spill-section h3 {
  font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--text-muted); margin: 1.2rem 0 0.5rem 0;
}

/* Summary table */
.summary-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
.summary-table th {
  text-align: left; font-size: 0.78rem; color: var(--text-muted); font-weight: 500;
  padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); width: 40%;
}
.summary-table td {
  padding: 0.4rem 0.6rem; font-size: 0.88rem; border-bottom: 1px solid var(--border);
}

/* Status badges */
.badge {
  display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.55rem;
  border-radius: 6px; text-transform: uppercase; letter-spacing: 0.04em;
}
.badge-active { background: rgba(239,68,68,0.15); color: var(--accent-red); border: 1px solid rgba(239,68,68,0.3); }
.badge-possible { background: rgba(234,179,8,0.15); color: var(--accent-yellow); border: 1px solid rgba(234,179,8,0.3); }
.badge-rejected { background: rgba(136,136,160,0.1); color: var(--text-muted); border: 1px solid rgba(136,136,160,0.2); }

/* Rules list */
.rules-list { list-style: none; padding: 0; margin: 0.5rem 0; }
.rules-list li {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.4rem 0.6rem; border-radius: 6px; margin-bottom: 0.3rem; font-size: 0.85rem;
}
.rule-pass { background: rgba(34,197,94,0.08); color: var(--accent-green); }
.rule-fail { background: rgba(239,68,68,0.08); color: var(--accent-red); }
.rule-icon { font-size: 0.9rem; width: 1.2rem; text-align: center; flex-shrink: 0; }
.rule-name { font-weight: 600; }
.rule-reason { color: var(--text-muted); font-size: 0.8rem; margin-left: auto; }

/* Vessel table */
.vessel-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
.vessel-table th {
  text-align: left; font-size: 0.72rem; color: var(--text-muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
  padding: 0.5rem 0.6rem; border-bottom: 2px solid var(--border);
}
.vessel-table td {
  padding: 0.5rem 0.6rem; font-size: 0.85rem; border-bottom: 1px solid var(--border);
  vertical-align: middle;
}
.vessel-table tr:hover { background: var(--surface-alt); }

.rank-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 1.5rem; height: 1.5rem; border-radius: 50%; font-size: 0.7rem; font-weight: 700;
}
.rank-1 { background: rgba(234,179,8,0.2); color: var(--accent-yellow); border: 1px solid rgba(234,179,8,0.3); }
.rank-2 { background: rgba(136,136,160,0.15); color: var(--text); border: 1px solid rgba(136,136,160,0.2); }
.rank-3 { background: rgba(249,115,22,0.15); color: var(--accent-orange); border: 1px solid rgba(249,115,22,0.2); }
.rank-n { background: var(--surface-alt); color: var(--text-muted); border: 1px solid var(--border); }

.score-bar {
  display: inline-block; height: 0.35rem; border-radius: 2px; min-width: 1.5rem;
}
.score-high { background: var(--accent-green); }
.score-med { background: var(--accent-yellow); }
.score-low { background: var(--accent-red); }
.score-text { font-size: 0.8rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-left: 0.3rem; }

/* Drift table */
.drift-table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; }
.drift-table th {
  text-align: left; font-size: 0.78rem; color: var(--text-muted); font-weight: 500;
  padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border);
}
.drift-table td {
  padding: 0.4rem 0.6rem; font-size: 0.85rem; border-bottom: 1px solid var(--border);
}
.drift-horizon {
  display: inline-block; background: rgba(249,115,22,0.12); color: var(--accent-orange);
  font-size: 0.75rem; font-weight: 700; padding: 0.1rem 0.45rem; border-radius: 4px;
}

/* Footer */
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.75rem; text-align: center; }

.no-data { color: var(--text-muted); font-style: italic; font-size: 0.85rem; padding: 0.5rem 0; }

.mono { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.meta-field { font-size: 0.85rem; color: var(--text-muted); margin: 0.1rem 0; }
.meta-field b { color: var(--text); }
.map-frame {
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
  margin: 0.8rem 0;
}
.map-frame iframe { width: 100%; height: 480px; border: none; }

@media print {
  * { transition: none !important; }
  body { background: white; color: #111; }
  .page { max-width: 100%; padding: 1rem; }
  .spill-section { break-inside: avoid; }
  .header { border-bottom-color: #ccc; }
  .header h1 span { color: #b45309; }
  .stat-card { border-color: #ddd; background: #f9f9f9; }
  .stat-card .label { color: #666; }
  .badge { border: 1px solid currentColor; }
  .map-frame { border-color: #ccc; }
}
"""


class ReportBuildError(RuntimeError):
    """A combined pipeline result could not be turned into a report."""


def _fmt(value: Optional[float], digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return '<span class="mono">unknown</span>'
    return f'<span class="mono">{value:.{digits}f}</span>{suffix}'


def _status_badge(status: Optional[str]) -> str:
    s = (status or "unknown").lower()
    cls = {
        "active": "badge-active",
        "new": "badge-active",
        "possible": "badge-possible",
        "rejected": "badge-rejected",
        "none": "badge-rejected",
    }.get(s, "badge-rejected")
    return f'<span class="badge {cls}">{html_lib.escape(s)}</span>'


def _spill_summary_table(spill: Dict[str, Any], alert: Dict[str, Any]) -> str:
    props = spill.get("properties") or {}
    rows = [
        ("Spill ID", f'<span class="mono">{html_lib.escape(str(props.get("spill_id", "unknown")))}</span>'),
        ("Scene ID", props.get("scene_id", "unknown")),
        ("Acquisition time", f'<span class="mono">{html_lib.escape(str(props.get("acquisition_timestamp") or "unknown"))}</span>'),
        ("Location", f"{_fmt(props.get('centroid_lon'), 5)}, {_fmt(props.get('centroid_lat'), 5)}"),
        ("Area", _fmt(props.get("area_km2"), 3, " km\u00b2")),
        ("Mean confidence", _fmt(props.get("mean_confidence"), 3)),
        ("Elongation", _fmt(props.get("elongation"), 2)),
        ("Estimated volume", _fmt(props.get("estimated_volume_m3"), 1, " m\u00b3")),
        ("Alert status", _status_badge(alert.get("status"))),
    ]
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f'<table class="summary-table">{body}</table>'


def _alert_rules_list(alert: Dict[str, Any]) -> str:
    rules = alert.get("rules") or []
    if not rules:
        return '<p class="no-data">No rule evaluations recorded.</p>'
    items = []
    for r in rules:
        passed = r.get("passed", False)
        icon = "\u2713" if passed else "\u2717"
        cls = "rule-pass" if passed else "rule-fail"
        name = html_lib.escape(str(r.get("name", "")))
        reason = html_lib.escape(str(r.get("reason", "")))
        items.append(
            f'<li class="{cls}">'
            f'<span class="rule-icon">{icon}</span>'
            f'<span class="rule-name">{name}</span>'
            f'<span class="rule-reason">{reason}</span>'
            f"</li>"
        )
    return f'<ul class="rules-list">{"".join(items)}</ul>'


def _vessel_table(vessels: List[Dict[str, Any]]) -> str:
    if not vessels:
        return '<p class="no-data">No candidate vessels attributed to this spill.</p>'
    header = (
        '<tr><th style="width:3rem">Rank</th><th>MMSI</th><th>Name</th><th>Type</th>'
        '<th style="width:140px">Score</th><th>CPA</th><th>Explanation</th></tr>'
    )
    rows = []
    for rank, v in enumerate(vessels):
        score = v.get("score") or 0
        score_pct = score * 100
        if score >= 0.7:
            bar_cls, txt_cls = "score-high", "color:var(--accent-green)"
        elif score >= 0.4:
            bar_cls, txt_cls = "score-med", "color:var(--accent-yellow)"
        else:
            bar_cls, txt_cls = "score-low", "color:var(--accent-red)"

        rank_num = rank + 1
        if rank_num == 1:
            rank_cls = "rank-1"
        elif rank_num == 2:
            rank_cls = "rank-2"
        elif rank_num == 3:
            rank_cls = "rank-3"
        else:
            rank_cls = "rank-n"

        rows.append(
            "<tr>"
            f'<td><span class="rank-badge {rank_cls}">{rank_num}</span></td>'
            f"<td><span class=\"mono\">{html_lib.escape(str(v.get('mmsi', 'unknown')))}</span></td>"
            f"<td>{html_lib.escape(str(v.get('vessel_name') or 'unknown'))}</td>"
            f"<td>{html_lib.escape(str(v.get('vessel_type') or 'unknown'))}</td>"
            f'<td>'
            f'<span class="score-bar {bar_cls}" style="width:{score_pct:.0f}%"></span>'
            f'<span class="score-text" style="{txt_cls}">{_fmt(v.get("score"), 3)}</span>'
            f"</td>"
            f"<td>{_fmt(v.get('cpa_distance_km'), 2, ' km')}</td>"
            f'<td style="font-size:0.82rem;color:var(--text-muted)">{html_lib.escape(str(v.get("explanation", "")))}</td>'
            "</tr>"
        )
    return f'<table class="vessel-table">{header}{"".join(rows)}</table>'


def _drift_forecast_table(forecast: List[Dict[str, Any]]) -> str:
    if not forecast:
        return '<p class="no-data">No drift forecast available for this spill.</p>'
    header = "<tr><th>Horizon</th><th>Estimated time</th></tr>"
    rows = []
    for f in forecast:
        hours = f.get("hours_elapsed") or f.get("hours") or 0
        time_str = html_lib.escape(str(f.get("time", "unknown")))
        rows.append(
            f"<tr>"
            f'<td><span class="drift-horizon">+{hours:.0f}h</span></td>'
            f"<td>{time_str}</td>"
            f"</tr>"
        )
    return f'<table class="drift-table">{header}{"".join(rows)}</table>'


def _spill_section(entry: Dict[str, Any], index: int) -> str:
    spill = entry["spill"]
    alert = entry.get("alert") or {}
    vessels = entry.get("vessels") or []
    forecast = entry.get("drift_forecast") or []
    props = spill.get("properties") or {}
    spill_id = html_lib.escape(str(props.get("spill_id", "unknown")))
    status = (alert.get("status") or "none").lower()
    accent = {"active": "var(--accent-red)", "new": "var(--accent-red)",
              "possible": "var(--accent-yellow)"}.get(status, "var(--text-muted)")

    return f"""
    <section class="spill-section">
      <h2 style="border-left:3px solid {accent}; padding-left:0.8rem;">
        Spill {index + 1}: {spill_id}
      </h2>
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
    alerted = sum(
        1 for e in spills
        if (e.get("alert") or {}).get("status") in ("active", "new", "possible")
    )
    total_area = sum(
        (e.get("spill", {}).get("properties") or {}).get("area_km2") or 0
        for e in spills
    )
    vessel_count = sum(len(e.get("vessels") or []) for e in spills)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Incident report - {scene_id}</title>
<style>{STYLE}</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>SAR<span>Oil</span>Spill &mdash; Incident Report</h1>
    <p class="meta">
      <span class="meta-field">Scene <b>{scene_id}</b></span> &middot; <span class="meta-field">Generated {generated_at}</span>
    </p>
    <div class="stats">
      <div class="stat-card stat-red">
        <div class="value">{len(spills)}</div>
        <div class="label">Spills Detected</div>
      </div>
      <div class="stat-card stat-green">
        <div class="value">{alerted}</div>
        <div class="label">Alerted</div>
      </div>
      <div class="stat-card stat-blue">
        <div class="value">{total_area:.2f}</div>
        <div class="label">Total Area (km&sup2;)</div>
      </div>
    </div>
  </div>
  <div class="map-section">
    <h2>Map Overview</h2>
    <div class="map-frame">
      {map_embed}
    </div>
  </div>
  {sections}
  <div class="footer">
    SAR Oil Spill Detection &amp; Attribution System &middot; Smart India Hackathon
  </div>
</div>
</body>
</html>
""" + _MAP_DARK_INJECT


def save_report(result: Dict[str, Any], output_path: Path) -> Path:
    """Build the report and save it as a standalone HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report_html(result), encoding="utf-8")
    return output_path


__all__ = ["ReportBuildError", "build_report_html", "save_report"]

_MAP_DARK_INJECT = """<script>
(function() {
  var iframes = document.querySelectorAll('iframe');
  for (var i = 0; i < iframes.length; i++) {
    var iframe = iframes[i];
    iframe.onload = function() {
      try {
        var doc = iframe.contentDocument || iframe.contentWindow.document;
        var s = doc.createElement('style');
        s.textContent = 'body{background:#0f0f13!important;margin:0;}' +
          '.leaflet-control{background:#1a1a22!important;color:#e4e4ec!important;border-color:#2a2a38!important;}' +
          '.leaflet-control-layers label{color:#e4e4ec!important;}' +
          '.leaflet-control-layers-toggle{background-color:#1a1a22!important;border-color:#2a2a38!important;}' +
          '.leaflet-control-zoom a{background:#1a1a22!important;color:#e4e4ec!important;border-color:#2a2a38!important;}' +
          '.leaflet-control-attribution{background:rgba(15,15,19,0.8)!important;color:#8888a0!important;}' +
          '.leaflet-control-attribution a{color:#3b82f6!important;}';
        doc.head.appendChild(s);
      } catch(e) {}
    };
  }
})();
</script>"""
