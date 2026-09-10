"""Natural-language Q&A over one already-computed pipeline run.

No new detection/scoring logic lives here - src/attribution/scoring.py
(Step 2.3) already produces, per candidate vessel, a score and a
human-readable explanation string. This module only explains that existing
output: :func:`build_context` serialises one run's spill + ranked vessels +
their explanation strings (reused verbatim, never regenerated) into a
compact text block, and :func:`answer_question` sends that block plus the
caller's question to an LLM instructed to answer only from it.

``run`` throughout is the same ``{spill, alert, vessels, drift}`` dict shape
``src/api/registry.py::get_run()`` already returns (matching
``src/api/models.py::PipelineRun`` field-for-field) - not the raw multi-spill
``pipeline_result.json`` - so this module stays a plain dict consumer with no
import dependency on ``src.api``. That also means a Postgres-backed run (see
``src/db.py``) has no vessels to cite: that table never stores them, by that
module's own design, not a gap this step fixes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config import Settings, get_settings

#: Anthropic model used for grounded Q&A - cheap/fast is fine here, the
#: whole point is explaining numbers that already exist, not reasoning hard.
DEFAULT_MODEL = "claude-sonnet-5"

_SYSTEM_PROMPT = (
    "You answer questions about one SAR oil-spill detection run, using ONLY "
    "the CONTEXT block you are given below the question. Never use outside "
    "knowledge and never guess - if the context does not cover the "
    "question, say so explicitly instead of speculating.\n\n"
    "End every answer with exactly one final line in this form:\n"
    "CITED: <mmsi1>,<mmsi2>\n"
    "listing the MMSI(s) of the vessel(s) your answer actually drew from, "
    "or:\n"
    "CITED: none\n"
    "if your answer did not draw from any specific vessel."
)

_CITED_LINE_RE = re.compile(r"^CITED:\s*(.*)$", re.IGNORECASE)


class QAError(RuntimeError):
    """The LLM could not be called - message names the missing package or
    credential and how to fix it, per this project's error convention."""


@dataclass
class QAAnswer:
    """An answer grounded in one run's context, plus which vessel(s) it
    drew from (for citation in the UI)."""

    answer: str
    cited_vessels: List[str] = field(default_factory=list)


def build_context(run: Dict[str, Any]) -> str:
    """A compact, structured text summary of one run's spill, alert, and
    every ranked vessel's score + explanation string (verbatim, not
    regenerated).

    ``run["vessels"]`` is assumed already ranked highest-score-first, the
    order :func:`src.attribution.scoring.score_candidates` produces and every
    caller of it (``scripts/run_pipeline.py``, ``src/api/registry.py``)
    preserves end to end.
    """
    spill = run.get("spill", {}) or {}
    alert = run.get("alert", {}) or {}
    vessels = run.get("vessels", []) or []

    lines = [
        "SPILL:",
        f"  scene_id: {spill.get('scene_id')}",
        f"  acquisition_timestamp: {spill.get('acquisition_timestamp')}",
        f"  area_km2: {spill.get('area_km2')}",
        f"  confidence: {spill.get('confidence')}",
        f"  centroid: {spill.get('centroid')}",
        f"  major_axis_bearing: {spill.get('major_axis_bearing')}",
        f"  elongation: {spill.get('elongation')}",
        "",
        "ALERT:",
        f"  status: {alert.get('status')}",
        f"  rules_fired: {alert.get('rules_fired')}",
        "",
        f"RANKED VESSELS ({len(vessels)}, highest score first):",
    ]
    if not vessels:
        lines.append(
            "  (none - either no AIS export was given for this run, or no "
            "candidate vessel passed the AIS search/filter)"
        )
    for rank, v in enumerate(vessels, start=1):
        lines.append(
            f"  {rank}. mmsi={v.get('mmsi')} name={v.get('name') or 'unknown'} "
            f"type={v.get('vessel_type') or 'unknown'} score={v.get('score')} "
            f"cpa_distance_km={v.get('cpa_distance_km')} cpa_time={v.get('cpa_time')}"
        )
        lines.append(f"     explanation: {v.get('explanation')}")

    return "\n".join(lines)


def _parse_cited(raw_text: str) -> Tuple[str, List[str]]:
    """Split the model's trailing ``CITED: ...`` line from the answer body."""
    lines = raw_text.strip().splitlines()
    if not lines:
        return raw_text.strip(), []
    match = _CITED_LINE_RE.match(lines[-1].strip())
    if not match:
        return raw_text.strip(), []
    cited_raw = match.group(1).strip()
    body = "\n".join(lines[:-1]).strip()
    if not cited_raw or cited_raw.lower() == "none":
        return body, []
    return body, [m.strip() for m in cited_raw.split(",") if m.strip()]


def _resolve_api_key(settings: Settings) -> Optional[str]:
    return settings.credentials.resolve().get("anthropic_api_key")


def _call_llm(context: str, question: str, model: str, api_key: str) -> str:
    """The actual Anthropic API call, isolated in its own function so tests
    can mock it without touching :func:`build_context`'s serialisation or
    :func:`answer_question`'s ``CITED:`` parsing."""
    try:
        import anthropic
    except ImportError as exc:
        raise QAError(
            "anthropic package not installed - pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def answer_question(
    run: Dict[str, Any],
    question: str,
    model: str = DEFAULT_MODEL,
    settings: Optional[Settings] = None,
) -> QAAnswer:
    """Answer ``question`` about ``run``, grounded only in its own context.

    Raises :class:`QAError` if the LLM can't be reached - missing
    ``anthropic`` package or missing ``ANTHROPIC_API_KEY`` (see
    ``.env.example``); the message names which.
    """
    settings = settings or get_settings()
    api_key = _resolve_api_key(settings)
    if not api_key:
        raise QAError(
            "ANTHROPIC_API_KEY not set - copy .env.example to .env and fill it in"
        )

    context = build_context(run)
    raw = _call_llm(context, question, model, api_key)
    body, cited = _parse_cited(raw)
    return QAAnswer(answer=body, cited_vessels=cited)


__all__ = ["QAAnswer", "QAError", "DEFAULT_MODEL", "build_context", "answer_question"]
