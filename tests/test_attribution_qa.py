"""Tests for src.attribution.qa (Step 8.1: attribution Q&A).

The LLM call itself (src.attribution.qa._call_llm) is mocked throughout —
these tests assert build_context() carries every ranked vessel's real
explanation string verbatim, and that answer_question() wires context +
question through to the (mocked) model and parses its trailing
``CITED: ...`` line correctly. Fully offline, like the rest of the suite.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.attribution import qa
from src.config import Settings


RUN: Dict[str, Any] = {
    "spill": {
        "scene_id": "scene_001",
        "acquisition_timestamp": "2024-04-10T14:20:00+00:00",
        "confidence": 0.82,
        "area_km2": 8.3,
        "centroid": {"lat": 24.11, "lon": -89.99},
        "major_axis_bearing": 45.0,
        "elongation": 2.1,
    },
    "alert": {
        "spill_id": "scene_001_spill_001",
        "status": "new",
        "rules_fired": [],
    },
    "vessels": [
        {
            "mmsi": "111111111",
            "name": "Chemical Pioneer",
            "vessel_type": "tanker",
            "score": 0.91,
            "explanation": "1.2 km CPA, 5h before image, track within 8° of slick axis, tanker",
            "cpa_distance_km": 1.2,
            "cpa_time": "2024-04-10T09:00:00+00:00",
        },
        {
            "mmsi": "222222222",
            "name": "Ocean Rambler",
            "vessel_type": "cargo",
            "score": 0.44,
            "explanation": "9.6 km CPA, 30h before image, heading unknown, cargo",
            "cpa_distance_km": 9.6,
            "cpa_time": "2024-04-08T20:00:00+00:00",
        },
    ],
}

RUN_NO_VESSELS: Dict[str, Any] = {
    "spill": RUN["spill"],
    "alert": {"spill_id": "scene_002_spill_001", "status": "possible", "rules_fired": ["wind_out_of_range"]},
    "vessels": [],
}


# --------------------------------------------------------------------------- #
# build_context
# --------------------------------------------------------------------------- #
def test_build_context_includes_every_vessel_explanation_verbatim() -> None:
    context = qa.build_context(RUN)
    for vessel in RUN["vessels"]:
        assert vessel["explanation"] in context
        assert vessel["mmsi"] in context

    assert "scene_001" in context
    assert "new" in context


def test_build_context_handles_no_vessels() -> None:
    context = qa.build_context(RUN_NO_VESSELS)
    assert "none" in context.lower()


# --------------------------------------------------------------------------- #
# answer_question — LLM call mocked
# --------------------------------------------------------------------------- #
def test_answer_question_is_answerable_purely_from_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A question about the top-ranked vessel should be answerable because
    build_context() put that vessel's own explanation string into the
    context the (mocked) LLM is handed."""
    captured: Dict[str, str] = {}

    def fake_call_llm(context: str, question: str, model: str, api_key: str) -> str:
        captured["context"] = context
        captured["question"] = question
        assert RUN["vessels"][0]["explanation"] in context
        return (
            "Chemical Pioneer (MMSI 111111111) is ranked first: 1.2 km CPA, "
            "5h before image, track within 8° of slick axis, tanker.\n"
            "CITED: 111111111"
        )

    monkeypatch.setattr(qa, "_call_llm", fake_call_llm)
    monkeypatch.setattr(qa, "_resolve_api_key", lambda settings: "dummy-key")

    result = qa.answer_question(
        RUN, "why is vessel 111111111 ranked first?", settings=Settings()
    )

    assert "Chemical Pioneer" in result.answer
    assert "CITED:" not in result.answer  # stripped from the returned body
    assert result.cited_vessels == ["111111111"]
    assert captured["question"] == "why is vessel 111111111 ranked first?"


def test_answer_question_parses_cited_none() -> None:
    def fake_call_llm(context: str, question: str, model: str, api_key: str) -> str:
        return "The context doesn't mention that.\nCITED: none"

    result = _answer_with_mocked_llm(fake_call_llm)
    assert result.cited_vessels == []
    assert "doesn't mention" in result.answer


def test_answer_question_parses_multiple_cited_vessels() -> None:
    def fake_call_llm(context: str, question: str, model: str, api_key: str) -> str:
        return "Both vessels passed close by.\nCITED: 111111111, 222222222"

    result = _answer_with_mocked_llm(fake_call_llm)
    assert result.cited_vessels == ["111111111", "222222222"]


def test_answer_question_raises_qa_error_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qa, "_resolve_api_key", lambda settings: None)
    with pytest.raises(qa.QAError, match="ANTHROPIC_API_KEY"):
        qa.answer_question(RUN, "who is responsible?", settings=Settings())


def _answer_with_mocked_llm(fake_call_llm) -> "qa.QAAnswer":
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(qa, "_call_llm", fake_call_llm)
    mp.setattr(qa, "_resolve_api_key", lambda settings: "dummy-key")
    try:
        return qa.answer_question(RUN, "some question", settings=Settings())
    finally:
        mp.undo()
