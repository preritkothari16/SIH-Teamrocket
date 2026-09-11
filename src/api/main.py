"""FastAPI application for the SAR oil-spill dashboard.

Endpoints
---------
GET  /api/runs              — list all processed runs (summary)
GET  /api/runs/{id}         — full PipelineRun contract for one scene
GET  /api/runs/{id}/report  — Step 5.3 report file (404 if not generated)
POST /api/runs              — trigger detection on a scene (sync, hackathon)
POST /api/runs/{id}/ask     — Step 8.1 natural-language Q&A over a run
GET  /api/regions           — Step 8.4 demo region presets
GET  /api/model/info        — Step 8.3 model card ({trained: false} if untrained)

CORS is configured for the Vite dev server at http://localhost:5173.

Usage::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.models import (
    AskRequest, AskResponse, PipelineRun, Region, RunRequest, RunSummary,
)
from src.api.registry import get_report_path, get_run, list_runs
from src.attribution.qa import QAError, answer_question
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_REGIONS_PATH = REPO_ROOT / "configs" / "demo_regions.yaml"

app = FastAPI(
    title="SAR Oil Spill API",
    description="Backend API for the SAR oil-spill detection dashboard",
    version="0.1.0",
)

# CORS — the local Vite dev server always, plus whatever the deployed
# frontend's origin is (SAROIL_API__CORS_ORIGINS, e.g. a Vercel URL) — set
# that as a Render env var, no code change/redeploy needed when it changes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://sih-teamrocket.vercel.app",
        *get_settings().api.extra_cors_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Demo regions (Step 8.4) — a fixed, committed YAML file, not settings/DB
# backed, so a separate function (not src/api/registry.py) is enough. Kept
# as its own indirection (not inlined into the two endpoints below) purely
# so tests can monkeypatch it without touching the real committed file.
# --------------------------------------------------------------------------- #
def _load_regions() -> List[Dict[str, Any]]:
    if not DEMO_REGIONS_PATH.is_file():
        return []
    with DEMO_REGIONS_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("regions") or []


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/regions", response_model=list[Region])
def api_list_regions() -> list[Region]:
    """The configured demo region presets — most are pending (scene_id:
    null) until a real local scene backs them; see configs/demo_regions.yaml."""
    return [Region(**r) for r in _load_regions()]


@app.get("/api/runs", response_model=list[RunSummary])
def api_list_runs(region: Optional[str] = None) -> list[RunSummary]:
    """List all registered runs with summary fields.

    ``region`` (a :class:`Region`'s ``id``) filters the list down to that
    region's own ``scene_id`` when it resolves to one; an unset, unknown, or
    still-pending (``scene_id: null``) region leaves the list unfiltered —
    the frontend never sends one of those (pending chips render disabled),
    so this is a soft, best-effort filter, not a hard resource lookup.
    """
    runs = list_runs()
    if region:
        target_scene_id = next(
            (r.get("scene_id") for r in _load_regions() if r.get("id") == region), None,
        )
        if target_scene_id:
            runs = [r for r in runs if r.get("scene_id") == target_scene_id]
    return [RunSummary(**r) for r in runs]


@app.get("/api/runs/{scene_id}", response_model=PipelineRun)
def api_get_run(scene_id: str) -> PipelineRun:
    """Full combined output for one run, matching the frontend contract."""
    run = get_run(scene_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{scene_id}' not found")
    return PipelineRun(**run)


@app.get("/api/runs/{scene_id}/report")
def api_get_report(scene_id: str):
    """Return the Step 5.3 standalone HTML incident report, 404 otherwise."""
    report = get_report_path(scene_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report for '{scene_id}' — run the pipeline with --report first",
        )
    return FileResponse(
        report,
        media_type="text/html",
        filename=f"{scene_id}_report.html",
    )


@app.post("/api/runs", response_model=PipelineRun)
def api_trigger_run(request: RunRequest) -> PipelineRun:
    """Trigger detection on a scene.

    Runs synchronously — fine for a hackathon demo, should become a background
    task for anything beyond that.
    """
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py")]

    if request.scene_path:
        cmd.extend(["--scene", request.scene_path])
    elif request.scene_id:
        cmd.extend(["--scene-id", request.scene_id])
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either scene_path or scene_id",
        )

    if request.stub_model:
        cmd.append("--stub-model")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pipeline timed out (300s)")

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed:\n{result.stderr[-2000:]}",
        )

    # The pipeline wrote the geojson; load it via the registry.
    scene_id = request.scene_id
    if not scene_id and request.scene_path:
        scene_id = Path(request.scene_path).stem

    run = get_run(scene_id)
    if run is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed but no output found",
        )

    return PipelineRun(**run)


@app.post("/api/runs/{scene_id}/ask", response_model=AskResponse)
def api_ask_run(scene_id: str, request: AskRequest) -> AskResponse:
    """Answer a natural-language question about one run (Step 8.1).

    Purely explanatory — grounded in that run's already-computed spill/
    alert/vessel data, no new detection or scoring. 503 (not 500) when the
    LLM itself can't be reached, since that's a missing package/credential,
    not this run's data being broken.
    """
    run = get_run(scene_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{scene_id}' not found")
    try:
        result = answer_question(run, request.question)
    except QAError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return AskResponse(answer=result.answer, cited_vessels=result.cited_vessels)


@app.get("/api/model/info")
def api_model_info() -> Dict[str, Any]:
    """Step 8.3 model card: ``src/detection/train.py``'s ``best_metrics.json``,
    written alongside ``models/best.pt`` every time training beats its own
    previous best.

    Returns ``{"trained": false}`` — not a 404 or a 500 — when that file
    doesn't exist, which is the current repo state (no checkpoint has been
    trained here yet, per CLAUDE.md). A corrupt/unreadable file degrades the
    same way rather than 500ing: "not trained" is the honest fallback either
    way, not a server error.
    """
    settings = get_settings()
    metrics_path = settings.paths.resolve(settings.paths.models_dir) / "best_metrics.json"
    if not metrics_path.is_file():
        return {"trained": False}
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable %s: %s", metrics_path, exc)
        return {"trained": False}
    metrics["trained"] = True
    return metrics


@app.get("/api/health")
def health():
    """Health check."""
    settings = get_settings()
    spills_dir = settings.paths.resolve(
        settings.paths.processed_dir / settings.characterization.spills_dirname
    )
    return {
        "status": "ok",
        "spills_dir": str(spills_dir),
        "spills_dir_exists": spills_dir.exists(),
    }
