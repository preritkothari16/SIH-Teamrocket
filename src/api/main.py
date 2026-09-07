"""FastAPI application for the SAR oil-spill dashboard.

Endpoints
---------
GET  /api/runs              — list all processed runs (summary)
GET  /api/runs/{id}         — full PipelineRun contract for one scene
GET  /api/runs/{id}/report  — Step 5.3 report file (404 if not generated)
POST /api/runs              — trigger detection on a scene (sync, hackathon)

CORS is configured for the Vite dev server at http://localhost:5173.

Usage::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api.models import PipelineRun, RunRequest, RunSummary
from src.api.registry import get_report_path, get_run, list_runs
from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="SAR Oil Spill API",
    description="Backend API for the SAR oil-spill detection dashboard",
    version="0.1.0",
)

# CORS — allow the Vite dev server and any local origin for hackathon demos.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/runs", response_model=list[RunSummary])
def api_list_runs() -> list[RunSummary]:
    """List all registered runs with summary fields."""
    runs = list_runs()
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
    """Return the Step 5.3 report file if it exists, 404 otherwise."""
    report = get_report_path(scene_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No report for '{scene_id}' — run the pipeline first",
        )
    return FileResponse(
        report,
        media_type="application/json",
        filename=f"{scene_id}_report.json",
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
