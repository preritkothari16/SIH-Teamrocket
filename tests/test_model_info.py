"""Tests for GET /api/model/info (Step 8.3).

Two states only: models/best_metrics.json present (real metrics returned)
or absent ({"trained": false}, never a 404/500) - the latter is the current
repo state, since no checkpoint has ever been trained here (CLAUDE.md:
"1.3 done, never trained - no GPU, no dataset here").
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import Settings

client = TestClient(app)


def _make_settings(tmp_path: Path) -> Settings:
    """Build a Settings rooted at tmp_path's models/ dir."""
    return Settings(_env_file=None, paths={"models_dir": str(tmp_path / "models")})


@pytest.fixture()
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the endpoint at a temporary models/ directory."""
    d = tmp_path / "models"
    d.mkdir(parents=True)
    from src.api import main as main_mod

    monkeypatch.setattr(main_mod, "get_settings", lambda: _make_settings(tmp_path))
    return d


SAMPLE_METRICS = {
    "architecture": {
        "arch": "unet",
        "encoder": "resnet34",
        "in_channels": 3,
        "num_classes": 5,
        "class_names": ["sea", "oil_spill", "look_alike", "ship", "land"],
        "image_size": 512,
    },
    "train_samples": 800,
    "val_samples": 140,
    "mean_iou": 0.71,
    "oil_iou": 0.62,
    "look_alike_iou": 0.55,
    "precision": 0.74,
    "recall": 0.68,
    "dice": 0.71,
    "oil_as_lookalike_rate": 0.08,
    "lookalike_as_oil_rate": 0.05,
    "trained_at": "2026-09-11T00:00:00+00:00",
}


class TestModelInfoUntrained:
    def test_returns_trained_false_when_no_checkpoint(self, models_dir: Path) -> None:
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        assert resp.json() == {"trained": False}

    def test_returns_trained_false_on_a_corrupt_file_not_a_500(
        self, models_dir: Path
    ) -> None:
        (models_dir / "best_metrics.json").write_text("not valid json{{{", encoding="utf-8")
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        assert resp.json() == {"trained": False}


class TestModelInfoTrained:
    def test_returns_the_real_metrics(self, models_dir: Path) -> None:
        (models_dir / "best_metrics.json").write_text(
            json.dumps(SAMPLE_METRICS), encoding="utf-8"
        )
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        body = resp.json()

        assert body["trained"] is True
        assert body["architecture"]["arch"] == "unet"
        assert body["architecture"]["encoder"] == "resnet34"
        assert body["train_samples"] == 800
        assert body["val_samples"] == 140
        assert body["mean_iou"] == pytest.approx(0.71)
        assert body["oil_iou"] == pytest.approx(0.62)
        assert body["look_alike_iou"] == pytest.approx(0.55)
        assert body["precision"] == pytest.approx(0.74)
        assert body["recall"] == pytest.approx(0.68)
        assert body["dice"] == pytest.approx(0.71)
        assert body["oil_as_lookalike_rate"] == pytest.approx(0.08)
        assert body["lookalike_as_oil_rate"] == pytest.approx(0.05)
        assert body["trained_at"] == "2026-09-11T00:00:00+00:00"
