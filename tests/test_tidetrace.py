"""TideTrace detector adapter tests.

A synthetic (untrained, small-encoder) checkpoint is built for the loading/
shape/remap tests, matching this project's fully-offline convention — no
network access, no real checkpoint file required for the suite to pass.
Separately, a couple of tests are skipped unless the real downloaded
checkpoint (models/tidetrace_oil_unet_best.pt) happens to be present
locally, for an end-to-end sanity check against the actual trained weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import segmentation_models_pytorch as smp
import torch

from src.config import get_settings
from src.detection.infer import NODATA_CLASS, infer_scene
from src.detection.tidetrace import (
    DEFAULT_NORMALISATION,
    TideTraceDetector,
    TideTraceError,
    _prepare_batch_input,
    load_tidetrace_model,
    order_bands,
)

from tests.test_detection_infer import make_index, write_scene_dir

REAL_CHECKPOINT = Path("models/tidetrace_oil_unet_best.pt")


# --------------------------------------------------------------------------- #
# order_bands / preprocessing
# --------------------------------------------------------------------------- #
def test_order_bands_puts_the_darker_band_first() -> None:
    bright = np.full((16, 16), -14.0, dtype=np.float32)  # co-pol, e.g. VV
    dark = np.full((16, 16), -21.0, dtype=np.float32)    # cross-pol, e.g. VH
    # Fed in as (bright, dark) — the wrong-if-assumed order.
    stack = np.stack([bright, dark])
    ordered = order_bands(stack)
    assert np.allclose(ordered[0], dark)
    assert np.allclose(ordered[1], bright)


def test_order_bands_is_a_no_op_already_correct() -> None:
    dark = np.full((8, 8), -22.0, dtype=np.float32)
    bright = np.full((8, 8), -15.0, dtype=np.float32)
    ordered = order_bands(np.stack([dark, bright]))
    assert np.allclose(ordered[0], dark)
    assert np.allclose(ordered[1], bright)


def test_order_bands_single_band_passthrough() -> None:
    band = np.full((1, 4, 4), -18.0, dtype=np.float32)
    assert np.array_equal(order_bands(band), band)


def test_prepare_batch_input_shape_and_normalisation() -> None:
    tile = np.stack([
        np.full((32, 32), -18.0, dtype=np.float32),
        np.full((32, 32), -12.0, dtype=np.float32),
    ])
    norm = {"mean_db": -18.0, "std_db": 3.0}
    out = _prepare_batch_input(tile, norm)
    assert out.shape == (3, 32, 32)
    # Darker band (-18) is the mean -> channel 0 and 2 (repeated) are ~0.
    assert np.allclose(out[0], 0.0, atol=1e-5)
    assert np.allclose(out[2], 0.0, atol=1e-5)
    # Brighter band (-12) standardises to (−12 − −18) / 3 = 2.0.
    assert np.allclose(out[1], 2.0, atol=1e-5)


def test_prepare_batch_input_fills_nan_with_the_mean() -> None:
    tile = np.stack([
        np.full((8, 8), np.nan, dtype=np.float32),
        np.full((8, 8), -18.0, dtype=np.float32),
    ])
    out = _prepare_batch_input(tile, {"mean_db": -18.0, "std_db": 5.0})
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def _write_synthetic_checkpoint(path: Path, arch: str = "UnetPlusPlus",
                                encoder: str = "resnet18") -> None:
    """A real, tiny, untrained TideTrace-format checkpoint — small/fast
    encoder, not the real timm-efficientnet-b0, since this is testing the
    *loading mechanism*, not the real trained weights."""
    factory = {"Unet": smp.Unet, "UnetPlusPlus": smp.UnetPlusPlus}[arch]
    net = factory(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "meta": {
            "arch": arch, "encoder": encoder, "in_channels": 3, "classes": 3,
            "normalisation": {"mean_db": -20.0, "std_db": 9.0},
        },
        "metrics": {"iou_oil": 0.5},
    }, path)


def test_load_tidetrace_model_success(tmp_path: Path) -> None:
    ckpt = tmp_path / "fake_oil_unet.pt"
    _write_synthetic_checkpoint(ckpt)
    detector = load_tidetrace_model(ckpt)
    assert isinstance(detector, TideTraceDetector)
    assert detector.normalisation == {"mean_db": -20.0, "std_db": 9.0}
    assert detector.device == "cpu"


def test_load_tidetrace_model_missing_file_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(TideTraceError, match="no TideTrace checkpoint at"):
        load_tidetrace_model(tmp_path / "does_not_exist.pt")


def test_load_tidetrace_model_not_a_checkpoint_fails_clearly(tmp_path: Path) -> None:
    bad = tmp_path / "not_a_checkpoint.pt"
    torch.save({"unexpected": "shape"}, bad)
    with pytest.raises(TideTraceError, match="does not look like a TideTrace checkpoint"):
        load_tidetrace_model(bad)


def test_load_tidetrace_model_architecture_mismatch_fails_clearly(tmp_path: Path) -> None:
    """A checkpoint whose meta claims one shape but whose weights are
    another (e.g. a different encoder) must fail loudly, not silently
    load a partially-initialised network."""
    ckpt = tmp_path / "mismatched.pt"
    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=3)
    torch.save({
        "state_dict": net.state_dict(),
        # meta claims resnet34, but the weights above are resnet18's shape.
        "meta": {"arch": "Unet", "encoder": "resnet34", "in_channels": 3, "classes": 3},
        "metrics": {},
    }, ckpt)
    with pytest.raises(TideTraceError, match="does not match"):
        load_tidetrace_model(ckpt)


def test_load_tidetrace_model_unknown_architecture_fails_clearly(tmp_path: Path) -> None:
    ckpt = tmp_path / "unknown_arch.pt"
    net = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=3)
    torch.save({
        "state_dict": net.state_dict(),
        "meta": {"arch": "SomeUnrecognisedArch", "encoder": "resnet18",
                 "in_channels": 3, "classes": 3},
        "metrics": {},
    }, ckpt)
    with pytest.raises(TideTraceError, match="does not recognise"):
        load_tidetrace_model(ckpt)


# --------------------------------------------------------------------------- #
# TideTraceDetector: class remap + shape
# --------------------------------------------------------------------------- #
def test_detector_output_has_five_channels_matching_our_class_scheme(tmp_path: Path) -> None:
    ckpt = tmp_path / "fake.pt"
    _write_synthetic_checkpoint(ckpt)
    detector = load_tidetrace_model(ckpt)

    batch = torch.zeros(2, 3, 32, 32)
    logits = detector(batch)
    assert logits.shape == (2, 5, 32, 32)  # len(CLASS_NAMES) == 5


def test_detector_remaps_oil_and_lookalike_to_our_indices() -> None:
    """TideTrace: 0 sea, 1 look_alike, 2 mineral_oil.
    Ours (CLASS_NAMES): 0 sea, 1 oil_spill, 2 look_alike.
    A net that always says "confidently mineral_oil" (its class 2) must
    come out as *our* oil_spill (index 1), not index 2."""
    class AlwaysOilNet(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            n, _, h, w = batch.shape
            logits = torch.full((n, 3, h, w), -5.0)
            logits[:, 2] = 5.0  # TideTrace's own "mineral_oil" class
            return logits

    detector = TideTraceDetector(AlwaysOilNet(), DEFAULT_NORMALISATION)
    out = detector(torch.zeros(1, 3, 8, 8))
    winner = out.argmax(dim=1)
    assert (winner == 1).all()  # our OIL_CLASS, not TideTrace's raw 2


def test_detector_never_predicts_ship_or_land() -> None:
    class AnythingNet(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            n, _, h, w = batch.shape
            return torch.randn(n, 3, h, w)

    detector = TideTraceDetector(AnythingNet(), DEFAULT_NORMALISATION)
    out = detector(torch.zeros(4, 3, 16, 16))
    winner = out.argmax(dim=1)
    assert not (winner == 3).any()  # ship
    assert not (winner == 4).any()  # land


# --------------------------------------------------------------------------- #
# integration: through infer_scene with prepare_fn
# --------------------------------------------------------------------------- #
def test_infer_scene_with_tidetrace_prepare_fn(tmp_path: Path) -> None:
    frame = make_index()
    scene_dir = write_scene_dir(tmp_path, frame, fill=-18.0)
    ckpt = tmp_path / "fake.pt"
    _write_synthetic_checkpoint(ckpt)
    detector = load_tidetrace_model(ckpt)
    settings = get_settings()

    result = infer_scene(
        scene_dir, model=detector, scene_id="synthetic", settings=settings,
        save=False, prepare_fn=detector.prepare,
    )
    assert result.mask.shape == (64, 64)
    assert result.probabilities.shape[0] == 5
    covered = result.mask != NODATA_CLASS
    assert covered.all()  # every tile in this fixture is fully valid


# --------------------------------------------------------------------------- #
# real checkpoint (skipped unless it's actually present locally)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_CHECKPOINT.is_file(),
                    reason="real TideTrace checkpoint not present locally")
def test_real_checkpoint_loads_and_infers() -> None:
    detector = load_tidetrace_model(REAL_CHECKPOINT)
    assert detector.net.__class__.__name__ == "UnetPlusPlus"
    tile = np.random.default_rng(0).normal(-18, 3, size=(2, 64, 64)).astype(np.float32)
    prepared = detector.prepare(tile)
    batch = torch.from_numpy(np.stack([prepared]))
    logits = detector(batch)
    assert logits.shape == (1, 5, 64, 64)
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(1, 64, 64), atol=1e-4)
