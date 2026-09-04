"""Segmentation model factory.

Deliberately thin. ``train.py`` and the later ``infer.py`` both call
:func:`get_model`, so the architecture is defined in exactly one place and a
checkpoint can never be loaded into a differently-shaped network by accident.

Defaults come from the config: U-Net with a ResNet34 encoder and 5 output
classes, matching the MKLab label scheme.

Example::

    from src.detection.model import get_model

    model = get_model(num_classes=5)
    logits = model(batch)          # (N, 5, H, W)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
from torch import nn

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


def get_model(
    num_classes: int = 5,
    arch: Optional[str] = None,
    encoder: Optional[str] = None,
    encoder_weights: Optional[str] = "imagenet",
    in_channels: Optional[int] = None,
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> nn.Module:
    """Build the segmentation network.

    Args:
        num_classes: output channels; 5 for the MKLab scheme.
        arch: ``unet`` by default, from ``detection.model_arch``.
        encoder: backbone name, from ``detection.encoder``.
        encoder_weights: ``"imagenet"`` for pretrained weights (downloads on
            first use), or ``None`` to start from scratch - which is also what
            keeps tests and offline runs from touching the network.
        in_channels: input channels; ``smp`` adapts pretrained first-layer
            weights when this is not 3, so single-band SAR works too.

    Returns the model on the CPU; the caller moves it to a device.
    """
    settings = settings or get_settings()
    arch = arch or settings.detection.model_arch
    encoder = encoder or settings.detection.encoder
    in_channels = in_channels or settings.training.in_channels

    model = smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
    logger.info(
        "built %s/%s: %d in, %d classes, weights=%s",
        arch, encoder, in_channels, num_classes, encoder_weights,
    )
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """(total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write weights plus the metadata needed to rebuild the same network."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "model": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path,
    model: Optional[nn.Module] = None,
    map_location: Any = "cpu",
    settings: Optional[Settings] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`.

    Uses the architecture recorded in the checkpoint, so a run trained with a
    different encoder still loads correctly.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is None:
        spec = payload.get("model", {})
        model = get_model(
            num_classes=spec.get("num_classes", 5),
            arch=spec.get("arch"),
            encoder=spec.get("encoder"),
            encoder_weights=None,  # weights come from the checkpoint
            in_channels=spec.get("in_channels"),
            settings=settings,
        )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


__all__ = ["get_model", "count_parameters", "save_checkpoint", "load_checkpoint"]
