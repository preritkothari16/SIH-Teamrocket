"""Adapter for TideTrace's trained oil-spill detector.

TideTrace (https://github.com/S-T-A-RNalin1/tidetrace) trained a real 3-class
UNet++ segmenter on the Zenodo Sentinel-1 SAR oil spill corpus — the first
real trained detector available to this project, which has never had a GPU
or labelled dataset of its own (see CLAUDE.md). This module is the whole
integration: everything else in this pipeline — ingestion, preprocessing,
the alert/AIS/drift chain, the API, the frontend — is completely unaware
TideTrace exists. It only ever sees an ``InferenceResult`` in this project's
own shape, produced through :func:`src.detection.infer.infer_scene` exactly
as the stub model always has been.

Two things are copied verbatim from TideTrace's own ``app/ml/{dataset,infer}.py``
rather than reimplemented from the README, because they encode a real,
previously-hit bug (see ``order_bands``'s own docstring) and the checkpoint's
own training statistics — guessing either one wrong silently produces a
"working" model that finds nothing, which is exactly what happened to
TideTrace itself once:

- ``order_bands``: sorts a 2-band stack by median brightness (co-pol last),
  because neither this project's nor TideTrace's own band ordering
  convention is announced anywhere in the data itself.
- ``_prepare_batch_input``: the checkpoint's own frozen dB normalisation
  (``mean_db``/``std_db``, stored in ``meta['normalisation']``), not this
  project's 0-255 remap (:func:`src.detection.infer.db_to_model_input`) —
  a different, incompatible preprocessing the checkpoint was never trained
  on. This is exactly why :func:`src.detection.infer.infer_scene` grew a
  ``prepare_fn`` override rather than this module trying to reuse the 0-255
  path.

Class mapping — TideTrace's checkpoint is 3-class (0 sea, 1 look_alike,
2 mineral_oil); this project's scheme is 5-class, in a different order
(0 sea, 1 oil_spill, 2 look_alike, 3 ship, 4 land — see
:mod:`src.detection.dataset`). ``TideTraceDetector.__call__`` reorders and
zero-pads TideTrace's 3 output logits into this project's 5-channel layout
before returning, so every downstream consumer of an ``InferenceResult``
(look-alike filter, characterisation, the alert manager) needs no changes
at all — they already only ever look for ``OIL_CLASS``/``LOOK_ALIKE_CLASS``
by this project's own index, never TideTrace's. "ship" and "land" get a
large negative constant logit (never the argmax winner after softmax) since
the checkpoint was never trained to predict them — this project's own
land-masking-to-NaN upstream already keeps land pixels out of the picture
the same way it always has, unrelated to this detector's own 3 classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.detection.dataset import CLASS_NAMES

logger = logging.getLogger(__name__)

#: TideTrace's own class order -> this project's index in CLASS_NAMES.
#: {sea: sea, look_alike: look_alike, mineral_oil: oil_spill}
_TIDETRACE_CLASS_TO_OURS = {0: 0, 1: 2, 2: 1}

#: Logit value standing in for a class TideTrace's checkpoint never
#: predicts (ship, land) — very negative so softmax assigns it ~0
#: probability and it can never win an argmax, without needing a special
#: case anywhere downstream.
_UNMODELLED_CLASS_LOGIT = -30.0

DEFAULT_NORMALISATION = {"mean_db": -18.0, "std_db": 6.0}


class TideTraceError(RuntimeError):
    """The TideTrace checkpoint could not be loaded or is unusable.

    Raised, never swallowed — this project's own convention is "an
    unavailable *dataset* is a fact, not an error" (wind, currents, AIS),
    but a *requested and missing/broken* detector checkpoint is a real
    configuration problem the caller needs to see, not something to quietly
    paper over with the stub.
    """


def order_bands(sigma0_db: np.ndarray) -> np.ndarray:
    """Sort a (bands, H, W) dB stack by ascending median: co-pol last.

    Copied verbatim from TideTrace's ``app/ml/dataset.py::order_bands`` —
    see this module's own docstring for why this is sorted by measured
    brightness rather than trusting either project's own band-order
    convention (VV-then-VH here, per ``src/ingestion/safe.py``; TideTrace's
    own training data ships the opposite way round, and assuming a fixed
    order the once cost TideTrace a checkpoint that called every pixel of
    every scene "oil").
    """
    a = np.asarray(sigma0_db, dtype=np.float32)
    if a.ndim == 2:
        a = a[None, :, :]
    if a.shape[0] < 2:
        return a
    medians = [float(np.nanmedian(b)) if np.isfinite(b).any() else 0.0 for b in a]
    return a[np.argsort(medians, kind="stable")]


def _prepare_batch_input(sigma0_db: np.ndarray, norm: Dict[str, float]) -> np.ndarray:
    """(bands, H, W) dB stack -> the 3-channel input TideTrace's checkpoint
    was actually trained on. Mirrors TideTrace's own ``app/ml/infer.py::prepare_input``.
    """
    a = order_bands(sigma0_db)
    mean = float(norm.get("mean_db", DEFAULT_NORMALISATION["mean_db"]))
    std = float(norm.get("std_db", DEFAULT_NORMALISATION["std_db"])) or DEFAULT_NORMALISATION["std_db"]
    cross = np.nan_to_num(a[0], nan=mean)
    co = np.nan_to_num(a[1], nan=mean) if a.shape[0] > 1 else cross
    stack = np.stack([cross, co, cross], axis=0)
    return ((stack - mean) / std).astype(np.float32)


class TideTraceDetector:
    """Wraps a loaded TideTrace UNet++ so it's callable exactly the way
    :func:`src.detection.infer.predict_tiles` already calls any model —
    ``logits = model(batch_tensor)`` — a drop-in replacement for
    :class:`~src.detection.infer.ThresholdStubModel` at that one call site,
    nothing else in the pipeline needs to know the difference.
    """

    def __init__(self, net: torch.nn.Module, normalisation: Dict[str, float], device: str = "cpu"):
        self.net = net
        self.normalisation = dict(normalisation or DEFAULT_NORMALISATION)
        self.device = device
        self.net.eval()
        self.net.to(device)

    def eval(self) -> "TideTraceDetector":
        self.net.eval()
        return self

    def prepare(self, tile: np.ndarray) -> np.ndarray:
        """``prepare_fn`` for :func:`src.detection.infer.infer_scene` —
        same signature/role as ``db_to_model_input``, different maths."""
        return _prepare_batch_input(tile, self.normalisation)

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        logits3 = self.net(batch)  # (N, 3, H, W), TideTrace's own class order
        n, _, h, w = logits3.shape
        out = torch.full(
            (n, len(CLASS_NAMES), h, w), _UNMODELLED_CLASS_LOGIT,
            dtype=logits3.dtype, device=logits3.device,
        )
        for tt_idx, our_idx in _TIDETRACE_CLASS_TO_OURS.items():
            out[:, our_idx] = logits3[:, tt_idx]
        return out


def _build_net(arch: str, encoder: str, in_channels: int, classes: int) -> torch.nn.Module:
    """Construct the bare architecture TideTrace's checkpoint expects.

    A tiny, local factory rather than reusing
    :func:`src.detection.model.get_model` — that function reads its
    defaults from *this project's* ``settings.detection``/``settings.training``
    (5-class MKLab scheme), which is exactly the mismatch this adapter
    exists to bridge; building straight from the checkpoint's own ``meta``
    keeps the two schemes from ever being conflated.
    """
    import segmentation_models_pytorch as smp

    factory = {"Unet": smp.Unet, "UnetPlusPlus": smp.UnetPlusPlus}.get(arch)
    if factory is None:
        raise TideTraceError(
            f"checkpoint requests architecture {arch!r}, which this adapter "
            "does not recognise (expected 'Unet' or 'UnetPlusPlus')"
        )
    return factory(
        encoder_name=encoder, encoder_weights=None,
        in_channels=in_channels, classes=classes,
    )


def load_tidetrace_model(
    checkpoint_path: Path, device: Optional[str] = None,
) -> TideTraceDetector:
    """Load a TideTrace checkpoint (``oil_unet_best.pt``) into a callable
    detector. Raises :class:`TideTraceError` on anything wrong with the
    checkpoint — never falls back to the stub silently; that decision
    belongs to the caller, which can choose to catch this and do so
    explicitly and visibly (see ``scripts/run_detection.py``).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise TideTraceError(f"no TideTrace checkpoint at {checkpoint_path}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        blob = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any load failure is a real, reportable problem
        raise TideTraceError(f"could not read checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise TideTraceError(
            f"{checkpoint_path} does not look like a TideTrace checkpoint "
            "(expected a dict with a 'state_dict' key)"
        )

    meta = blob.get("meta", {})
    arch = meta.get("arch", "UnetPlusPlus")
    encoder = meta.get("encoder", "timm-efficientnet-b0")
    in_channels = int(meta.get("in_channels", 3))
    classes = int(meta.get("classes", 3))
    normalisation = meta.get("normalisation", DEFAULT_NORMALISATION)

    try:
        net = _build_net(arch, encoder, in_channels, classes)
        missing, unexpected = net.load_state_dict(blob["state_dict"], strict=True)
    except TideTraceError:
        raise
    except Exception as exc:
        raise TideTraceError(
            f"checkpoint {checkpoint_path} does not match the "
            f"{arch}/{encoder} architecture its own metadata declares: {exc}"
        ) from exc
    if missing or unexpected:
        raise TideTraceError(
            f"checkpoint {checkpoint_path} state_dict mismatch — "
            f"missing={missing}, unexpected={unexpected}"
        )

    logger.info(
        "loaded TideTrace checkpoint %s: %s/%s, %d classes, device=%s",
        checkpoint_path, arch, encoder, classes, device,
    )
    return TideTraceDetector(net, normalisation, device=device)


def available(checkpoint_path: Path) -> bool:
    """True if a TideTrace checkpoint file exists at this path — cheap,
    no torch import, no load attempt. For CLI/status checks that shouldn't
    pay the cost (or the import) of a real load just to ask "is it there".
    """
    return Path(checkpoint_path).is_file()


__all__ = [
    "TideTraceDetector",
    "TideTraceError",
    "load_tidetrace_model",
    "order_bands",
    "available",
    "DEFAULT_NORMALISATION",
]
