"""Train the oil-spill segmentation model.

Handles the thing that decides whether this model is useful at all: oil is a
fraction of a percent of the pixels in this dataset, so an unweighted loss
converges to "everything is sea" and reports a fine-looking accuracy. Two
explicit corrections:

* class weights from inverse pixel frequency, computed from the actual training
  split rather than guessed;
* Dice combined with cross-entropy, since Dice scores a rare class by overlap
  rather than by pixel count.

Validation reports per-class IoU every epoch, and calls out **oil vs look-alike**
separately - separating those two is the entire problem, and a model that scores
well overall while confusing them is worthless downstream.

Run it::

    # real training, once data/raw/mklab is populated
    python -m src.detection.train --epochs 40 --batch-size 8

    # tiny end-to-end check on generated data, no dataset and no GPU needed
    python -m src.detection.train --smoke-test

Checkpoints go to ``models/``: ``best.pt`` (highest oil IoU) and ``last.pt``.
Every time ``best.pt`` is (re)written, ``models/best_metrics.json`` is too -
a small model-card summary (architecture, sample counts, headline oil-class
precision/recall/dice/IoU, oil<->look-alike confusion rates) for
``GET /api/model/info`` (step 8.3) to serve without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import Settings, get_settings
from src.detection.dataset import (
    CLASS_NAMES,
    LOOK_ALIKE_CLASS,
    OIL_CLASS,
    MKLabOilSpillDataset,
    describe_class_balance,
    inverse_frequency_weights,
    train_val_datasets,
    write_synthetic_dataset,
)
from src.detection.model import count_parameters, get_model, save_checkpoint

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> np.ndarray:
    """Accumulate a ``num_classes x num_classes`` confusion matrix (true, pred)."""
    pred = predictions.reshape(-1).to(torch.int64)
    true = targets.reshape(-1).to(torch.int64)
    valid = (true >= 0) & (true < num_classes) & (pred >= 0) & (pred < num_classes)
    pairs = true[valid] * num_classes + pred[valid]
    counts = torch.bincount(pairs, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes).cpu().numpy()


def per_class_iou(matrix: np.ndarray) -> np.ndarray:
    """IoU per class; NaN for classes absent from both truth and prediction."""
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, np.nan)


def per_class_precision(matrix: np.ndarray) -> np.ndarray:
    """Precision per class: TP / (TP + FP) - of everything *predicted* as
    this class, how much actually was it. NaN when the class was never
    predicted at all (rows are truth, columns are predictions - see
    :func:`confusion_matrix` - so this is TP over each column's sum)."""
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(predicted > 0, tp / predicted, np.nan)


def per_class_recall(matrix: np.ndarray) -> np.ndarray:
    """Recall per class: TP / (TP + FN) - of everything *truly* this class,
    how much the model caught (TP over each row's sum). NaN when the class
    never appears in the ground truth."""
    tp = np.diag(matrix).astype(np.float64)
    actual = matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(actual > 0, tp / actual, np.nan)


def dice_from_iou(iou: np.ndarray) -> np.ndarray:
    """Dice from IoU already computed: ``2*iou / (1+iou)``.

    A plain algebraic identity (both measure the same intersection/union;
    with U = TP+FP+FN, IoU = TP/U and Dice = 2TP/(TP+U) = 2*IoU/(IoU+1)) -
    not an approximation, so this needs no second pass over predictions.
    NaN propagates through exactly where IoU was already NaN (class absent
    from both truth and prediction).
    """
    return 2.0 * iou / (1.0 + iou)


def pairwise_confusion(
    matrix: np.ndarray, a: int, b: int
) -> Dict[str, float]:
    """How often the two classes that matter get mistaken for each other."""
    a_as_b = float(matrix[a, b])
    b_as_a = float(matrix[b, a])
    a_total = float(matrix[a].sum())
    b_total = float(matrix[b].sum())
    return {
        "a_as_b": a_as_b,
        "b_as_a": b_as_a,
        "a_as_b_rate": a_as_b / a_total if a_total else float("nan"),
        "b_as_a_rate": b_as_a / b_total if b_total else float("nan"),
    }


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
class DiceCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy plus multiclass Dice.

    Cross-entropy with inverse-frequency weights fixes the gradient imbalance;
    Dice fixes the metric imbalance, because it scores a rare class by region
    overlap instead of by how many pixels it happens to own.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = float(dice_weight)
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(logits, targets)
        if self.dice_weight <= 0.0:
            return ce

        probabilities = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(targets, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)

        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        cardinality = probabilities.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Only score classes that actually appear in the batch.
        present = one_hot.sum(dims) > 0
        dice_loss = 1.0 - (dice[present].mean() if present.any() else dice.mean())
        return (1.0 - self.dice_weight) * ce + self.dice_weight * dice_loss


def build_loss(
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    kind: str = "dice_ce",
    dice_weight: float = 0.5,
) -> nn.Module:
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    return DiceCrossEntropyLoss(num_classes, class_weights, dice_weight)


# --------------------------------------------------------------------------- #
# epoch loops
# --------------------------------------------------------------------------- #
@dataclass
class EpochResult:
    """Everything one validation pass produced.

    ``precision``/``recall`` are computed here, off the same confusion
    matrix ``iou`` already came from - see :func:`per_class_precision`/
    :func:`per_class_recall`. ``dice`` is not stored: it is a pure function
    of ``iou`` (:func:`dice_from_iou`), so it is a property instead of a
    third array that could drift out of sync with it.
    """

    loss: float
    iou: np.ndarray
    matrix: np.ndarray
    precision: np.ndarray = field(default_factory=lambda: np.array([]))
    recall: np.ndarray = field(default_factory=lambda: np.array([]))
    class_names: Sequence[str] = field(default_factory=lambda: list(CLASS_NAMES))

    @property
    def mean_iou(self) -> float:
        return float(np.nanmean(self.iou)) if self.iou.size else float("nan")

    @property
    def dice(self) -> np.ndarray:
        return dice_from_iou(self.iou)

    def named_iou(self) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.class_names, self.iou)
        }

    def named_precision(self) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.class_names, self.precision)
        }

    def named_recall(self) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.class_names, self.recall)
        }

    def named_dice(self) -> Dict[str, float]:
        return {
            name: float(value) for name, value in zip(self.class_names, self.dice)
        }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any] = None,
    log_every: int = 20,
) -> float:
    """One pass over the training split; returns the mean loss."""
    model.train()
    total, batches = 0.0, 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()

        total += float(loss.detach())
        batches += 1
        if log_every and step % log_every == 0:
            logger.info("  step %d/%d  loss %.4f", step, len(loader), total / batches)

    return total / max(batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str] = CLASS_NAMES,
) -> EpochResult:
    """One pass over the validation split, accumulating the confusion matrix."""
    model.eval()
    total, batches = 0.0, 0
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        logits = model(images)
        total += float(criterion(logits, masks))
        batches += 1
        matrix += confusion_matrix(logits.argmax(dim=1), masks, num_classes)

    return EpochResult(
        loss=total / max(batches, 1),
        iou=per_class_iou(matrix),
        matrix=matrix,
        precision=per_class_precision(matrix),
        recall=per_class_recall(matrix),
        class_names=list(class_names)[:num_classes],
    )


def _best_metrics_payload(
    model_spec: Dict[str, Any],
    train_ds: "MKLabOilSpillDataset",
    val_ds: "MKLabOilSpillDataset",
    result: EpochResult,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Step 8.3's model card - the frontend's ``GET /api/model/info``
    payload - built from the SAME confusion matrix :func:`validate` already
    produced for this (the new-best) epoch. No second evaluation pass.

    ``precision``/``recall``/``dice`` are the **oil class's own** values,
    not a per-class average - this file exists to answer "is the model
    good at the one class that matters", the same lens
    :func:`report_validation`'s own "the number that matters" block already
    uses for IoU. The full per-class breakdown (all classes, every epoch)
    still lives in ``training_history.json``'s ``history[i]`` entries - see
    the ``history.append`` call above.

    Deliberately two IoUs and two confusion rates, not one invented
    "oil_vs_lookalike_iou" number - they are genuinely distinct quantities
    (:func:`pairwise_confusion` measures misclassification rate, not IoU).
    """
    named_iou = result.named_iou()
    oil_name = class_names[OIL_CLASS] if len(class_names) > OIL_CLASS else None
    look_name = class_names[LOOK_ALIKE_CLASS] if len(class_names) > LOOK_ALIKE_CLASS else None

    if len(class_names) > max(OIL_CLASS, LOOK_ALIKE_CLASS):
        mix = pairwise_confusion(result.matrix, OIL_CLASS, LOOK_ALIKE_CLASS)
    else:
        mix = {"a_as_b_rate": float("nan"), "b_as_a_rate": float("nan")}

    return {
        "architecture": model_spec,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "mean_iou": result.mean_iou,
        "oil_iou": named_iou.get(oil_name, float("nan")) if oil_name else float("nan"),
        "look_alike_iou": named_iou.get(look_name, float("nan")) if look_name else float("nan"),
        "precision": float(result.precision[OIL_CLASS]),
        "recall": float(result.recall[OIL_CLASS]),
        "dice": float(result.dice[OIL_CLASS]),
        "oil_as_lookalike_rate": mix["a_as_b_rate"],
        "lookalike_as_oil_rate": mix["b_as_a_rate"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def report_validation(result: EpochResult, epoch: int, epochs: int) -> None:
    """Print the validation numbers, oil vs look-alike first and separately."""
    names = list(result.class_names)
    iou = result.named_iou()

    oil = iou.get(names[OIL_CLASS], float("nan")) if len(names) > OIL_CLASS else float("nan")
    look = (
        iou.get(names[LOOK_ALIKE_CLASS], float("nan"))
        if len(names) > LOOK_ALIKE_CLASS
        else float("nan")
    )

    print(f"\n=== epoch {epoch}/{epochs} | val loss {result.loss:.4f} ===")
    print("--- the number that matters -------------------------------")
    print(f"  oil_spill  IoU : {oil:.4f}")
    print(f"  look_alike IoU : {look:.4f}")
    if len(names) > max(OIL_CLASS, LOOK_ALIKE_CLASS):
        mix = pairwise_confusion(result.matrix, OIL_CLASS, LOOK_ALIKE_CLASS)
        print(
            f"  confusion      : {mix['a_as_b_rate']:.2%} of oil called look-alike, "
            f"{mix['b_as_a_rate']:.2%} of look-alike called oil"
        )
    print("--- all classes -------------------------------------------")
    for name, value in iou.items():
        print(f"  {name:<12} IoU : {value:.4f}")
    print(f"  {'mean':<12} IoU : {result.mean_iou:.4f}\n")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    image_size: Optional[int] = None,
    val_split: Optional[float] = None,
    num_workers: Optional[int] = None,
    device: str = "auto",
    encoder_weights: Optional[str] = "config",
    amp: Optional[bool] = None,
    seed: Optional[int] = None,
    limit: Optional[int] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    """Run training end to end and return the history.

    Every argument falls back to the ``training`` section of the config, so the
    CLI flags are overrides rather than a second source of truth.
    """
    settings = settings or get_settings()
    cfg = settings.training

    epochs = epochs or cfg.epochs
    batch_size = batch_size or cfg.batch_size
    learning_rate = learning_rate or cfg.learning_rate
    image_size = image_size or cfg.image_size
    val_split = cfg.val_split if val_split is None else val_split
    num_workers = cfg.num_workers if num_workers is None else num_workers
    seed = cfg.seed if seed is None else seed
    if encoder_weights == "config":
        encoder_weights = cfg.encoder_weights

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch_device = resolve_device(device)
    use_amp = (cfg.amp if amp is None else amp) and torch_device.type == "cuda"
    if amp and torch_device.type != "cuda":
        logger.warning("AMP requested but %s is not CUDA; running in fp32", torch_device)

    train_ds, val_ds = train_val_datasets(
        root=data_dir,
        val_split=val_split,
        seed=seed,
        image_size=image_size,
        settings=settings,
    )
    if limit:
        train_ds.pairs = train_ds.pairs[:limit]
        val_ds.pairs = val_ds.pairs[: max(1, limit // 4)]

    counts = train_ds.class_pixel_counts()
    print("\nclass balance of the training split:")
    print(describe_class_balance(counts, train_ds.class_names))

    weights = (
        torch.tensor(cfg.class_weights, dtype=torch.float32)
        if cfg.class_weights
        else inverse_frequency_weights(counts, cfg.max_class_weight)
    )
    print(
        "\nclass weights: "
        + ", ".join(f"{n}={w:.2f}" for n, w in zip(train_ds.class_names, weights.tolist()))
        + "\n"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=torch_device.type == "cuda", drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    model = get_model(
        num_classes=cfg.num_classes,
        encoder_weights=encoder_weights,
        settings=settings,
    ).to(torch_device)
    total_params, trainable = count_parameters(model)
    logger.info("model has %.1fM parameters (%.1fM trainable)",
                total_params / 1e6, trainable / 1e6)

    criterion = build_loss(
        cfg.num_classes, weights.to(torch_device), cfg.loss, cfg.dice_weight
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    out_dir = Path(output_dir) if output_dir else settings.paths.resolve(
        settings.paths.models_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_index = train_ds.class_names.index(cfg.checkpoint_metric)
    model_spec = {
        "arch": settings.detection.model_arch,
        "encoder": settings.detection.encoder,
        "in_channels": cfg.in_channels,
        "num_classes": cfg.num_classes,
        "class_names": train_ds.class_names,
        "image_size": image_size,
    }

    history: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_epoch = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, torch_device, scaler
        )
        result = validate(
            model, val_loader, criterion, torch_device, cfg.num_classes,
            train_ds.class_names,
        )
        scheduler.step()
        report_validation(result, epoch, epochs)

        score = result.iou[metric_index]
        score = -np.inf if np.isnan(score) else float(score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": result.loss,
                "mean_iou": result.mean_iou,
                "iou": result.named_iou(),
                "precision": result.named_precision(),
                "recall": result.named_recall(),
                "dice": result.named_dice(),
                "seconds": round(time.time() - epoch_start, 1),
            }
        )

        save_checkpoint(out_dir / "last.pt", model, epoch, history[-1], model_spec)
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(out_dir / "best.pt", model, epoch, history[-1], model_spec)
            (out_dir / "best_metrics.json").write_text(
                json.dumps(
                    _best_metrics_payload(model_spec, train_ds, val_ds, result, train_ds.class_names),
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"  -> new best {cfg.checkpoint_metric} IoU {score:.4f}, checkpoint saved")
        elif epoch - best_epoch >= cfg.early_stopping_patience:
            print(
                f"  -> no {cfg.checkpoint_metric} IoU improvement for "
                f"{cfg.early_stopping_patience} epochs, stopping early"
            )
            break

    summary = {
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        f"best_{cfg.checkpoint_metric}_iou": None if best_score == -np.inf else best_score,
        "minutes": round((time.time() - started) / 60.0, 2),
        "device": str(torch_device),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "history": history,
    }
    (out_dir / "training_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"finished: best {cfg.checkpoint_metric} IoU "
        f"{summary[f'best_{cfg.checkpoint_metric}_iou']} at epoch {best_epoch}; "
        f"checkpoints in {out_dir}"
    )
    return summary


def smoke_test(output_dir: Optional[Path] = None, **overrides: Any) -> Dict[str, Any]:
    """Tiny end-to-end run on generated data.

    Proves the loop, the loss, the metrics and checkpointing all work without a
    dataset, a download or a GPU. Not a substitute for real training.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "mklab"
        write_synthetic_dataset(root, n_samples=8, size=64, split="train")
        params: Dict[str, Any] = dict(
            data_dir=root,
            output_dir=output_dir or Path(tmp) / "models",
            epochs=2,
            batch_size=2,
            image_size=64,
            val_split=0.25,
            num_workers=0,
            device="cpu",
            encoder_weights=None,  # no download
            amp=False,
        )
        params.update(overrides)
        return train(**params)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the oil-spill segmentation model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="dataset root; defaults to training.dataset_dir")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where checkpoints go; defaults to models/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of training pairs, for quick runs")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="train from scratch instead of ImageNet weights")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision even on CUDA")
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny run on generated data; needs no dataset or GPU")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.smoke_test:
        print("running the synthetic smoke test (not real training)\n")
        smoke_test(output_dir=args.output_dir)
        print("\nsmoke test passed")
        return 0

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        device=args.device,
        encoder_weights=None if args.no_pretrained else "config",
        amp=False if args.no_amp else None,
        seed=args.seed,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())


__all__ = [
    "train",
    "smoke_test",
    "train_one_epoch",
    "validate",
    "EpochResult",
    "DiceCrossEntropyLoss",
    "build_loss",
    "confusion_matrix",
    "per_class_iou",
    "per_class_precision",
    "per_class_recall",
    "dice_from_iou",
    "pairwise_confusion",
    "report_validation",
    "resolve_device",
    "main",
]
