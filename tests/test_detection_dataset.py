"""Detection dataset tests.

Runs against a handful of tiny image/mask pairs written in the test, never the
real MKLab download. Also covers the pieces the training loop leans on that are
cheap to check: class weighting, the loss, and the IoU metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from src.detection.dataset import (
    CLASS_COLOURS,
    CLASS_NAMES,
    LOOK_ALIKE_CLASS,
    OIL_CLASS,
    DatasetError,
    MKLabOilSpillDataset,
    build_augmentations,
    decode_rgb_labels,
    describe_class_balance,
    inverse_frequency_weights,
    read_label,
    train_val_datasets,
    write_synthetic_dataset,
)
from src.detection.train import (
    DiceCrossEntropyLoss,
    build_loss,
    confusion_matrix,
    pairwise_confusion,
    per_class_iou,
)

SIZE = 32


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def write_pair(
    split_dir: Path,
    stem: str,
    image: np.ndarray,
    mask: np.ndarray,
    labels_dirname: str = "labels_1D",
    image_suffix: str = ".jpg",
) -> Tuple[Path, Path]:
    """Write one image/mask pair in the MKLab layout."""
    images_dir = split_dir / "images"
    labels_dir = split_dir / labels_dirname
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{stem}{image_suffix}"
    label_path = labels_dir / f"{stem}.png"
    Image.fromarray(image, mode="L").convert("RGB").save(image_path, quality=100)
    Image.fromarray(mask, mode="L").save(label_path)
    return image_path, label_path


def hand_made_pair(seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """A patch with every class present in known places."""
    rng = np.random.default_rng(seed)
    image = rng.integers(100, 140, size=(SIZE, SIZE), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

    image[4:12, 4:12] = 20
    mask[4:12, 4:12] = OIL_CLASS  # 64 oil pixels

    image[20:24, 4:8] = 35
    mask[20:24, 4:8] = LOOK_ALIKE_CLASS  # 16 look-alike pixels

    image[0:2, 30:32] = 255
    mask[0:2, 30:32] = 3  # 4 ship pixels

    image[:, 28:30] = 200
    mask[:, 28:30] = 4  # 64 land pixels
    return image, mask


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Two hand-made pairs under ``<root>/train``."""
    split_dir = tmp_path / "mklab" / "train"
    for index in range(2):
        image, mask = hand_made_pair(seed=index)
        write_pair(split_dir, f"patch_{index:02d}", image, mask)
    return tmp_path / "mklab"


@pytest.fixture
def dataset(dataset_root: Path) -> MKLabOilSpillDataset:
    return MKLabOilSpillDataset(root=dataset_root, split="train", train=False)


# --------------------------------------------------------------------------- #
# class scheme
# --------------------------------------------------------------------------- #
def test_class_scheme_is_the_mklab_one() -> None:
    assert CLASS_NAMES == ("sea", "oil_spill", "look_alike", "ship", "land")
    assert (OIL_CLASS, LOOK_ALIKE_CLASS) == (1, 2)
    assert len(CLASS_COLOURS) == len(CLASS_NAMES)


def test_decode_rgb_labels_maps_the_palette() -> None:
    rgb = np.array([[c for c in CLASS_COLOURS]], dtype=np.uint8)
    assert decode_rgb_labels(rgb).tolist() == [[0, 1, 2, 3, 4]]


def test_decode_rgb_labels_snaps_near_misses() -> None:
    """Redistributed copies re-encode the label PNGs and shift colours a little."""
    nudged = np.array([[[2, 253, 250], [250, 4, 3]]], dtype=np.uint8)
    assert decode_rgb_labels(nudged).tolist() == [[OIL_CLASS, LOOK_ALIKE_CLASS]]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def test_dataset_finds_both_pairs(dataset: MKLabOilSpillDataset) -> None:
    assert len(dataset) == 2
    assert dataset.stems == ["patch_00", "patch_01"]


def test_item_shapes_and_dtypes(dataset: MKLabOilSpillDataset) -> None:
    item = dataset[0]
    assert set(item) == {"image", "mask", "stem"}
    assert item["image"].shape == (3, SIZE, SIZE)
    assert item["image"].dtype == torch.float32
    assert item["mask"].shape == (SIZE, SIZE)
    assert item["mask"].dtype == torch.int64
    assert item["stem"] == "patch_00"


def test_masks_stay_inside_the_class_scheme(dataset: MKLabOilSpillDataset) -> None:
    for index in range(len(dataset)):
        mask = dataset[index]["mask"]
        assert int(mask.min()) >= 0
        assert int(mask.max()) < len(CLASS_NAMES)


def test_every_class_survives_loading(dataset: MKLabOilSpillDataset) -> None:
    present = set()
    for index in range(len(dataset)):
        present |= set(dataset[index]["mask"].unique().tolist())
    assert present == set(range(len(CLASS_NAMES)))


def test_single_channel_mode(dataset_root: Path) -> None:
    single = MKLabOilSpillDataset(root=dataset_root, split="train", train=False,
                                  in_channels=1)
    assert single[0]["image"].shape == (1, SIZE, SIZE)


def test_rgb_label_folder_is_decoded(tmp_path: Path) -> None:
    """A dataset copy shipping only the colour-coded labels still works."""
    root = tmp_path / "mklab"
    write_synthetic_dataset(root, n_samples=2, size=SIZE, rgb_labels=True)
    ds = MKLabOilSpillDataset(root=root, split="train", train=False,
                              labels_dirname="labels")
    mask = ds[0]["mask"]
    assert int(mask.max()) < len(CLASS_NAMES)
    assert OIL_CLASS in set(mask.unique().tolist())


def test_custom_directory_names(tmp_path: Path) -> None:
    """The knobs the docstring promises actually work."""
    split_dir = tmp_path / "custom" / "training"
    image, mask = hand_made_pair()
    write_pair(split_dir, "a", image, mask, labels_dirname="gt")

    ds = MKLabOilSpillDataset(
        root=tmp_path / "custom", split="training", train=False,
        images_dirname="images", labels_dirname="gt",
    )
    assert len(ds) == 1


def test_missing_split_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="dataset split not found"):
        MKLabOilSpillDataset(root=tmp_path / "nowhere", split="train")


def test_missing_labels_directory_is_reported(tmp_path: Path) -> None:
    (tmp_path / "mklab" / "train" / "images").mkdir(parents=True)
    with pytest.raises(DatasetError, match="no labels directory"):
        MKLabOilSpillDataset(root=tmp_path / "mklab", split="train")


def test_an_unlabelled_image_is_an_error_not_a_silent_skip(dataset_root: Path) -> None:
    image, mask = hand_made_pair()
    write_pair(dataset_root / "train", "orphan", image, mask)
    (dataset_root / "train" / "labels_1D" / "orphan.png").unlink()

    with pytest.raises(DatasetError, match="have no label"):
        MKLabOilSpillDataset(root=dataset_root, split="train")


def test_out_of_range_label_is_reported(tmp_path: Path) -> None:
    split_dir = tmp_path / "mklab" / "train"
    image, mask = hand_made_pair()
    mask[0, 0] = 9  # no such class
    write_pair(split_dir, "bad", image, mask)

    ds = MKLabOilSpillDataset(root=tmp_path / "mklab", split="train", train=False)
    with pytest.raises(DatasetError, match="outside the 5-class scheme"):
        _ = ds[0]


def test_mismatched_image_and_label_size_is_reported(tmp_path: Path) -> None:
    split_dir = tmp_path / "mklab" / "train"
    image, mask = hand_made_pair()
    write_pair(split_dir, "mismatch", image, mask[:16])

    ds = MKLabOilSpillDataset(root=tmp_path / "mklab", split="train", train=False)
    with pytest.raises(DatasetError, match="but its label is"):
        _ = ds[0]


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def test_augmentation_does_not_crash_and_keeps_shapes(dataset_root: Path) -> None:
    augmented = MKLabOilSpillDataset(root=dataset_root, split="train", train=True)
    for _ in range(10):  # random transforms, so run it a few times
        item = augmented[0]
        assert item["image"].shape == (3, SIZE, SIZE)
        assert item["mask"].shape == (SIZE, SIZE)
        assert torch.isfinite(item["image"]).all()


def test_augmentation_moves_pixels_but_keeps_the_class_mix(dataset_root: Path) -> None:
    """Flips and rotations relabel nothing: the pixel counts per class hold."""
    plain = MKLabOilSpillDataset(root=dataset_root, split="train", train=False)
    augmented = MKLabOilSpillDataset(root=dataset_root, split="train", train=True)

    baseline = torch.bincount(plain[0]["mask"].ravel(), minlength=len(CLASS_NAMES))
    for _ in range(10):
        counts = torch.bincount(augmented[0]["mask"].ravel(), minlength=len(CLASS_NAMES))
        assert torch.equal(counts, baseline)


def test_augmentation_is_geometry_only() -> None:
    """No brightness/contrast/noise transform may sneak in - values are dB."""
    names = {type(t).__name__ for t in build_augmentations(train=True).transforms}
    assert names == {"HorizontalFlip", "VerticalFlip", "RandomRotate90",
                     "Transpose", "Normalize", "ToTensorV2"}


def test_validation_pipeline_has_no_random_transforms() -> None:
    names = {type(t).__name__ for t in build_augmentations(train=False).transforms}
    assert names == {"Normalize", "ToTensorV2"}


def test_augmentation_crops_and_pads_to_the_requested_size(dataset_root: Path) -> None:
    bigger = MKLabOilSpillDataset(root=dataset_root, split="train", train=True,
                                  image_size=48)
    item = bigger[0]
    assert item["image"].shape == (3, 48, 48)
    assert item["mask"].shape == (48, 48)


def test_normalize_none_keeps_raw_counts(dataset_root: Path) -> None:
    raw = MKLabOilSpillDataset(root=dataset_root, split="train", train=False,
                               normalize="none")
    assert float(raw[0]["image"].max()) > 1.5  # still 0-255, not normalised


# --------------------------------------------------------------------------- #
# splitting and class weighting
# --------------------------------------------------------------------------- #
def test_train_val_split_is_disjoint_and_covers_everything(tmp_path: Path) -> None:
    root = tmp_path / "mklab"
    write_synthetic_dataset(root, n_samples=10, size=SIZE)

    train_ds, val_ds = train_val_datasets(root=root, val_split=0.3, seed=1)

    assert len(train_ds) + len(val_ds) == 10
    assert set(train_ds.stems).isdisjoint(val_ds.stems)
    assert len(val_ds) == 3


def test_val_split_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "mklab"
    write_synthetic_dataset(root, n_samples=8, size=SIZE)
    first = train_val_datasets(root=root, seed=7)[1].stems
    second = train_val_datasets(root=root, seed=7)[1].stems
    assert first == second


def test_val_half_is_not_augmented(tmp_path: Path) -> None:
    root = tmp_path / "mklab"
    write_synthetic_dataset(root, n_samples=6, size=SIZE)
    _, val_ds = train_val_datasets(root=root)
    names = {type(t).__name__ for t in val_ds.transform.transforms}
    assert "HorizontalFlip" not in names


def test_class_pixel_counts_match_the_hand_made_masks(
    dataset: MKLabOilSpillDataset,
) -> None:
    counts = dataset.class_pixel_counts()
    assert counts.shape == (5,)
    assert counts[OIL_CLASS] == 2 * 64
    assert counts[LOOK_ALIKE_CLASS] == 2 * 16
    assert counts.sum() == 2 * SIZE * SIZE


def test_inverse_frequency_weights_lift_the_rare_class() -> None:
    counts = np.array([990_000, 5_000, 4_000, 500, 500], dtype=np.int64)
    weights = inverse_frequency_weights(counts)
    assert weights[OIL_CLASS] > weights[0] * 50
    assert float(weights.mean()) == pytest.approx(1.0, rel=1e-5)


def test_inverse_frequency_weights_are_capped() -> None:
    counts = np.array([10_000_000, 1, 1000, 1000, 1000], dtype=np.int64)
    weights = inverse_frequency_weights(counts, max_weight=10.0)
    assert torch.isfinite(weights).all()
    # the cap applies before the mean normalisation, so no class runs away
    assert float(weights.max()) / float(weights.min()) < 1e4


def test_absent_classes_get_zero_weight() -> None:
    counts = np.array([1000, 100, 0, 10, 10], dtype=np.int64)
    assert float(inverse_frequency_weights(counts)[2]) == 0.0


def test_describe_class_balance_lists_every_class(
    dataset: MKLabOilSpillDataset,
) -> None:
    text = describe_class_balance(dataset.class_pixel_counts(), CLASS_NAMES)
    for name in CLASS_NAMES:
        assert name in text


def test_dataset_summary(dataset: MKLabOilSpillDataset) -> None:
    summary = dataset.summary()
    assert summary["pairs"] == 2
    assert set(summary["pixels_per_class"]) == set(CLASS_NAMES)
    assert sum(summary["fraction_per_class"].values()) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# loss and metrics the training loop depends on
# --------------------------------------------------------------------------- #
def test_confusion_matrix_and_iou_on_a_perfect_prediction() -> None:
    target = torch.tensor([[[0, 1], [2, 3]]])
    matrix = confusion_matrix(target, target, 5)
    assert matrix.trace() == 4
    iou = per_class_iou(matrix)
    assert iou[:4] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert np.isnan(iou[4])  # class absent from truth and prediction


def test_iou_penalises_calling_oil_a_look_alike() -> None:
    target = torch.full((1, 4, 4), OIL_CLASS)
    prediction = torch.full((1, 4, 4), LOOK_ALIKE_CLASS)
    iou = per_class_iou(confusion_matrix(prediction, target, 5))
    assert iou[OIL_CLASS] == 0.0
    assert iou[LOOK_ALIKE_CLASS] == 0.0


def test_pairwise_confusion_reports_both_directions() -> None:
    target = torch.tensor([[OIL_CLASS, OIL_CLASS, LOOK_ALIKE_CLASS, LOOK_ALIKE_CLASS]])
    prediction = torch.tensor([[OIL_CLASS, LOOK_ALIKE_CLASS, LOOK_ALIKE_CLASS, OIL_CLASS]])
    mix = pairwise_confusion(
        confusion_matrix(prediction, target, 5), OIL_CLASS, LOOK_ALIKE_CLASS
    )
    assert mix["a_as_b_rate"] == pytest.approx(0.5)
    assert mix["b_as_a_rate"] == pytest.approx(0.5)


def test_dice_ce_loss_runs_and_rewards_the_right_answer() -> None:
    targets = torch.zeros((2, 8, 8), dtype=torch.long)
    targets[:, :4, :4] = OIL_CLASS

    good = torch.nn.functional.one_hot(targets, 5).permute(0, 3, 1, 2).float() * 10.0
    bad = torch.zeros_like(good)

    loss = DiceCrossEntropyLoss(5, dice_weight=0.5)
    assert float(loss(good, targets)) < float(loss(bad, targets))
    assert torch.isfinite(loss(bad, targets))


def test_dice_ce_loss_backpropagates() -> None:
    logits = torch.randn(2, 5, 8, 8, requires_grad=True)
    targets = torch.randint(0, 5, (2, 8, 8))
    build_loss(5, None, "dice_ce", 0.5)(logits, targets).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_class_weights_make_missing_oil_cost_more_than_missing_sea() -> None:
    """The whole point of weighting: one missed oil pixel must hurt more.

    Cross-entropy normalises by the summed weight of the target pixels, so the
    same targets are used for both cases and only the misclassified pixel moves.
    """
    targets = torch.zeros((1, 4, 4), dtype=torch.long)
    targets[0, 0, 0] = OIL_CLASS

    weights = torch.ones(5)
    weights[OIL_CLASS] = 20.0
    criterion = build_loss(5, weights, "ce")

    def logits_wrong_at(row: int, col: int) -> torch.Tensor:
        logits = torch.full((1, 5, 4, 4), -10.0)
        for r in range(4):
            for c in range(4):
                logits[0, int(targets[0, r, c]), r, c] = 10.0
        # flip this one pixel to a confidently wrong class
        logits[0, :, row, col] = -10.0
        logits[0, 3, row, col] = 10.0
        return logits

    missed_oil = float(criterion(logits_wrong_at(0, 0), targets))
    missed_sea = float(criterion(logits_wrong_at(3, 3), targets))
    assert missed_oil > missed_sea * 10


def test_class_weights_are_attached_to_the_loss() -> None:
    weights = torch.tensor([1.0, 20.0, 5.0, 1.0, 1.0])
    assert torch.equal(build_loss(5, weights, "ce").weight, weights)
    assert torch.equal(build_loss(5, weights, "dice_ce").cross_entropy.weight, weights)


def test_ce_only_loss_is_available() -> None:
    loss = build_loss(5, None, "ce")
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


# --------------------------------------------------------------------------- #
# synthetic generator used by the smoke test
# --------------------------------------------------------------------------- #
def test_synthetic_dataset_contains_every_class(tmp_path: Path) -> None:
    root = tmp_path / "mklab"
    write_synthetic_dataset(root, n_samples=4, size=64)

    ds = MKLabOilSpillDataset(root=root, split="train", train=False)
    assert len(ds) == 4

    present = set()
    for index in range(len(ds)):
        present |= set(ds[index]["mask"].unique().tolist())
    assert present == set(range(len(CLASS_NAMES)))


def test_synthetic_labels_are_written_as_indices(tmp_path: Path) -> None:
    root = tmp_path / "mklab"
    write_synthetic_dataset(root, n_samples=1, size=32)
    label = read_label(next((root / "train" / "labels_1D").glob("*.png")))
    assert label.max() < len(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# the training loop itself, on generated data
# --------------------------------------------------------------------------- #
def test_training_smoke_run(tmp_path: Path) -> None:
    """One tiny end-to-end run: loop, loss, metrics and checkpointing all wire up.

    Deliberately small and untrained - this asserts nothing about accuracy, only
    that a real run would not fall over.
    """
    from src.detection.train import smoke_test

    models_dir = tmp_path / "models"
    summary = smoke_test(output_dir=models_dir, epochs=1, batch_size=2)

    assert summary["epochs_run"] == 1
    assert summary["device"] == "cpu"
    assert summary["train_samples"] > 0 and summary["val_samples"] > 0
    assert (models_dir / "best.pt").is_file()
    assert (models_dir / "last.pt").is_file()
    assert (models_dir / "training_history.json").is_file()


def test_checkpoint_round_trips(tmp_path: Path) -> None:
    """A checkpoint rebuilds the same network without being told its shape."""
    from src.detection.model import get_model, load_checkpoint, save_checkpoint

    model = get_model(num_classes=5, encoder_weights=None, in_channels=3)
    path = save_checkpoint(
        tmp_path / "ckpt.pt", model, epoch=3,
        extra={"arch": "unet", "encoder": "resnet34", "in_channels": 3,
               "num_classes": 5},
    )

    restored, payload = load_checkpoint(path)
    assert payload["epoch"] == 3

    batch = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model.eval()(batch), restored(batch), atol=1e-6)


def test_get_model_shapes() -> None:
    from src.detection.model import count_parameters, get_model

    model = get_model(num_classes=5, encoder_weights=None).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 5, 64, 64)
    total, trainable = count_parameters(model)
    assert total == trainable > 0
