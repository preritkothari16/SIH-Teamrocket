"""PyTorch dataset for the MKLab / Krestenitis SAR oil-spill dataset.

Expected layout
---------------
The loader assumes the dataset is unpacked under ``data/raw/mklab/`` exactly as
it ships from MKLab (Krestenitis et al., *Oil Spill Identification from
Satellite Images Using Deep Neural Networks*, Remote Sensing 2019)::

    data/raw/mklab/
    |-- train/
    |   |-- images/         *.jpg   the SAR patches
    |   |-- labels_1D/      *.png   single channel, pixel value = class index
    |   `-- labels/         *.png   optional, RGB colour-coded version
    `-- test/
        |-- images/
        |-- labels_1D/
        `-- labels/

**If your download differs, this is the part to reconcile.** The three knobs
are ``images_dirname``, ``labels_dirname`` and ``split``; every one is a
constructor argument, so a different folder naming scheme needs no code change:

    MKLabOilSpillDataset(root, split="training", images_dirname="img",
                         labels_dirname="mask")

Pairing is by filename stem, so ``images/oil_0123.jpg`` pairs with
``labels_1D/oil_0123.png``. Extensions may differ between the two; anything
Pillow can open works. An image with no matching label is an error, not a
silent skip - a half-labelled training set is worse than a missing one.

Labels
------
Five classes, and **the index order is fixed by the dataset**:

===== ============ =====================
Index Name         RGB in ``labels/``
===== ============ =====================
0     sea          (0, 0, 0)       black
1     oil_spill    (0, 255, 255)   cyan
2     look_alike   (255, 0, 0)     red
3     ship         (153, 76, 0)    brown
4     land         (0, 153, 0)     green
===== ============ =====================

``labels_1D`` is preferred because it is already class indices. When only the
RGB ``labels`` folder is present the palette above is decoded, with unknown
colours snapped to the nearest palette entry (JPEG-ish artefacts in the label
PNGs are common in redistributed copies).

Augmentation
------------
Flips, 90-degree rotations and transposes only - the dihedral group of the
square. Brightness, contrast, blur and noise augmentations are deliberately
absent: pixel values here are radar backscatter, and shifting them shifts the
physical quantity the look-alike filter later reasons about.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Class index order. Fixed by the dataset - reordering silently relabels data.
CLASS_NAMES: Tuple[str, ...] = ("sea", "oil_spill", "look_alike", "ship", "land")

#: RGB palette used by the dataset's colour-coded ``labels/`` folder.
CLASS_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0 sea
    (0, 255, 255),  # 1 oil spill
    (255, 0, 0),  # 2 look-alike
    (153, 76, 0),  # 3 ship
    (0, 153, 0),  # 4 land
)

OIL_CLASS = CLASS_NAMES.index("oil_spill")
LOOK_ALIKE_CLASS = CLASS_NAMES.index("look_alike")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
LABEL_SUFFIXES = (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")

#: Preferred first, so an index-coded label folder wins over the RGB one.
LABEL_DIRNAMES = ("labels_1D", "labels_1d", "labels", "masks", "annotations")
IMAGE_DIRNAMES = ("images", "img", "imgs")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DatasetError(RuntimeError):
    """The dataset on disk does not match the expected layout."""


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def build_augmentations(
    train: bool = True,
    image_size: Optional[int] = None,
    normalize: str = "imagenet",
    in_channels: int = 3,
) -> A.Compose:
    """Albumentations pipeline for one split.

    Geometry only. Every transform here maps the image and its mask the same
    way and leaves backscatter values untouched, which is the whole point: the
    downstream look-alike filter thresholds on dB values, so an augmentation
    that brightened a patch would be teaching the model a lie.
    """
    steps: List[A.BasicTransform] = []

    if image_size:
        # Pad first so a patch smaller than the crop is still usable.
        steps.append(
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
                fill_mask=0,
            )
        )
        steps.append(
            A.RandomCrop(height=image_size, width=image_size)
            if train
            else A.CenterCrop(height=image_size, width=image_size)
        )

    if train:
        steps += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ]

    if normalize == "imagenet":
        mean = IMAGENET_MEAN[:in_channels] if in_channels <= 3 else IMAGENET_MEAN
        std = IMAGENET_STD[:in_channels] if in_channels <= 3 else IMAGENET_STD
        steps.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0))
    elif normalize == "unit":
        steps.append(A.Normalize(mean=(0.0,) * in_channels, std=(1.0,) * in_channels,
                                 max_pixel_value=255.0))
    elif normalize not in ("none", None):
        raise DatasetError(f"unknown normalize mode {normalize!r}")

    steps.append(ToTensorV2())
    return A.Compose(steps)


# --------------------------------------------------------------------------- #
# label decoding
# --------------------------------------------------------------------------- #
def decode_rgb_labels(
    rgb: np.ndarray, palette: Sequence[Tuple[int, int, int]] = CLASS_COLOURS
) -> np.ndarray:
    """Map an RGB label image to class indices via the nearest palette colour.

    Nearest-colour rather than exact match, because redistributed copies of the
    dataset have often been through a lossy re-encode that shifts label pixels
    by a few counts.
    """
    flat = rgb.reshape(-1, 3).astype(np.int16)
    colours = np.asarray(palette, dtype=np.int16)
    distance = np.abs(flat[:, None, :] - colours[None, :, :]).sum(axis=2)
    return distance.argmin(axis=1).astype(np.uint8).reshape(rgb.shape[:2])


def read_image(path: Path, in_channels: int = 3) -> np.ndarray:
    """Read a SAR patch as ``(H, W, C)`` uint8."""
    with Image.open(path) as handle:
        image = handle.convert("L" if in_channels == 1 else "RGB")
        # copy(): PIL hands back a read-only buffer, which torch complains about
        array = np.array(image, dtype=np.uint8, copy=True)
    return array[..., np.newaxis] if array.ndim == 2 else array


def read_label(path: Path, num_classes: int = len(CLASS_NAMES)) -> np.ndarray:
    """Read a label image as ``(H, W)`` uint8 class indices."""
    with Image.open(path) as handle:
        if handle.mode in ("RGB", "RGBA", "P"):
            array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            # An RGB file whose values are all tiny is an index map saved as RGB.
            if array.max() < num_classes:
                return array[..., 0].copy()
            return decode_rgb_labels(array)
        return np.asarray(handle.convert("L"), dtype=np.uint8).copy()


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class MKLabOilSpillDataset(Dataset):
    """Image/mask pairs from the MKLab oil-spill dataset.

    Yields ``{"image": FloatTensor(C, H, W), "mask": LongTensor(H, W),
    "stem": str}``.

    Args:
        root: dataset root, e.g. ``data/raw/mklab``. May also point straight at
            a split directory that itself contains ``images/`` and labels.
        split: subdirectory to read (``"train"``/``"test"``), or ``None`` when
            ``root`` is already the split.
        transform: albumentations pipeline; built from ``train`` when omitted.
        stems: restrict to these filename stems - how the train/val split is
            made without duplicating files on disk.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        split: Optional[str] = "train",
        transform: Optional[A.Compose] = None,
        train: bool = True,
        stems: Optional[Sequence[str]] = None,
        images_dirname: Optional[str] = None,
        labels_dirname: Optional[str] = None,
        num_classes: Optional[int] = None,
        in_channels: Optional[int] = None,
        image_size: Optional[int] = None,
        normalize: str = "imagenet",
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        cfg = self.settings.training

        base = Path(root) if root is not None else self.settings.paths.resolve(
            cfg.dataset_dir
        )
        self.root = base / split if split else base
        if not self.root.is_dir():
            raise DatasetError(
                f"dataset split not found at {self.root}. Expected the MKLab "
                "layout <root>/<split>/images and <root>/<split>/labels_1D; see "
                "the module docstring if your download is arranged differently."
            )

        self.num_classes = num_classes or cfg.num_classes
        self.in_channels = in_channels or cfg.in_channels
        self.class_names = list(cfg.class_names)[: self.num_classes]

        self.images_dir = self._resolve_dir(images_dirname, IMAGE_DIRNAMES, "images")
        self.labels_dir = self._resolve_dir(labels_dirname, LABEL_DIRNAMES, "labels")

        self.pairs = self._index_pairs(stems)
        if not self.pairs:
            raise DatasetError(f"no image/label pairs found under {self.root}")

        self.transform = transform or build_augmentations(
            train=train,
            image_size=image_size,
            normalize=normalize,
            in_channels=self.in_channels,
        )
        logger.info("%s: %d pair(s) from %s", type(self).__name__, len(self.pairs), self.root)

    # -- discovery -------------------------------------------------------- #
    def _resolve_dir(
        self, explicit: Optional[str], candidates: Sequence[str], kind: str
    ) -> Path:
        if explicit:
            path = self.root / explicit
            if not path.is_dir():
                raise DatasetError(f"{kind} directory not found: {path}")
            return path
        for name in candidates:
            path = self.root / name
            if path.is_dir():
                return path
        raise DatasetError(
            f"no {kind} directory under {self.root}; tried {list(candidates)}. "
            f"Pass {kind}_dirname= if your download names it differently."
        )

    def _index_pairs(
        self, stems: Optional[Sequence[str]]
    ) -> List[Tuple[Path, Path]]:
        """Pair every image with its label by filename stem."""
        labels = {
            path.stem: path
            for suffix in LABEL_SUFFIXES
            for path in sorted(self.labels_dir.glob(f"*{suffix}"))
        }
        wanted = set(stems) if stems is not None else None

        pairs: List[Tuple[Path, Path]] = []
        missing: List[str] = []
        for suffix in IMAGE_SUFFIXES:
            for image_path in sorted(self.images_dir.glob(f"*{suffix}")):
                if wanted is not None and image_path.stem not in wanted:
                    continue
                label_path = labels.get(image_path.stem)
                if label_path is None:
                    missing.append(image_path.name)
                    continue
                pairs.append((image_path, label_path))

        if missing:
            raise DatasetError(
                f"{len(missing)} image(s) under {self.images_dir} have no label in "
                f"{self.labels_dir}, e.g. {missing[:3]}. Labels are matched by "
                "filename stem."
            )
        return sorted(pairs, key=lambda pair: pair[0].stem)

    @property
    def stems(self) -> List[str]:
        return [image.stem for image, _ in self.pairs]

    # -- torch API -------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_path, label_path = self.pairs[index]
        image = read_image(image_path, self.in_channels)
        mask = read_label(label_path, self.num_classes)

        if image.shape[:2] != mask.shape[:2]:
            raise DatasetError(
                f"{image_path.name}: image is {image.shape[:2]} but its label is "
                f"{mask.shape[:2]}"
            )
        if mask.max() >= self.num_classes:
            raise DatasetError(
                f"{label_path.name}: label index {int(mask.max())} is outside the "
                f"{self.num_classes}-class scheme {self.class_names}. If this is "
                "the RGB label folder, point labels_dirname at labels_1D."
            )

        augmented = self.transform(image=image, mask=mask)
        return {
            "image": augmented["image"].float(),
            "mask": augmented["mask"].long(),
            "stem": image_path.stem,
        }

    # -- inspection ------------------------------------------------------- #
    def class_pixel_counts(self, limit: Optional[int] = None) -> np.ndarray:
        """Pixels per class across the split - the input to class weighting.

        Reads labels straight off disk, skipping augmentation, so it stays cheap
        and deterministic. ``limit`` samples only the first N pairs.
        """
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, label_path in self.pairs[:limit]:
            mask = read_label(label_path, self.num_classes)
            counts += np.bincount(mask.ravel(), minlength=self.num_classes)[
                : self.num_classes
            ]
        return counts

    def summary(self) -> Dict[str, Any]:
        counts = self.class_pixel_counts()
        total = max(int(counts.sum()), 1)
        return {
            "root": str(self.root),
            "pairs": len(self.pairs),
            "pixels_per_class": dict(zip(self.class_names, counts.tolist())),
            "fraction_per_class": {
                name: round(float(c) / total, 6)
                for name, c in zip(self.class_names, counts.tolist())
            },
        }


# --------------------------------------------------------------------------- #
# splitting and weighting
# --------------------------------------------------------------------------- #
def train_val_datasets(
    root: Optional[Path] = None,
    split: Optional[str] = "train",
    val_split: Optional[float] = None,
    seed: Optional[int] = None,
    image_size: Optional[int] = None,
    normalize: str = "imagenet",
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> Tuple[MKLabOilSpillDataset, MKLabOilSpillDataset]:
    """Split one folder into train and validation datasets.

    Two dataset objects over disjoint stem lists rather than a ``Subset``, so
    the validation half gets deterministic centre crops and no flips while the
    training half is augmented.
    """
    settings = settings or get_settings()
    cfg = settings.training
    val_split = cfg.val_split if val_split is None else val_split
    seed = cfg.seed if seed is None else seed

    probe = MKLabOilSpillDataset(
        root=root, split=split, train=False, settings=settings,
        image_size=image_size, normalize=normalize, **kwargs
    )
    stems = probe.stems

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(stems))
    n_val = max(1, int(round(len(stems) * val_split))) if len(stems) > 1 else 0
    val_stems = [stems[i] for i in order[:n_val]]
    train_stems = [stems[i] for i in order[n_val:]]

    if not train_stems:
        raise DatasetError(
            f"val_split={val_split} leaves no training samples out of {len(stems)}"
        )

    shared = dict(
        root=root, split=split, settings=settings, image_size=image_size,
        normalize=normalize, **kwargs
    )
    train_ds = MKLabOilSpillDataset(train=True, stems=train_stems, **shared)
    val_ds = MKLabOilSpillDataset(train=False, stems=val_stems, **shared)
    logger.info("split %d pair(s) into %d train / %d val",
                len(stems), len(train_ds), len(val_ds))
    return train_ds, val_ds


def inverse_frequency_weights(
    counts: np.ndarray, max_weight: float = 50.0
) -> torch.Tensor:
    """Class weights that lift rare classes, normalised to mean 1.

    Oil pixels are a fraction of a percent of this dataset, so an unweighted
    cross-entropy converges happily to "everything is sea". Weighting by inverse
    frequency is the explicit correction; the cap stops a class that appears in
    a handful of pixels from dominating the gradient entirely.
    """
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    weights = np.ones_like(counts)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights = np.clip(weights, 0.0, max_weight)
    weights[~present] = 0.0
    mean = weights[present].mean() if present.any() else 1.0
    return torch.as_tensor(weights / max(mean, 1e-12), dtype=torch.float32)


def describe_class_balance(counts: np.ndarray, names: Sequence[str] = CLASS_NAMES) -> str:
    """One-line-per-class breakdown for the training log."""
    total = max(int(np.sum(counts)), 1)
    lines = [f"{'class':<12}{'pixels':>14}{'share':>10}"]
    for name, count in zip(names, np.asarray(counts).tolist()):
        lines.append(f"{name:<12}{int(count):>14,}{100.0 * count / total:>9.4f}%")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# synthetic data (tests and the training smoke test)
# --------------------------------------------------------------------------- #
def write_synthetic_dataset(
    root: Path,
    n_samples: int = 6,
    size: int = 64,
    split: str = "train",
    num_classes: int = len(CLASS_NAMES),
    seed: int = 0,
    rgb_labels: bool = False,
) -> Path:
    """Write a tiny fake MKLab-shaped dataset - for tests and smoke runs only.

    Each sample is speckled "sea" with a darker oil blob, a brighter look-alike
    patch, a small bright ship and a land strip, so every class is represented.
    """
    rng = np.random.default_rng(seed)
    split_dir = Path(root) / split
    images_dir = split_dir / "images"
    labels_dir = split_dir / ("labels" if rgb_labels else "labels_1D")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    half = max(size // 2, 2)
    for i in range(n_samples):
        image = rng.integers(90, 150, size=(size, size), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        # oil: a dark blob in the upper-left quadrant
        r0, c0 = rng.integers(0, half - 1, size=2)
        image[r0 : r0 + half // 2, c0 : c0 + half // 2] = rng.integers(10, 40)
        mask[r0 : r0 + half // 2, c0 : c0 + half // 2] = OIL_CLASS

        # look-alike: also dark, deliberately similar to the oil blob
        if num_classes > LOOK_ALIKE_CLASS:
            r1 = rng.integers(half, size - 2)
            c1 = rng.integers(0, half)
            image[r1 : r1 + half // 3, c1 : c1 + half // 3] = rng.integers(20, 50)
            mask[r1 : r1 + half // 3, c1 : c1 + half // 3] = LOOK_ALIKE_CLASS

        # ship: a couple of very bright pixels
        if num_classes > 3:
            image[2:4, size - 4 : size - 2] = 255
            mask[2:4, size - 4 : size - 2] = 3

        # land: a strip down the right-hand edge
        if num_classes > 4:
            image[:, size - 2 :] = rng.integers(150, 220)
            mask[:, size - 2 :] = 4

        Image.fromarray(image, mode="L").convert("RGB").save(
            images_dir / f"sample_{i:03d}.jpg", quality=95
        )
        label_image = (
            Image.fromarray(_encode_rgb_labels(mask), mode="RGB")
            if rgb_labels
            else Image.fromarray(mask, mode="L")
        )
        label_image.save(labels_dir / f"sample_{i:03d}.png")

    return split_dir


def _encode_rgb_labels(
    mask: np.ndarray, palette: Sequence[Tuple[int, int, int]] = CLASS_COLOURS
) -> np.ndarray:
    """Class indices back to the dataset's RGB palette."""
    lut = np.asarray(palette, dtype=np.uint8)
    return lut[np.clip(mask, 0, len(palette) - 1)]


__all__ = [
    "CLASS_NAMES",
    "CLASS_COLOURS",
    "OIL_CLASS",
    "LOOK_ALIKE_CLASS",
    "DatasetError",
    "MKLabOilSpillDataset",
    "build_augmentations",
    "train_val_datasets",
    "inverse_frequency_weights",
    "describe_class_balance",
    "decode_rgb_labels",
    "read_image",
    "read_label",
    "write_synthetic_dataset",
]
