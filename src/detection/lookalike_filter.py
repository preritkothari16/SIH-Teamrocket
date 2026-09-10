"""Reject look-alikes from the segmentation model's oil mask.

The segmentation model works on texture and shape in a 512 px window. Plenty of
things on the sea surface are just as dark as oil in SAR - low-wind cells,
biogenic films, rain cells, upwelling, current shear - and at tile scale they
can look identical. This stage applies physical rules to each connected oil
blob and drops the ones whose *properties* do not behave like a slick, which is
also what makes the decision explainable in a demo: every rejection names the
test it failed and the number it failed on.

Three measurements per blob, each rejecting a different look-alike family:

``contrast_db``
    Mean backscatter inside the blob minus the mean of a ring around it. Oil
    damps capillary waves, so a real slick sits several dB below the water
    around it. A weak dip is usually a low-wind patch or a shadow, not oil.

``elongation``
    Major over minor axis of the blob's second-moment ellipse. Slicks are
    stretched by wind and current into streaks and comet shapes; low-wind cells
    and rain footprints are round or amorphous. This is the single most useful
    discriminator when the wind is unknown.

``edge_gradient``
    Mean backscatter gradient across the blob boundary, in dB per pixel. Oil
    has a sharp, well-defined edge because the damping either happens or does
    not. Low-wind areas fade gradually into the surrounding sea.

``wind_speed`` is accepted and **not yet used** - Step 3.1 wires real wind data
in. When it arrives the rule is already known: below
``detection.lookalike_min_wind_speed_ms`` the sea is too calm for a dark patch
to mean anything, and above ``lookalike_max_wind_speed_ms`` a real slick gets
mixed away and stops being visible, so a dark patch outside that band is
suspect regardless of its shape.

Thresholds live in the config, not here, because they are the numbers most
likely to need tuning against real data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from src.config import Settings, get_settings
from src.detection.dataset import LOOK_ALIKE_CLASS, OIL_CLASS

logger = logging.getLogger(__name__)


@dataclass
class BlobVerdict:
    """One connected oil blob, its measurements, and why it was kept or dropped."""

    blob_id: int
    pixels: int
    centroid: Tuple[float, float]  # (row, col)
    bbox: Tuple[int, int, int, int]  # (row_min, col_min, row_max, col_max)
    mean_db: float
    surround_db: float
    contrast_db: float
    elongation: float
    edge_gradient: float
    orientation_deg: float
    mean_confidence: float
    is_oil: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "oil" if self.is_oil else "look-alike"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blob_id": self.blob_id,
            "pixels": self.pixels,
            "centroid_row": self.centroid[0],
            "centroid_col": self.centroid[1],
            "mean_db": self.mean_db,
            "surround_db": self.surround_db,
            "contrast_db": self.contrast_db,
            "elongation": self.elongation,
            "edge_gradient": self.edge_gradient,
            "orientation_deg": self.orientation_deg,
            "mean_confidence": self.mean_confidence,
            "is_oil": self.is_oil,
            "reasons": list(self.reasons),
        }


def _detail(blob: "BlobVerdict") -> str:
    return (
        f"contrast {blob.contrast_db:+.1f} dB, elongation {blob.elongation:.2f}, "
        f"edge {blob.edge_gradient:.2f} dB/px"
    )


@dataclass
class FilterResult:
    """What the filter made of one scene."""

    mask: np.ndarray  # (H, W) bool - the surviving oil pixels
    labels: np.ndarray  # (H, W) int32 - blob ids, 0 = background
    blobs: List[BlobVerdict]

    @property
    def kept(self) -> List[BlobVerdict]:
        return [b for b in self.blobs if b.is_oil]

    @property
    def rejected(self) -> List[BlobVerdict]:
        return [b for b in self.blobs if not b.is_oil]

    def rejection_counts(self) -> Dict[str, int]:
        """How many blobs failed on each test, keyed by the test's name."""
        counts: Dict[str, int] = {}
        for blob in self.rejected:
            for reason in blob.reasons:
                key = reason.split(" ")[0] if not reason.startswith("only") else "size"
                counts[key] = counts.get(key, 0) + 1
        return counts

    def summary(self, max_kept: int = 10, max_rejected: int = 5) -> str:
        """The largest kept blobs, then a sample of rejections and why they failed.

        Both lists are capped. An untrained or thresholding model can leave
        thousands of dark specks, and printing every one of them buries the
        result; the tallies below still account for all of them.
        """
        kept = sorted(self.kept, key=lambda b: b.pixels, reverse=True)
        lines = [f"{len(kept)} oil / {len(self.blobs)} candidate blob(s)"]
        for blob in kept[:max_kept]:
            lines.append(f"  blob {blob.blob_id}: oil  ({_detail(blob)}, {blob.pixels} px)")
        if len(kept) > max_kept:
            total_px = sum(b.pixels for b in kept)
            lines.append(
                f"  ... {len(kept) - max_kept} more kept, {total_px:,} oil px in total"
            )

        rejected = self.rejected
        for blob in rejected[:max_rejected]:
            lines.append(
                f"  blob {blob.blob_id}: look-alike  ({_detail(blob)})"
                "  <- " + "; ".join(blob.reasons)
            )
        if len(rejected) > max_rejected:
            tally = ", ".join(
                f"{name} {count}" for name, count in sorted(self.rejection_counts().items())
            )
            lines.append(
                f"  ... {len(rejected) - max_rejected} more rejected "
                f"(failed: {tally})"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# blob measurements
# --------------------------------------------------------------------------- #
def label_oil_blobs(
    mask: np.ndarray, oil_class: int = OIL_CLASS, connectivity: int = 2
) -> Tuple[np.ndarray, int]:
    """Connected components of the oil class.

    8-connectivity by default, so a slick that necks down to a diagonal thread
    stays one blob rather than being split into fragments that each then look
    too small to matter.
    """
    binary = mask.astype(bool) if mask.dtype == bool else (mask == oil_class)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, count = ndimage.label(binary, structure=structure)
    return labels.astype(np.int32), int(count)


def blob_elongation(coords: np.ndarray) -> Tuple[float, float]:
    """(elongation, orientation in degrees) of a blob's second-moment ellipse.

    Elongation is the ratio of the ellipse's axes, so 1.0 is a disc and larger
    is more streak-like. A blob thinner than one pixel in either direction gets
    a floor on the minor axis, which keeps single-pixel-wide streaks finite.
    """
    if coords.shape[0] < 2:
        return 1.0, 0.0

    centred = coords - coords.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    if not np.all(np.isfinite(covariance)):
        return 1.0, 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor, major = np.sqrt(eigenvalues[0]), np.sqrt(eigenvalues[1])

    elongation = float(major / max(minor, 0.5))
    principal = eigenvectors[:, 1]
    orientation = float(np.degrees(np.arctan2(principal[0], principal[1])) % 180.0)
    return elongation, orientation


def blob_contrast(
    image: np.ndarray, blob: np.ndarray, ring_pixels: int = 15
) -> Tuple[float, float, float]:
    """(inside mean, ring mean, contrast) in dB for one blob.

    The ring is a dilation of the blob minus the blob itself, so the comparison
    is against the water immediately around the slick rather than the whole
    scene - which matters, because backscatter varies across a scene with
    incidence angle and wind.
    """
    inside = image[blob]
    inside = inside[np.isfinite(inside)]
    if inside.size == 0:
        return float("nan"), float("nan"), 0.0

    dilated = ndimage.binary_dilation(
        blob, structure=ndimage.generate_binary_structure(2, 2), iterations=ring_pixels
    )
    ring = dilated & ~blob
    surround = image[ring]
    surround = surround[np.isfinite(surround)]

    inside_mean = float(inside.mean())
    if surround.size == 0:
        return inside_mean, float("nan"), 0.0

    surround_mean = float(surround.mean())
    # positive contrast = the blob is darker than the water around it
    return inside_mean, surround_mean, surround_mean - inside_mean


def blob_edge_gradient(image: np.ndarray, blob: np.ndarray) -> float:
    """Mean gradient magnitude on the blob boundary, in dB per pixel."""
    boundary = blob & ~ndimage.binary_erosion(blob)
    if not boundary.any():
        return 0.0

    filled = np.nan_to_num(image, nan=float(np.nanmean(image)) if np.isfinite(image).any() else 0.0)
    gy = ndimage.sobel(filled, axis=0, mode="nearest") / 8.0
    gx = ndimage.sobel(filled, axis=1, mode="nearest") / 8.0
    magnitude = np.hypot(gy, gx)
    return float(magnitude[boundary].mean())


# --------------------------------------------------------------------------- #
# the filter
# --------------------------------------------------------------------------- #
def classify_blob(
    blob: np.ndarray,
    blob_id: int,
    backscatter: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    settings: Optional[Settings] = None,
    wind_speed: Optional[float] = None,
    offset: Tuple[int, int] = (0, 0),
) -> BlobVerdict:
    """Measure one blob and decide whether it behaves like oil.

    ``blob`` and ``backscatter`` may be crops around the blob rather than
    full-scene arrays - see :func:`filter_lookalikes`, which crops so that a
    scene full of small blobs does not cost a full-scene dilation each. Row and
    column positions are reported back in scene coordinates via ``offset``.
    """
    settings = settings or get_settings()
    cfg = settings.detection
    row_offset, col_offset = offset

    pixels = int(np.count_nonzero(blob))

    # Blobs covering a huge fraction of the scene are never real oil spills;
    # skip the expensive argwhere/centroid/elongation measurements for them.
    MAX_BLOB_PIXELS = 5_000_000
    if pixels > MAX_BLOB_PIXELS:
        return BlobVerdict(
            blob_id=0,
            pixels=pixels,
            reasons=[f"blob too large ({pixels:,} px, max {MAX_BLOB_PIXELS:,})"],
            centroid=(0.0, 0.0),
            bbox=(0, 0, 0, 0),
            mean_db=0.0, surround_db=0.0, contrast_db=0.0,
            elongation=1.0, edge_gradient=0.0, orientation_deg=0.0,
            mean_confidence=0.0, is_oil=False,
        )

    coords = np.argwhere(blob)
    centroid = (
        (float(coords[:, 0].mean()) + row_offset, float(coords[:, 1].mean()) + col_offset)
        if pixels
        else (0.0, 0.0)
    )
    bbox = (
        int(coords[:, 0].min()) + row_offset, int(coords[:, 1].min()) + col_offset,
        int(coords[:, 0].max()) + row_offset, int(coords[:, 1].max()) + col_offset,
    ) if pixels else (0, 0, 0, 0)

    mean_db, surround_db, contrast_db = blob_contrast(
        backscatter, blob, cfg.lookalike_ring_pixels
    )
    elongation, orientation = blob_elongation(coords)
    edge_gradient = blob_edge_gradient(backscatter, blob)
    mean_confidence = (
        float(np.nanmean(confidence[blob])) if confidence is not None and pixels else 1.0
    )

    reasons: List[str] = []

    # Too small to be a reportable spill, and small blobs make every other
    # measurement here unstable - a handful of pixels has no meaningful shape.
    if pixels < cfg.min_blob_pixels:
        reasons.append(f"only {pixels} px (min {cfg.min_blob_pixels})")

    # Oil damps capillary waves hard; a shallow dip is a low-wind patch or a
    # shadow, not a slick.
    if contrast_db < cfg.lookalike_min_contrast_db:
        reasons.append(
            f"contrast {contrast_db:.1f} dB below {cfg.lookalike_min_contrast_db} dB"
        )

    # Wind and current stretch slicks into streaks. Round dark patches are the
    # classic low-wind / rain-cell false positive.
    if elongation < cfg.lookalike_min_elongation:
        reasons.append(
            f"elongation {elongation:.2f} below {cfg.lookalike_min_elongation}"
        )

    # A slick's edge is sharp because the damping is a step change. Low-wind
    # areas fade into the sea around them.
    if edge_gradient < cfg.lookalike_min_edge_gradient:
        reasons.append(
            f"edge {edge_gradient:.2f} dB/px below {cfg.lookalike_min_edge_gradient}"
        )

    # Not applied yet - Step 3.1 supplies real wind fields. Recorded so a run
    # made before that lands is not mistaken for one that checked the wind.
    if wind_speed is not None:
        if wind_speed < cfg.lookalike_min_wind_speed_ms:
            reasons.append(
                f"wind {wind_speed:.1f} m/s below {cfg.lookalike_min_wind_speed_ms} "
                "m/s - sea too calm for a dark patch to mean oil"
            )
        elif wind_speed > cfg.lookalike_max_wind_speed_ms:
            reasons.append(
                f"wind {wind_speed:.1f} m/s above {cfg.lookalike_max_wind_speed_ms} "
                "m/s - a slick would be mixed away"
            )

    return BlobVerdict(
        blob_id=blob_id,
        pixels=pixels,
        centroid=centroid,
        bbox=bbox,
        mean_db=mean_db,
        surround_db=surround_db,
        contrast_db=contrast_db,
        elongation=elongation,
        edge_gradient=edge_gradient,
        orientation_deg=orientation,
        mean_confidence=mean_confidence,
        is_oil=not reasons,
        reasons=reasons,
    )


def filter_lookalikes(
    mask: np.ndarray,
    backscatter: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    wind_speed: Optional[float] = None,
    oil_class: int = OIL_CLASS,
    settings: Optional[Settings] = None,
) -> FilterResult:
    """Keep only the oil blobs whose physical properties behave like a slick.

    Args:
        mask: full-scene class indices from inference, or a boolean oil mask.
        backscatter: the scene's dB backscatter, same shape as ``mask``. When
            multi-band, the first band is used - VV is the polarisation slicks
            show up in.
        confidence: optional per-pixel model confidence, averaged per blob and
            carried through to characterization.
        wind_speed: **accepted but not applied yet** - Step 3.1 wires in real
            wind data. Passing it now records the check in each verdict.

    Returns a :class:`FilterResult` holding the surviving mask, the blob labels
    and every blob's measurements with a pass/fail reason.
    """
    settings = settings or get_settings()

    if backscatter.ndim == 3:
        backscatter = backscatter[0]
    if backscatter.shape != mask.shape:
        raise ValueError(
            f"backscatter is {backscatter.shape} but the mask is {mask.shape}"
        )

    labels, count = label_oil_blobs(mask, oil_class)
    kept = np.zeros(mask.shape, dtype=bool)
    verdicts: List[BlobVerdict] = []

    # Each blob is measured inside its own padded bounding box. Measuring in
    # place would dilate the whole scene once per blob, which on a 2048px scene
    # with hundreds of candidates is minutes of work for no extra information -
    # every measurement here is local to the blob and its immediate surrounds.
    margin = settings.detection.lookalike_ring_pixels + 2
    windows = ndimage.find_objects(labels)

    for blob_id, window in enumerate(windows, start=1):
        if window is None:
            continue
        rows, cols = _pad_window(window, mask.shape, margin)
        blob = labels[rows, cols] == blob_id

        verdict = classify_blob(
            blob,
            blob_id,
            backscatter[rows, cols],
            confidence[rows, cols] if confidence is not None else None,
            settings,
            wind_speed,
            offset=(rows.start, cols.start),
        )
        verdicts.append(verdict)
        if verdict.is_oil:
            kept[rows, cols] |= blob

    logger.info(
        "look-alike filter kept %d of %d blob(s)",
        sum(v.is_oil for v in verdicts), count,
    )
    return FilterResult(mask=kept, labels=labels, blobs=verdicts)


def _pad_window(
    window: Tuple[slice, slice], shape: Tuple[int, int], margin: int
) -> Tuple[slice, slice]:
    """Grow a blob's bounding box by ``margin``, clipped to the scene."""
    rows, cols = window
    return (
        slice(max(rows.start - margin, 0), min(rows.stop + margin, shape[0])),
        slice(max(cols.start - margin, 0), min(cols.stop + margin, shape[1])),
    )


def apply_to_class_mask(
    mask: np.ndarray,
    result: FilterResult,
    oil_class: int = OIL_CLASS,
    lookalike_class: int = LOOK_ALIKE_CLASS,
) -> np.ndarray:
    """Rewrite rejected oil pixels in a class mask as look-alike.

    Keeps the full multi-class mask meaningful after filtering, instead of
    quietly turning rejected slicks into sea and losing the fact that something
    dark was there at all.
    """
    updated = mask.copy()
    rejected = (mask == oil_class) & ~result.mask
    updated[rejected] = lookalike_class
    return updated


__all__ = [
    "BlobVerdict",
    "FilterResult",
    "filter_lookalikes",
    "classify_blob",
    "label_oil_blobs",
    "blob_elongation",
    "blob_contrast",
    "blob_edge_gradient",
    "apply_to_class_mask",
]
