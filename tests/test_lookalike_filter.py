"""Look-alike filter tests.

Synthetic blobs painted into a synthetic backscatter field: an elongated,
high-contrast, sharp-edged streak that should read as oil, and a round, faint,
soft-edged patch that should be rejected.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from src.config import load_settings
from src.detection.dataset import LOOK_ALIKE_CLASS, OIL_CLASS
from src.detection.lookalike_filter import (
    BlobVerdict,
    apply_to_class_mask,
    blob_contrast,
    blob_edge_gradient,
    blob_elongation,
    classify_blob,
    filter_lookalikes,
    label_oil_blobs,
)

SCENE = (160, 160)
SEA_DB = -10.0


def sea_field(shape=SCENE, seed: int = 0) -> np.ndarray:
    """Flat water with a little texture, in dB."""
    rng = np.random.default_rng(seed)
    return (SEA_DB + rng.normal(0.0, 0.2, size=shape)).astype(np.float32)


def paint_ellipse(
    shape, centre, semi_major: float, semi_minor: float, angle_deg: float = 0.0
) -> np.ndarray:
    """Boolean ellipse mask - the shape a slick or a look-alike is drawn as."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    dr = rows - centre[0]
    dc = cols - centre[1]
    theta = np.radians(angle_deg)
    major = dc * np.cos(theta) + dr * np.sin(theta)
    minor = -dc * np.sin(theta) + dr * np.cos(theta)
    return (major / semi_major) ** 2 + (minor / semi_minor) ** 2 <= 1.0


def oil_like_blob(shape=SCENE):
    """Long, thin, 8 dB darker than the sea, with a hard edge."""
    return paint_ellipse(shape, (45, 80), semi_major=45.0, semi_minor=7.0, angle_deg=20.0)


def lookalike_blob(shape=SCENE):
    """Round, only 1 dB darker, and blurred so its edge is soft."""
    return paint_ellipse(shape, (120, 60), semi_major=18.0, semi_minor=17.0)


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def scene():
    """A field holding both blobs, plus the class mask that calls both oil."""
    image = sea_field()
    mask = np.zeros(SCENE, dtype=np.uint8)

    oil = oil_like_blob()
    image[oil] = SEA_DB - 8.0  # deep, sharp-edged damping
    mask[oil] = OIL_CLASS

    faint = lookalike_blob()
    soft = ndimage.gaussian_filter(faint.astype(np.float32), sigma=6.0)
    image = image - 1.0 * soft  # shallow, gradual dip
    mask[faint] = OIL_CLASS

    return image.astype(np.float32), mask, oil, faint


# --------------------------------------------------------------------------- #
# blob extraction
# --------------------------------------------------------------------------- #
def test_label_oil_blobs_finds_both(scene) -> None:
    _, mask, _, _ = scene
    labels, count = label_oil_blobs(mask)
    assert count == 2
    assert labels.shape == mask.shape
    assert set(np.unique(labels)) == {0, 1, 2}


def test_label_oil_blobs_ignores_other_classes() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:8, 4:8] = LOOK_ALIKE_CLASS
    mask[20:24, 20:24] = 4  # land
    _, count = label_oil_blobs(mask)
    assert count == 0


def test_diagonal_threads_keep_a_slick_in_one_piece() -> None:
    """8-connectivity, so a necked slick is one blob and not two small ones."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:7, 4:7] = OIL_CLASS
    mask[7:10, 7:10] = OIL_CLASS  # touches only at a corner
    assert label_oil_blobs(mask, connectivity=2)[1] == 1
    assert label_oil_blobs(mask, connectivity=1)[1] == 2


def test_label_accepts_a_boolean_mask() -> None:
    boolean = np.zeros((16, 16), dtype=bool)
    boolean[2:6, 2:6] = True
    assert label_oil_blobs(boolean)[1] == 1


# --------------------------------------------------------------------------- #
# the three measurements
# --------------------------------------------------------------------------- #
def test_elongation_separates_a_streak_from_a_disc() -> None:
    streak = np.argwhere(paint_ellipse(SCENE, (80, 80), 40.0, 5.0))
    disc = np.argwhere(paint_ellipse(SCENE, (80, 80), 20.0, 20.0))
    assert blob_elongation(streak)[0] > 5.0
    assert blob_elongation(disc)[0] == pytest.approx(1.0, abs=0.1)


def test_elongation_orientation_follows_the_streak() -> None:
    horizontal = np.argwhere(paint_ellipse(SCENE, (80, 80), 40.0, 5.0, angle_deg=0.0))
    assert blob_elongation(horizontal)[1] % 180 == pytest.approx(0.0, abs=5.0)


def test_elongation_of_a_single_pixel_is_one() -> None:
    assert blob_elongation(np.array([[5, 5]]))[0] == 1.0


def test_contrast_is_measured_against_the_surrounding_ring(scene) -> None:
    image, _, oil, _ = scene
    inside, surround, contrast = blob_contrast(image, oil, ring_pixels=10)
    assert inside == pytest.approx(SEA_DB - 8.0, abs=0.5)
    assert surround == pytest.approx(SEA_DB, abs=0.5)
    assert contrast == pytest.approx(8.0, abs=0.6)


def test_contrast_is_near_zero_for_a_patch_that_is_not_dark() -> None:
    image = sea_field()
    blob = paint_ellipse(SCENE, (80, 80), 15.0, 15.0)
    assert abs(blob_contrast(image, blob)[2]) < 0.5


def test_edge_gradient_is_higher_for_a_sharp_edge(scene) -> None:
    image, _, oil, faint = scene
    assert blob_edge_gradient(image, oil) > blob_edge_gradient(image, faint)


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #
def test_the_two_blobs_are_treated_differently(scene, settings) -> None:
    """The point of the whole module: one passes, one does not."""
    image, mask, _, _ = scene
    result = filter_lookalikes(mask, image, settings=settings)

    assert len(result.blobs) == 2
    assert len(result.kept) == 1
    assert len(result.rejected) == 1

    kept, rejected = result.kept[0], result.rejected[0]
    assert kept.elongation > rejected.elongation
    assert kept.contrast_db > rejected.contrast_db
    assert rejected.reasons, "a rejection must say what it failed"


def test_the_oil_streak_passes_every_test(scene, settings) -> None:
    image, mask, oil, _ = scene
    verdict = classify_blob(oil, 1, image, settings=settings)
    assert verdict.is_oil
    assert verdict.reasons == []
    assert verdict.contrast_db > settings.detection.lookalike_min_contrast_db
    assert verdict.elongation > settings.detection.lookalike_min_elongation


def test_the_round_faint_patch_is_rejected(scene, settings) -> None:
    image, _, _, faint = scene
    verdict = classify_blob(faint, 2, image, settings=settings)
    assert not verdict.is_oil
    joined = " ".join(verdict.reasons)
    assert "contrast" in joined or "elongation" in joined


def test_the_filtered_mask_keeps_only_the_survivor(scene, settings) -> None:
    image, mask, oil, faint = scene
    result = filter_lookalikes(mask, image, settings=settings)

    assert result.mask[oil].mean() > 0.99
    assert not result.mask[faint].any()


def test_a_tiny_blob_is_dropped_as_speckle(settings) -> None:
    image = sea_field()
    mask = np.zeros(SCENE, dtype=np.uint8)
    mask[80:83, 80:83] = OIL_CLASS  # 9 px, well under min_blob_pixels
    image[80:83, 80:83] = SEA_DB - 12.0

    result = filter_lookalikes(mask, image, settings=settings)
    assert result.kept == []
    assert "px" in result.rejected[0].reasons[0]


def test_verdict_carries_the_measurements(scene, settings) -> None:
    image, mask, _, _ = scene
    blob = filter_lookalikes(mask, image, settings=settings).blobs[0]
    payload = blob.to_dict()
    for key in ("contrast_db", "elongation", "edge_gradient", "is_oil", "reasons"):
        assert key in payload
    assert isinstance(blob, BlobVerdict)
    assert blob.verdict in ("oil", "look-alike")


def test_mean_confidence_is_averaged_over_the_blob(scene, settings) -> None:
    image, mask, oil, _ = scene
    confidence = np.full(SCENE, 0.4, dtype=np.float32)
    confidence[oil] = 0.9

    result = filter_lookalikes(mask, image, confidence=confidence, settings=settings)
    kept = result.kept[0]
    assert kept.mean_confidence == pytest.approx(0.9, abs=0.01)


def test_summary_names_every_blob_and_its_verdict(scene, settings) -> None:
    image, mask, _, _ = scene
    text = filter_lookalikes(mask, image, settings=settings).summary()
    assert "oil" in text and "look-alike" in text
    assert text.count("blob") >= 2


# --------------------------------------------------------------------------- #
# wind, which is not wired in yet
# --------------------------------------------------------------------------- #
def test_wind_speed_defaults_to_unused(scene, settings) -> None:
    """Step 3.1 supplies real wind; until then the result must not change."""
    image, mask, _, _ = scene
    without = filter_lookalikes(mask, image, settings=settings)
    explicit_none = filter_lookalikes(mask, image, wind_speed=None, settings=settings)
    assert [b.is_oil for b in without.blobs] == [b.is_oil for b in explicit_none.blobs]


def test_wind_outside_the_band_rejects_even_a_good_blob(scene, settings) -> None:
    """The rule is already written, so it works the moment wind data arrives."""
    image, _, oil, _ = scene
    assert classify_blob(oil, 1, image, settings=settings, wind_speed=1.0).is_oil is False
    assert classify_blob(oil, 1, image, settings=settings, wind_speed=25.0).is_oil is False
    assert classify_blob(oil, 1, image, settings=settings, wind_speed=7.0).is_oil is True


# --------------------------------------------------------------------------- #
# writing the decision back
# --------------------------------------------------------------------------- #
def test_rejected_blobs_become_look_alike_not_sea(scene, settings) -> None:
    """Rejected pixels stay marked as something dark, not silently erased."""
    image, mask, oil, faint = scene
    result = filter_lookalikes(mask, image, settings=settings)

    updated = apply_to_class_mask(mask, result)
    assert (updated[oil] == OIL_CLASS).all()
    assert (updated[faint] == LOOK_ALIKE_CLASS).all()
    assert (updated == 0).sum() == (mask == 0).sum()


def test_multiband_backscatter_uses_the_first_band(scene, settings) -> None:
    image, mask, _, _ = scene
    stacked = np.stack([image, image + 5.0])
    single = filter_lookalikes(mask, image, settings=settings)
    multi = filter_lookalikes(mask, stacked, settings=settings)
    assert [b.is_oil for b in single.blobs] == [b.is_oil for b in multi.blobs]


def test_shape_mismatch_is_reported(settings) -> None:
    with pytest.raises(ValueError, match="but the mask is"):
        filter_lookalikes(np.zeros((10, 10), np.uint8), np.zeros((8, 8), np.float32),
                          settings=settings)


def test_an_empty_mask_yields_no_blobs(settings) -> None:
    result = filter_lookalikes(
        np.zeros(SCENE, np.uint8), sea_field(), settings=settings
    )
    assert result.blobs == [] and not result.mask.any()
