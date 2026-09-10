"""Tests for Sentinel-1 SAFE-format ingestion.

Creates a small synthetic SAFE-like .zip fixture with VV and VH measurement
TIFFs (rasterio-generated, not real Sentinel-1 data) and exercises
:func:`safe_measurement_members`, :func:`is_safe_archive`, and
:func:`read_safe_bands`.
"""

from __future__ import annotations

import io
import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from src.ingestion.safe import (
    SAFEError,
    is_safe_archive,
    read_safe_bands,
    safe_measurement_members,
)


def _make_measurement_tiff(
    polarisation: str,
    *,
    width: int = 64,
    height: int = 48,
    crs: CRS = CRS.from_epsg(4326),
    transform: Affine | None = None,
) -> bytes:
    """Render a small GeoTIFF in memory, return the raw bytes."""
    if transform is None:
        transform = from_bounds(10.0, 55.0, 11.0, 56.0, width, height)

    buf = io.BytesIO()
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        data = np.random.default_rng(42).uniform(-25.0, -10.0, (height, width)).astype(
            np.float32
        )
        dst.write(data, 1)
        dst.set_band_description(1, polarisation)
    return buf.getvalue()


def _build_safe_zip(target: Path, *, width: int = 64, height: int = 48) -> None:
    """Write a minimal SAFE-structured .zip to *target*."""
    scene_name = "S1A_IW_GRDH_1SDV_20230514T003332_20230514T003357_048456_05D45D_1A2B"
    vv_path = f"{scene_name}.SAFE/measurement/s1a-iw-grd-vv-20230514t003332-20230514t003357-048456-05d45d-001.tiff"
    vh_path = f"{scene_name}.SAFE/measurement/s1a-iw-grd-vh-20230514t003332-20230514t003357-048456-05d45d-002.tiff"

    vv_bytes = _make_measurement_tiff("VV", width=width, height=height)
    vh_bytes = _make_measurement_tiff("VH", width=width, height=height)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{scene_name}.SAFE/manifest.safe", "<fake/>")
        zf.writestr(vv_path, vv_bytes)
        zf.writestr(vh_path, vh_bytes)


@pytest.fixture()
def safe_zip(tmp_path: Path) -> Path:
    """A small synthetic SAFE .zip fixture."""
    zpath = tmp_path / "test_safe.zip"
    _build_safe_zip(zpath, width=32, height=24)
    return zpath


# ---- is_safe_archive ---- #

def test_is_safe_archive_true(safe_zip: Path) -> None:
    assert is_safe_archive(safe_zip) is True


def test_is_safe_archive_false_for_tif(tmp_path: Path) -> None:
    tif = tmp_path / "scene.tif"
    tif.write_bytes(b"not a real tif")
    assert is_safe_archive(tif) is False


def test_is_safe_archive_false_for_bad_zip(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    assert is_safe_archive(bad) is False


# ---- safe_measurement_members ---- #

def test_safe_measurement_members_finds_vv_vh(safe_zip: Path) -> None:
    members = safe_measurement_members(safe_zip)
    assert members is not None
    assert "VV" in members
    assert "VH" in members
    assert members["VV"].endswith(".tiff")
    assert members["VH"].endswith(".tiff")


def test_safe_measurement_members_none_for_non_zip(tmp_path: Path) -> None:
    tif = tmp_path / "scene.tif"
    tif.write_bytes(b"data")
    assert safe_measurement_members(tif) is None


def test_safe_measurement_members_raises_on_bad_zip(tmp_path: Path) -> None:
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(SAFEError, match="could not read archive"):
        safe_measurement_members(bad)


def test_safe_measurement_members_raises_on_missing_polarisation(tmp_path: Path) -> None:
    scene = "S1A_IW_GRDH_1SDV_20230514T003332_20230514T003357_048456_05D45D_1A2B"
    vv_path = f"{scene}.SAFE/measurement/s1a-iw-grd-vv-20230514t003332-20230514t003357-048456-05d45d-001.tiff"
    zpath = tmp_path / "vv_only.zip"
    vv_bytes = _make_measurement_tiff("VV")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(vv_path, vv_bytes)
    with pytest.raises(SAFEError, match="missing measurement TIFF"):
        safe_measurement_members(zpath)


# ---- read_safe_bands ---- #

def test_read_safe_bands_shape_and_dtype(safe_zip: Path) -> None:
    data, transform, crs, nodata, gcps, gcp_crs, band_names = read_safe_bands(safe_zip)
    assert data.ndim == 3
    assert data.shape[0] == 2  # VV + VH
    assert data.dtype == np.float32
    assert band_names == ["VV", "VH"]


def test_read_safe_bands_georef(safe_zip: Path) -> None:
    data, transform, crs, nodata, gcps, gcp_crs, _ = read_safe_bands(safe_zip)
    assert isinstance(transform, Affine)
    # The fixture uses EPSG:4326
    assert crs is not None


def test_read_safe_bands_vv_before_vh(safe_zip: Path) -> None:
    """Band 0 must be VV, band 1 must be VH."""
    data, *_ = read_safe_bands(safe_zip)
    # Both bands are random; just check shape
    assert data.shape[0] == 2


def test_read_safe_bands_rejects_band_selection(safe_zip: Path) -> None:
    with pytest.raises(SAFEError, match="band selection is not supported"):
        read_safe_bands(safe_zip, bands=[1])


def test_read_safe_bands_raises_on_non_safe(tmp_path: Path) -> None:
    tif = tmp_path / "not_safe.tif"
    tif.write_bytes(b"nope")
    with pytest.raises(SAFEError, match="does not appear to be a Sentinel-1 SAFE archive"):
        read_safe_bands(tif)


# ---- integration with pipeline._read_raster ---- #

def test_read_raster_delegates_to_safe(safe_zip: Path) -> None:
    """pipeline._read_raster should produce the same result as read_safe_bands."""
    from src.preprocessing.pipeline import _read_raster

    data_pipe, tf_pipe, crs_pipe, nd_pipe, gcps_pipe, gcp_crs_pipe, names_pipe = (
        _read_raster(safe_zip, bands=None)
    )
    data_safe, tf_safe, crs_safe, nd_safe, gcps_safe, gcp_crs_safe, names_safe = (
        read_safe_bands(safe_zip)
    )

    np.testing.assert_array_equal(data_pipe, data_safe)
    assert tf_pipe == tf_safe
    assert names_pipe == names_safe
