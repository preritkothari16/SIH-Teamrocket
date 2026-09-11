"""Sentinel-1 SAFE-format ingestion.

Reads VV+VH measurement TIFFs from either a SAFE .zip archive, via GDAL's
``/vsizip/`` virtual filesystem (no physical extraction needed), or an
already-extracted ``.SAFE`` directory (opened directly off disk) — e.g. a
product someone unzipped by hand rather than one ``CDSECatalogue.fetch()``
downloaded itself. Both paths converge on the same measurement-lookup and
band-reading logic; only *how a member's bytes are opened* differs.

Usage::

    from src.ingestion.safe import read_safe_bands, is_safe_archive

    if is_safe_archive(path):
        data, transform, crs, nodata, gcps, gcp_crs, band_names = read_safe_bands(path)
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

import logging

logger = logging.getLogger(__name__)

_SAFE_MEASUREMENT_RE = re.compile(
    r"(?:^|/)measurement/.*-(?P<polarisation>vv|vh)-.*\.tiff?$", re.IGNORECASE
)


class SAFEError(RuntimeError):
    """A SAFE archive could not be read or is missing required data."""


def _is_safe_dir(path: Path) -> bool:
    return path.is_dir() and path.name.upper().endswith(".SAFE")


def is_safe_archive(path: Path) -> bool:
    """True if *path* is a .zip or an extracted ``.SAFE`` directory
    containing a Sentinel-1 SAFE measurement structure."""
    path = Path(path)
    if _is_safe_dir(path):
        return safe_measurement_members(path) is not None
    if path.suffix.lower() != ".zip":
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if _SAFE_MEASUREMENT_RE.search(name):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def safe_measurement_members(path: Path) -> Optional[Dict[str, str]]:
    """Locate VV and VH measurement TIFFs inside a SAFE archive or directory.

    Returns a dict ``{"VV": "<member-or-file-path>", "VH": "..."}`` or
    ``None`` when the path is neither a .zip nor a ``.SAFE`` directory, or
    does not contain Sentinel-1 measurements. For a directory, values are
    real filesystem paths (as strings); for a .zip, archive-relative member
    names — :func:`read_safe_bands` knows how to open each.

    Raises :class:`SAFEError` when the archive/directory is validly a SAFE
    product but is missing one of the two required polarisations.
    """
    path = Path(path)

    if _is_safe_dir(path):
        measurement_dir = path / "measurement"
        if not measurement_dir.is_dir():
            return None
        members: Dict[str, str] = {}
        for member_path in measurement_dir.iterdir():
            match = _SAFE_MEASUREMENT_RE.search(f"measurement/{member_path.name}")
            if match:
                members[match.group("polarisation").upper()] = str(member_path)
        if not members:
            return None
        missing = {"VV", "VH"} - members.keys()
        if missing:
            raise SAFEError(
                f"SAFE directory {path} is missing measurement TIFF(s): {', '.join(sorted(missing))}"
            )
        return members

    if path.suffix.lower() != ".zip":
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            members = {}
            for name in archive.namelist():
                match = _SAFE_MEASUREMENT_RE.search(name)
                if match:
                    members[match.group("polarisation").upper()] = name
    except zipfile.BadZipFile as exc:
        raise SAFEError(f"could not read archive {path}: {exc}") from exc

    if not members:
        return None
    missing = {"VV", "VH"} - members.keys()
    if missing:
        raise SAFEError(
            f"SAFE archive {path} is missing measurement TIFF(s): {', '.join(sorted(missing))}"
        )
    return members


def read_safe_bands(
    path: Path,
    bands: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Affine, Optional[CRS], Optional[float], Any, Optional[CRS], List[str]]:
    """Read VV+VH from a SAFE .zip and stack them into a 2-band array.

    Parameters
    ----------
    path : Path
        Path to a Sentinel-1 SAFE .zip archive.
    bands : ignored
        Present for API compatibility with ``_read_raster``; band selection is
        not supported for SAFE archives (both VV and VH are always read).

    Returns
    -------
    data : np.ndarray
        ``float32`` array of shape ``(2, height, width)`` — band 0 is VV,
        band 1 is VH.  Values are whatever the TIFF stores (typically DN or
        sigma0 linear).
    transform : Affine
        Geotransform from the first measurement TIFF.
    crs : CRS or None
        Coordinate reference system (often None for GRD products that carry GCPs).
    nodata : float or None
        Nodata value, if any.
    gcps : list of GCP
        Ground control points.
    gcp_crs : CRS or None
        CRS of the GCPs.
    band_names : list of str
        ``["VV", "VH"]``.
    """
    path = Path(path)
    if bands is not None:
        raise SAFEError("band selection is not supported for Sentinel-1 SAFE archives")

    members = safe_measurement_members(path)
    if members is None:
        raise SAFEError(f"{path} does not appear to be a Sentinel-1 SAFE archive")

    is_dir = _is_safe_dir(path)
    rasters: List[np.ndarray] = []
    reference: Optional[Tuple[Tuple[int, int, Affine, Optional[CRS]], Optional[float], Any]] = None

    for polarisation in ("VV", "VH"):
        # A directory's member is already a real file path; a .zip member
        # is opened in place through GDAL's virtual filesystem instead of
        # extracting it to disk first.
        member_path = (
            members[polarisation] if is_dir
            else f"/vsizip/{path.resolve().as_posix()}/{members[polarisation]}"
        )
        with rasterio.open(member_path) as dataset:
            grid = (dataset.width, dataset.height, dataset.transform, dataset.crs)
            if reference is None:
                reference = grid, dataset.nodata, dataset.get_gcps()
            elif grid != reference[0]:
                raise SAFEError(
                    f"SAFE measurements in {path} do not share a raster grid"
                )
            rasters.append(dataset.read(1).astype(np.float32))

    assert reference is not None
    (width, height, transform, crs), nodata, (gcps, gcp_crs) = reference

    logger.info(
        "read SAFE %s: VV+VH %dx%d, crs=%s",
        path.name, width, height, crs,
    )

    return np.stack(rasters), transform, crs, nodata, gcps, gcp_crs, ["VV", "VH"]


__all__ = [
    "SAFEError",
    "is_safe_archive",
    "safe_measurement_members",
    "read_safe_bands",
]
