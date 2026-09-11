# src/api backend — FastAPI + rasterio/geopandas/torch + Postgres/PostGIS.
#
# GDAL-preinstalled base: rasterio/geopandas/fiona need libgdal at the
# system level; building it from source in a plain python:3.11-slim image
# is slow and fragile. The OSGeo image ships it, prebuilt, matched to a
# known-good GDAL version.
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.3

# The OSGeo Ubuntu image ships python3 (3.12 on this base, not the 3.11
# this repo develops against) but no pip. Every dependency in
# requirements.txt publishes 3.12-compatible wheels, so this is a deploy
# platform difference to know about, not a functional one.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# A venv, not --break-system-packages: the base image's system Python
# already has apt-managed packages (e.g. numpy, for GDAL's own Python
# bindings) that pip can't safely upgrade in place (no RECORD file since
# apt installed them, not pip) — a venv sidesteps that conflict entirely
# rather than fighting it package by package.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .

# requirements.txt pins a bare torch==2.14.0 (no +cpu) — correct for local
# Windows dev, where PyPI's own wheel happens to be CPU-only, but on Linux
# PyPI's torch wheel bundles full CUDA (~2GB extra, and a needless pull on
# a CPU-only deploy target). Installing it explicitly from the CPU-only
# index FIRST, pinned to the exact same version, satisfies the later plain
# `torch==2.14.0` in requirements.txt (pip accepts an installed local
# version like 2.14.0+cpu against a non-local `==2.14.0` constraint and
# leaves it alone) without needing to touch that pin.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY migrations ./migrations

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Shell form, not exec form: Render's Docker runtime injects $PORT and
# expects the container to bind to it (it auto-detects the EXPOSEd port
# only as a fallback when $PORT isn't set) - the exec-form CMD this
# replaced hardcoded 8000, which only ever worked because that fallback
# happened to match. ${PORT:-8000} keeps `docker run` with no PORT set
# (local testing) working the same as before.
CMD python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
