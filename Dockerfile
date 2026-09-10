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

# torch==2.14.0+cpu (requirements.txt) is a PyTorch-CPU-index-only build —
# it does not exist on plain PyPI. --extra-index-url adds that index
# without replacing PyPI for everything else in the file.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
