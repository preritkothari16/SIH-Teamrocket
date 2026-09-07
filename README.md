# sar-oilspill

Detects marine oil spills in Sentinel-1 SAR imagery and attributes them to the
vessel most likely responsible. Preprocessed SAR scenes are segmented by a deep
learning model, dark patches are filtered against wind and contrast criteria to
reject look-alikes, and confirmed slicks are characterized by area, thickness
class and estimated volume. Each confirmed spill then fans out in parallel to
AIS-based vessel attribution and to a Lagrangian drift forecast, both surfaced
on a dashboard alongside the alert. Built for Smart India Hackathon.

## Repo layout

```
sar-oilspill/
├── configs/                  # config.yaml + AOI geojson
├── data/
│   ├── raw/                  # downloaded SAR products      (gitignored)
│   ├── processed/            # calibrated / speckle-filtered (gitignored)
│   └── ais/                  # AIS extracts                  (gitignored)
├── models/                   # trained segmentation weights  (gitignored)
├── notebooks/                # exploration + figures
├── src/
│   ├── config.py             # pydantic settings (yaml + env overrides)
│   ├── ingestion/            # 1.x  satellite search & download
│   ├── preprocessing/        # 2.x  calibration, speckle filter, reprojection
│   ├── detection/            # 3.x  segmentation + look-alike filter
│   ├── characterization/     # 3.x  area, thickness, volume
│   ├── alerts/               # 4.x  alert manager
│   ├── env_data/             # wind / wave / current retrieval
│   ├── ais/                  # 5.x  AIS retrieval
│   ├── attribution/          # 5.x  spill-to-vessel matching
│   ├── drift/                # 5.x  drift prediction
│   └── output/               # FastAPI dashboard + exports
├── tests/
├── requirements.txt
└── .env.example
```

## Setup

Python 3.11 is the target runtime.

```bash
# 1. clone and enter the repo
cd sar-oilspill

# 2. create and activate a virtual environment
py -3.11 -m venv .venv          # Windows
.venv\Scripts\activate
# python3.11 -m venv .venv      # macOS / Linux
# source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. run the tests
pytest
```

### Environment variables

Copy `.env.example` to `.env` and fill in your credentials. `.env` is
gitignored — never commit real secrets.

```bash
cp .env.example .env
```

| Variable | Where to get it |
| --- | --- |
| `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` | [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) — Sentinel-1 search and download |
| `CDS_API_KEY` | [Climate Data Store](https://cds.climate.copernicus.eu) — ERA5 wind and wave fields |
| `CMEMS_USERNAME` / `CMEMS_PASSWORD` | [Copernicus Marine Service](https://marine.copernicus.eu) — ocean currents for drift |

Non-secret settings live in `configs/config.yaml`. Any of them can be
overridden per-run by an environment variable using the `SAROIL_` prefix and
`__` between nesting levels:

```bash
SAROIL_DETECTION__PROB_THRESHOLD=0.65
SAROIL_AIS__SEARCH_WINDOW_HOURS=12
```

Read settings from code with:

```python
from src.config import get_settings

settings = get_settings()
settings.detection.prob_threshold
```

## Running the full pipeline

Once a scene is on disk, run detection, alerting, and (if AIS data is on
hand) vessel attribution in one call:

```bash
python scripts/run_pipeline.py --scene path/to/scene.tif --stub-model
```

`--stub-model` is required until a real checkpoint exists — a dark-pixel
threshold stands in for the trained detector, and says so in its own output.
Add `--ais path/to/export.csv` (or `.parquet`) to also run AIS-based vessel
attribution for any spill that alerts; without it, an alerted spill still
gets a full alert decision, just an empty vessel list rather than a crash.

This writes one combined JSON per run — by default
`data/processed/<scene_id>/pipeline_result.json` — holding, per spill: the
spill object (polygon, area, confidence, …), the alert decision (including
which rule(s) drove it — confidence, area, wind, exclusion zone, dedup), and
the ranked vessel list (empty if not alerted or no `--ais` given).

Add `--map` to also save a basic Folium map next to it (`map.html` by
default, or `--map-output <path>`), showing every spill's polygon and any
candidate vessel tracks, colour-coded by rank, with a popup per vessel naming
its score and explanation string. Add `--report` to also save a standalone
HTML incident report (`report.html` by default, or `--report-output <path>`)
— the same map embedded, plus the ranked vessel table and drift forecast.

## Running the demo (web API + frontend)

Two processes, no credentials needed for the local demo path — the demo
scene is pre-downloaded and already processed with `--stub-model`, so
none of the CDSE/CDS/CMEMS variables above are required.

**1. Backend** (from the repo root, with `.venv` active):

```bash
.venv/Scripts/python.exe -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Serves the JSON API the frontend consumes: `GET /api/runs` (every processed
scene under `data/processed/`), `GET /api/runs/{scene_id}` (full detection +
alert + vessels + drift for one scene), `GET /api/runs/{scene_id}/report`
(the standalone HTML incident report, if `--report` was run for that scene),
and `POST /api/runs` (trigger `run_pipeline.py` synchronously on a scene —
fine for a demo, not for concurrent use). At least one scene needs a
`pipeline_result.json` under `data/processed/<scene_id>/` for `/api/runs` to
return anything — run the pipeline command above first if `data/processed/`
is empty.

**2. Frontend** (separate terminal, from `frontend/`):

```bash
npm install
cp .env.example .env   # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev
```

Open the printed local URL (Vite's default is `http://localhost:5173`). The
scene dropdown's "Backend runs" group lists every scene the API found; mock
fixture scenarios stay available below it as an offline fallback if the
backend isn't running.

**What a judge sees, running the included demo case (scene `00000`, a real
2048² Sentinel-1-shaped scene, 304 detected candidate blobs):** selecting it
populates the map with the one alerted slick's polygon and its 6/12/24/48h
drift forecast, the spill panel with its confidence/area/bearing/centroid,
the alert badge reading "Possible" (this scene's wind check didn't clear the
bar for "active" — see the note below), and the vessel list explicitly
reading "No vessel attribution data" (no AIS export was supplied for this
run — attribution isn't skipped, there's just nothing to attribute against).
A "Report" button appears for any real backend run and opens the same
scene's full HTML incident report in a new tab.

A few things worth saying out loud rather than letting a judge ask: every
other one of the 304 candidate blobs in this scene shows "rejected" (mostly
on minimum-area), which is normal — one alert per scene is realistic, not a
sign the detector under-fired; the "active" alert status never appears in
this environment because it requires a real wind-speed reading and this
demo runs on a local wind fixture, not live ERA5; and the vessel list being
empty here is a supply gap (no AIS export shipped with the repo), not a
broken attribution path — `scripts/run_attribution.py`'s own tests exercise
that path against synthetic AIS.

## Roadmap

- **1.1** Satellite product search
- **1.2** Product download
- **1.3** AOI subsetting
- **2.1** Radiometric calibration
- **2.2** Speckle filtering
- **2.3** Terrain correction and reprojection
- **2.4** Land masking
- **2.5** Tiling and normalization
- **3.1** Segmentation model training
- **3.2** Spill inference
- **3.3** Look-alike filtering
- **3.4** Spill polygonization
- **3.5** Spill characterization
- **4.1** Environmental data retrieval
- **4.2** Confidence scoring
- **4.3** Alert manager
- **5.1** AIS retrieval
- **5.2** Vessel track reconstruction
- **5.3** Vessel attribution
- **5.4** Drift prediction
- **5.5** Dashboard and API

## Future work

- SAR-based dark-vessel detection in the same scene, to catch a spiller with its AIS transponder off entirely.
- Optical (Sentinel-2) fusion, to confirm and better delineate a slick where cloud-free imagery overlaps the SAR acquisition.
- Oil weathering, thickness, and volume estimation from multi-band/multi-temporal signatures, beyond today's single-thickness-class area estimate.
- Full behavioral-anomaly ML for attribution, learning suspicious AIS patterns directly rather than the two hand-built factors in `src/attribution/anomaly_features.py`.
- Automated live alert notifications (email/SMS/webhook) triggered directly from the alert manager's decisions.
