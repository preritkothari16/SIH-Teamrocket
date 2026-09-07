# sar-oilspill

SAR satellite oil-spill detection + vessel attribution. Smart India Hackathon.
Python 3.11, Windows. **Read this instead of re-auditing the codebase.**

Pipeline: ingestion → preprocessing → detection → characterization → alerts →
[AIS → attribution] ∥ [drift] → dashboard.

## Status

| Step | Module | State |
|---|---|---|
| 1.1 ingestion | `src/ingestion/` | **done** — CDSE catalogue + local disk source |
| 1.2 preprocessing | `src/preprocessing/pipeline.py` | **done** — calibrate → despeckle → geocode → mask land → tile |
| 1.3 training | `src/detection/{dataset,model,train}.py` | **done, never trained** — no GPU, no dataset here |
| 1.4 inference + stitching | `src/detection/infer.py` | **done** — overlaps averaged, not trimmed |
| 1.5 look-alike filter | `src/detection/lookalike_filter.py` | **done** — contrast / elongation / edge rules |
| 1.6 characterization | `src/characterization/spill_object.py` | **done** — equal-area polygons → GeoJSON |
| Phase 1 chain | `scripts/run_detection.py` | **done** — 1.1→1.6 for one scene |
| 2.1 AIS load + query | `src/ais/{loader,query}.py` | **done** — space-time box around a spill |
| 2.2 tracks + filter | `src/ais/{tracks,filter}.py` | **done** — interpolated CPA, stationarity screen |
| 2.3 attribution scoring | `src/attribution/scoring.py` | **done** — 4-factor weighted score + explanation |
| Phase 2 chain | `scripts/run_attribution.py` | **done, synthetic-validated only** — see Open Questions |
| 3.1 wind lookup | `src/env_data/wind.py` | **done, local-fixture-only** — no CDS credentials here |
| 3.2 alert registry + manager | `src/alerts/{registry,manager}.py` | **done** — sqlite registry, 5-rule decision + explain trail |
| Phase 3 chain | `scripts/run_pipeline.py` | **done** — 1.1→3.2→(2.1-2.3) in one call, real-scene verified |
| 3.4 map output | `src/output/map.py` | **done** — basic Folium map, `--map` on the Phase 3 chain |
| 4.1 currents + env service | `src/env_data/{currents,service,grid}.py` | **done, local-fixture-only** — no CMEMS credentials here |
| 4.2 particle drift model | `src/drift/particle_model.py` | **done** — pure numpy Lagrangian stepper, same function hindcasts (`dt` sign flip) |
| 4.3 forward drift forecast | `src/drift/forward.py` | **done** — 6/12/24/48h horizons, constant-environment assumption (see below) |
| 4.4 hindcasting + attribution | `src/drift/hindcast.py`, `src/attribution/scoring.py` | **done** — origin corridor; `attribution.use_hindcasting` (default off) swaps it into scoring |
| 4.5 drift wired into chain | `scripts/run_pipeline.py`, `src/output/map.py` | **done** — every spill gets a forecast; map/report/dashboard all overlay it |
| 5.1 live ingestion polling | `src/ingestion/poller.py`, `scripts/run_live.py` | **done, never run against real CDSE** — stub-catalogue-tested only |
| 5.2 AIS anomaly features | `src/attribution/anomaly_features.py` | **done** — gap + speed-change bonus factors, wired into scoring |
| 5.3 incident report | `src/output/report.py` | **done** — standalone HTML (not PDF), `--report` on the Phase 3-4 chain |
| 5.4 dashboard | `src/output/dashboard.py` | **done** — FastAPI, one page, toggleable map layers, real-scene verified |
| 5.5 web API + frontend | `src/api/{main,models,registry}.py`, `frontend/` | **done** — separate FastAPI JSON API + Vite/TS SPA, real-scene verified |

`pytest` → 486 passing, ~55-60s, fully offline. Run it before believing anything here.

Phase 1 runs end to end: `scripts/run_detection.py --scene <tif> --stub-model`
takes ~72 s on a 2048² scene. **There is still no trained checkpoint**, so
`--stub-model` (a dark-pixel threshold, not a detector) is the only way to run
the chain today; it labels its own output as such.

Phase 2 runs end to end: `scripts/run_attribution.py --spill <geojson> --ais <csv|parquet>`
— query → tracks → filter → scoring, writes ranked candidates as JSON. Needs an
AIS export (source-agnostic loader, see `src/ais/loader.py`); nothing ships
with the repo, so this has only been run against synthetic fixtures so far.

Phase 3 is the real end-to-end script: `scripts/run_pipeline.py --scene <tif>
--stub-model [--ais <csv|parquet>] [--map]` — imports `run_detection.run()`
and `run_attribution.attribute_spill()` as libraries (no shelling out),
chains detection → alert manager → (if alerted) attribution, writes one
combined JSON, and optionally a Folium map. **Verified against a real local
scene** from the 1200-tile dataset below (2048², `--stub-model`), not just
synthetic fixtures.

Phase 4 (drift) is wired all the way through as of step 4.5. `src/drift/
forward.py` samples wind + current **once**, at the spill's centroid and
acquisition time, and holds that single vector pair constant for the whole
run — there is no real forecast time series here (same reason as the
wind/currents fixtures below), so a 48h-out forecast is honestly wrong the
moment the real wind shifts, by design, not a bug. Hindcasting
(`src/drift/hindcast.py`) reuses the forward stepper with `dt` negated, per
`particle_model.py`'s own design — there is no separate backward code path.
`attribution.use_hindcasting` (config, default `false`) is the only thing
that turns the hindcast corridor into an actual scoring change; when off,
`scoring.py` is byte-for-byte the step-2.3 static-buffer path.
`scripts/run_pipeline.py` computes a forward forecast for **every** spill
(alerted or not — it needs neither AIS nor alert status) and, only when
`use_hindcasting` is on, a hindcast corridor for each attributed spill.

Phase 5 is final polish, all four steps independent of each other:

- **5.1 live polling** (`src/ingestion/poller.py` + `scripts/run_live.py`):
  `poll_once()` drives any `SceneSource` (real `CDSECatalogue` or a test
  stub) — search since the last poll (strictly newer, so the boundary scene
  is never re-fetched), download, run the pipeline. **Never run against the
  real CDSE network** — no credentials here, tested only against a stubbed
  catalogue (`tests/test_poller.py`).
- **5.2 AIS anomaly features** (`src/attribution/anomaly_features.py`): two
  bonus scoring factors — an AIS reporting gap, and slowing/stopping near
  the slick relative to the vessel's own baseline speed. Each scores 0 (not
  a penalty) when there isn't enough evidence. `attribution.weight_gap` /
  `weight_speed_change` (0.075 each by default); the other four weights were
  trimmed proportionally to keep the sum at 1.0. **Synthetic-validated
  only** — no real AIS export has ever had a genuine transponder gap or
  slowdown run through this.
- **5.3 incident report** (`src/output/report.py`): standalone HTML (not
  PDF — no new rendering dependency, and it reuses `build_map()`'s own
  embed), `--report`/`--report-output` on `run_pipeline.py`.
- **5.4 dashboard** (`src/output/dashboard.py`): FastAPI (already a
  dependency; Streamlit is not installed here), one page at `/` per
  `python -m src.output.dashboard --result <pipeline_result.json>` —
  metadata, the same toggleable map, alert status, ranked vessel table.
  Loads one result once at startup; not wired to the live poller.

- **5.5 web API + frontend** (`src/api/{main,models,registry}.py`,
  `frontend/`): a separate FastAPI app from 5.4's `dashboard.py` — this one
  is a pure JSON API (`GET /api/runs`, `GET /api/runs/{id}`, `GET
  /api/runs/{id}/report`, `POST /api/runs`) with no server-rendered HTML,
  meant to be consumed by `frontend/`'s Vite + vanilla TypeScript SPA (no
  React — plain classes: `App`, `MapView`, `SpillPanel`, `VesselList`,
  `DriftTimeline`, `AlertBadge`, one page, no router yet). `src/api/
  registry.py` scans `data/processed/` and translates either a Phase 3
  `pipeline_result.json` or a Phase-1-only `spills/*.geojson` into the
  frontend's `PipelineRun` contract (`frontend/src/types/schema.ts` ==
  `src/api/models.py`, field-for-field, checked by `tests/test_api.py`).
  CORS is opened for the Vite dev server (5173/5174, both localhost and
  127.0.0.1); `frontend/vite.config.ts` also proxies `/api` to
  `localhost:8000` so `frontend/src/api/client.ts` can use relative paths in
  dev — `VITE_API_BASE_URL` (`frontend/.env`, gitignored;
  `frontend/.env.example` tracked) overrides this for a build with no dev
  proxy. MSW is a `package.json` devDependency but was never wired up (no
  handlers) — the actual offline-demo fallback is simpler: `App.ts`'s
  dropdown fetches the static fixtures under `frontend/src/mocks/fixtures/`
  directly when not pointed at a real backend run. **Real-scene verified**
  (scene `00000`, 304 real spills) via a live `uvicorn` + `npm run dev` pair
  and a Playwright smoke check — no console errors, real drift/spill/alert
  data rendered.
  - A run's alert status has to be picked from *some* spill in the scene
    (`data["spills"]` is a list) and then translated from the backend's raw
    word to the frontend's word — both steps are centralized in
    `registry.py` specifically because they drifted apart once already:
    `_select_target_spill()` (prefers the first *alerted* spill over
    `spills[0]` — an alert firing anywhere in the scene matters more than
    array order) and `_map_alert_status()` (`active`→`new`, `possible`→
    `possible`, `rejected`→`none`; frontend's `"update"` word is reserved
    for a future re-detection case, not produced yet). **Both**
    `GET /api/runs` (`_summary_from_pipeline`) and `GET /api/runs/{id}`
    (`_pipeline_to_contract` → `_spill_entry_to_contract`) call both
    helpers — don't let a third code path reimplement either rule inline.
  - The real local dataset (scene `00000`) only ever produces raw statuses
    `"rejected"` (303/304 spills) and `"possible"` (1/304) — `"active"`
    requires `alerts/manager.py`'s wind-speed check to actually pass, which
    needs real wind data this environment doesn't have. Don't take that as
    "the active/new case can't happen" — it's reachable by design the
    moment real wind data is, and `tests/test_api.py::TestStatusWordConsistency`
    covers all three words synthetically since the real dataset can't.

`src/output/map.py::build_map()` underlies all three of 4.5/5.3/5.4: slick,
drift forecast, hindcast corridor (compute-on-demand, never persisted in the
combined JSON — only the dashboard actually computes and passes it in), and
vessel tracks are each their own `folium.FeatureGroup` under one
`folium.LayerControl`, so any one layer can be toggled off in the rendered
map itself; this is what "layer toggle" means in step 5.4, not
dashboard-specific code.

## Environment facts

- venv at `.venv/` (Python 3.11.9). Run things as `.venv/Scripts/python.exe -m ...`.
- **torch is CPU-only** (`2.14.0+cpu`). No CUDA. Real training must happen elsewhere.
- `data/raw/mklab/` does **not** exist — the labelled training set was never downloaded.
- A different dataset does exist: `D:\SIH\01_Train_Val_Oil_Spill_images\Oil\` —
  1200 GeoTIFFs, 48.7 GB, 2048², 2-band float32 **dB** (VH, VV), EPSG:4326,
  globally scattered (North Sea, Gulf of Mexico, Nile delta).
  **It has no labels and no acquisition timestamps** (filenames are `00000.tif`,
  no TIFF tags), so it can feed preprocessing but not training, and not AIS
  attribution. `LocalSceneSource` reads all 1200 in ~13 s.
- Vendored coastline covers **only the Gujarat AOI**. For the dataset above,
  call `download_coastline()` first or nothing gets masked as land.
- **No CDS API access here**: no `cdsapi` package, no `CDS_API_KEY`/`~/.cdsapirc`.
  `get_wind()` needs an explicit `dataset=`/`env_data.wind_dataset_path`, or it
  raises naming what's missing — real ERA5 has never been fetched or read in
  this environment, only a synthetic fixture matching its layout.
- **No CMEMS credentials here either**: no `copernicusmarine` package, no
  `CMEMS_USERNAME`/`CMEMS_PASSWORD`. `get_current()` needs an explicit
  `dataset=`/`env_data.current_dataset_path` for the same reason — real
  CMEMS/HYCOM current data has never been fetched here.
- **`fastapi`, `uvicorn`, `httpx` are already project dependencies**
  (`requirements.txt`; `config.yaml`'s `api:` block existed before the
  dashboard did) — step 5.4 needed no new install. Streamlit is not
  installed here; that's why the dashboard is FastAPI, not a preference.

## Architecture decisions worth not relitigating

- **One scene type.** `src/ingestion/types.py::Scene` is the contract. Both the
  network catalogue and the local reader return it; nothing downstream may
  branch on where pixels came from. Footprints are always shapely in EPSG:4326,
  times always tz-aware UTC.
- **Config is one place.** `configs/config.yaml` → `src/config.py` pydantic
  models. Env overrides via `SAROIL_SECTION__KEY`. `extra="forbid"`, so an
  unknown YAML key fails loudly. Secrets are never in YAML — it stores only the
  *names* of env vars (see `.env.example`).
- **Preprocessing stage order is deliberate**: despeckle *before* geocode, while
  speckle is still uncorrelated. Resampling first ruins the Lee statistics.
- **Land is masked to NaN, never 0.** Zeroed land looks exactly like an oil
  slick to a dark-patch detector.
- **Tile size/overlap live on `detection`, not `preprocessing`** — tiles exist
  to feed the model, so the model owns those numbers.
- **Every tile carries its own geotransform** in `tile_index.parquet`, which is
  how detections get stitched back to scene → map coordinates later.
- **Augmentation is geometry only** (flips/rot90/transpose). No brightness,
  contrast or noise: pixel values are backscatter in dB, and the look-alike
  filter thresholds on them. A test enforces this.
- **Class imbalance is handled explicitly**: inverse-frequency weights from the
  real training split + Dice alongside CE. Best checkpoint selects on **oil
  IoU**, not mean IoU. Oil-vs-look-alike is the whole problem.
- **Tile overlaps are averaged when stitching, not trimmed.** Trimming discards
  the boundary-straddling slick that the overlap exists to catch.
- **Uncovered pixels are `NODATA_CLASS` (255), never sea.** A dropped tile must
  not read as clean water.
- **Areas are computed in a per-feature Lambert azimuthal equal-area CRS.**
  Never in degrees — a square degree is ~12,300 km² and shrinks with latitude.
- **Blob measurements run inside each blob's bounding window.** Measuring in
  place dilated the full scene once per blob: >600 s vs 72 s for one scene.
- **Inference maps dB onto the 0-255 range the model trained on** using
  `detection.input_db_min/max`. That mapping is an assumption, not a
  calibration — revisit once a checkpoint is trained on real tiles.
- **AIS search window is one-sided**: `[acquisition_time - ais.search_window_hours,
  acquisition_time]`, never symmetric. A vessel has to have been near the
  slick at or before it was seen, not after — don't re-add a forward window.
- **CPA is measured against each track's resampled (interpolated) positions**,
  not raw AIS pings (`src/ais/tracks.py`) — AIS sampling is sparse/uneven, so
  the true closest approach often falls between two pings.
- **`orientation_and_elongation()`'s angle is CCW from east**, despite its own
  docstring claiming clockwise — verified empirically by rotating known
  rectangles. `src/attribution/scoring.py::slick_axis_bearing()` does the
  `(90 - angle) % 180` conversion to a compass bearing; don't "fix" the sign
  without re-deriving it, the docstring is the wrong one.
- **The alert registry is sqlite3, not parquet/JSON.** Dedup has to update an
  existing row in place (drift, refined area, escalated status) when a later
  scene re-detects the same event; a flat file would need a full rewrite to
  do that safely. One row per persistent *event*, not per scene detection.
- **`evaluate_alert()` takes wind speed as a plain `Optional[float]`, not a
  wind-dataset lookup.** `process_spill()` is the thin wrapper that calls
  `get_wind()` and swallows `EnvDataError`. Keeps every alert rule testable
  with synthetic inputs and no dataset; an unavailable wind reading is
  treated the same as an out-of-range one (marks "possible", never rejects).
- **`ThresholdStubModel` calls the darkest 15% of a *tile* oil, by rank, not
  by any absolute darkness.** Against a small patch on a flat, noiseless
  background this ties out to near-100% "oil" (the quantile cutoff lands on
  the background's own repeated value). Any synthetic scene fixture needs
  background noise *and* a patch sized close to or above that 15% share, or
  the stub model won't isolate it. Bit both the Phase 1 review's reprojection
  test and Step 3.3's pipeline fixture before this was understood.
- **`env_data/wind.py` and `env_data/currents.py` share one generic lookup**
  (`env_data/grid.py::lookup_vector()`), and `env_data/service.py` wraps both
  behind `get_environment()`, each field independently degrading to `None`
  rather than failing the call — same "unavailable is a fact, not an error"
  treatment `alerts/manager.py` already gave wind before this existed.
- **`particle_model.py`'s hindcast is a sign flip, not a second
  implementation.** `step_particles(dt_hours=...)` displaces by
  `velocity * dt_seconds`; only that final multiply carries `dt`'s sign. The
  Ekman deflection itself does **not** flip — it's a fixed transform of the
  (always forward-sense) wind vector. Diffusion noise is redrawn every call,
  so a forward-then-backward round trip lands close to, not exactly on, the
  start; a `diffusion_std_ms=0` round trip is exact, and a test pins both.
- **Forward forecast and hindcast both sample wind/current once** (at the
  spill's centroid and acquisition time) and hold it constant for the whole
  run — there is no real forecast time series in this environment, only an
  instant field at best. Documented in both modules' docstrings, not a bug to
  "fix" by fetching per-step; swap in a real series later without touching
  the stepping itself.
- **A particle swarm's footprint is its convex hull**, everywhere in
  `drift/`. No density contours or KDE — matches this project's general bias
  toward the simplest representation that is still useful downstream (map
  display, hindcast-vs-vessel-position scoring).
- **`attribution.use_hindcasting` is `false` by default and inert unless a
  corridor is actually passed in** — `score_candidate()`/`score_candidates()`
  take an optional `hindcast_corridor`; with the flag off, or no corridor, or
  a candidate with no `position_at_cpa`, scoring is exactly the step-2.3
  static-buffer path. `CandidateVessel.position_at_cpa` (lon, lat) exists
  solely to let scoring re-measure distance against a *different* polygon
  (the hindcast snapshot nearest the candidate's own CPA time) than the one
  `ais/filter.py` originally screened candidates against.
- **A bonus attribution factor's neutral value is 0, not 0.5.**
  `anomaly_features.py`'s gap/speed-change scores return 0 whenever there
  isn't enough evidence (no SOG, single ping, already-slow baseline) —
  unlike `alignment_score`'s 0.5-when-unknown, because there is no "neutral"
  reading for a bonus factor to split the difference on: an unremarkable
  track just isn't evidence, in either direction.
- **`anomaly_features.py` operates on `CandidateVessel.track` (raw pings)**,
  not `VesselTrack.resampled` — deliberately no interpolation fallback for
  missing SOG (unlike `ais/filter.py::is_stationary()`'s displacement
  fallback), since real AIS sources this project targets (NOAA, DMA) both
  report SOG directly; keeping this simple was the actual tradeoff, not an
  oversight.
- **`report.py` and `dashboard.py` do not share table-rendering code**,
  despite looking similar (both build a vessel table from the same payload
  shape) — the report's is per-spill, the dashboard's aggregates every
  alerted spill with an added spill-id column. Both are ~10-line functions;
  forcing one shared, parameterized version was judged not worth it.

## Conventions

- Comments explain *why*, never *what*. No decorative headers on obvious code.
- Tests use synthetic fixtures (rasterio/numpy/PIL generated in-test). Nothing
  in the suite touches the network or needs a real Sentinel-1 product.
- Errors name the fix: which env var, which path, which config key.
- Commit style: `type(scope): summary`, body explains reasoning in prose.
  Trailers required — see the session's attribution instructions.
- Prefer editing config + a stage function over adding a new module.

## Commands

```bash
.venv/Scripts/python.exe -m pytest                      # 515 tests, offline
.venv/Scripts/python.exe scripts/run_detection.py --scene <tif> --stub-model
.venv/Scripts/python.exe -m src.detection.train --smoke-test   # end-to-end, no data/GPU
.venv/Scripts/python.exe scripts/run_attribution.py --spill <geojson> --ais <csv|parquet>
.venv/Scripts/python.exe scripts/run_pipeline.py --scene <tif> --stub-model --map --report
.venv/Scripts/python.exe scripts/run_live.py --aoi configs/aoi.geojson --interval 900   # needs CDSE creds, untested here
.venv/Scripts/python.exe -m src.output.dashboard --result data/processed/<scene_id>/pipeline_result.json
```

```python
from src.ingestion import LocalSceneSource, load_aoi
from src.preprocessing.pipeline import run_pipeline

scene = LocalSceneSource(root="D:/SIH/01_Train_Val_Oil_Spill_images").scenes()[0]
run_pipeline(scene)          # -> data/processed/<scene_id>/ + tile_index.parquet
```

## Engineering review (2026-09-07)

A full backend review (all phases, no code changes) rated the backend **7/10
— functionally complete, not yet demo/production-hardened**. Architecture and
phase boundaries were confirmed clean; a frontend can be built against the
combined-JSON contract (`spill`/`alert`/`vessels`/`drift_forecast`) as-is.
Findings not yet fixed, in priority order:

**Must fix**
- `requirements.txt` is missing `httpx` — `tests/test_dashboard.py` only
  collects today because `huggingface_hub` happens to pull it in transitively;
  a clean install breaks the test suite.
- No dependency version pins — a live `numpy`/C-extension ABI warning already
  appears in `test_env_currents.py`; pin at least `numpy`, `netCDF4`,
  `rasterio`, `shapely`, `pandas`, `torch`.
- `src/output/map.py`'s popups/tooltips (`_spill_popup`, `_vessel_popup`,
  `_forecast_popup`) don't `html.escape()` their values, unlike
  `report.py`/`dashboard.py` which do — a vessel_name from an AIS export
  renders raw HTML in the map.

**Should fix**
- `src/drift/particle_model.py::scatter_particles()`'s rejection-sampling
  loop hangs forever on a degenerate/zero-area geometry (verified: a
  `LineString` input never returns). Unreachable from the real pipeline today
  (`mask_to_polygon()` never produces one), but a real risk the moment a
  frontend lets a caller supply a hand-edited polygon.
- `scripts/run_live.py`: `poll_once()`/`CDSECatalogue.search()` isn't inside
  the per-scene try/except (only `run_pipeline.run()` is) — one network
  failure permanently kills the "continuous" poller. Also `since` advances
  even when a scene's pipeline run failed, so that scene is never retried.
- `src/output/dashboard.py`'s `GET /` rebuilds the whole map and recomputes
  every alerted spill's hindcast corridor on every request, not once at
  startup; one malformed spill's `hindcast_origin()` call (uncaught) would
  500 the entire page on every reload.
- `src/detection/infer.py::db_to_model_input()` collapses a real 2-band
  VH/VV tile to band-0-repeated-3x instead of using both bands — fix before
  any real training run (currently invisible, no checkpoint exists yet).
- Wind-gated look-alike rejection (`lookalike_filter.py::classify_blob()`)
  is fully built and tested but `scripts/run_detection.py` never passes
  `wind_speed=` — dead in the real run path. Wire it or note it here as
  intentionally unwired.
- `scripts/run_pipeline.py`'s `with SpillRegistry(...) as registry:` block
  (around line 125) holds the sqlite connection open across per-spill drift
  forecasting and AIS attribution, not just registry reads/writes — risks a
  lock timeout if two processes (e.g. the live poller + a manual run) hit the
  same registry file concurrently. `evaluate_alert()`'s dedup check-then-act
  is also a non-atomic read+write race under concurrent writers.
- `SpillRegistry` (`src/alerts/registry.py`) has no direct unit tests — only
  exercised indirectly via `evaluate_alert`/`process_spill`.
- `src/attribution/anomaly_features.py::speed_change_score()` divides by
  `baseline` after only checking `baseline < min_baseline_speed_knots` — if
  that config field is ever set to `0.0` and baseline is exactly `0.0`,
  `ZeroDivisionError`.
- `src/ais/query.py` and `src/ais/filter.py` use unvectorized per-row Python
  loops (`.apply(axis=1)`, a list comprehension over `.distance()`) for
  AIS-point-in-area and CPA-distance — fine at every AIS export size used so
  far, will be the first thing to slow down on a real, busy-shipping-lane
  export.

Full findings (including Optional/Note-only items, and everything confirmed
*not* a problem) are in that review's own report, not duplicated here.

## Working agreement

The user drives this step by step and says **"stop here"** at the end of each
brief. Respect it — do not start the next pipeline stage unasked. Commit only
when asked; push only when asked (they are separate requests).

## Open questions

1. Where are the MKLab labels / is that dataset being downloaded at all?
2. Can acquisition timestamps be recovered for the 1200-tile set? Without them
   AIS attribution cannot work — it needs a real acquisition time to search
   AIS around (`ais.search_window_hours`, now 48h one-sided).
3. Which machine has a GPU for the real training run?
4. No real-case validation for AIS attribution yet. Researched two real,
   documented spill+AIS cases (Maersk Kiera 2012 UK, prosecuted; Dona Liberta
   2012 Angola/Congo, SkyTruth-documented) but neither is usable: both are
   open-ocean cases identified via satellite AIS, and this project's two open
   archives (Danish Maritime Authority, NOAA MarineCadastre) are terrestrial-
   only. Next best lead: search HELCOM's Baltic surveillance reports for a
   case that actually sits inside DMA's coverage. Currently validated only
   against `tests/test_attribution_integration.py`'s synthetic fixture.
5. `use_hindcasting` has never been exercised against a real AIS export —
   synthetic fixtures only, same caveat as #4.
6. Live polling (5.1) has never touched the real CDSE network — no
   credentials in this environment, `poll_once()`/`run_live.py` are
   stub-catalogue-tested only. First real run will be the first time
   `CDSECatalogue.fetch()`'s downloaded-archive naming and
   `LocalSceneSource`'s `.SAFE.zip` reading actually meet in practice for a
   *live* scene (Phase 1 has only ever read pre-downloaded local files).
7. The dashboard (5.4) and live poller (5.1) are not connected — the
   dashboard serves one already-computed `pipeline_result.json`, loaded once
   at startup; there is no live-refreshing view of a running poller yet.
8. No PDF export — step 5.3 chose standalone HTML over PDF for reliability
   (no new rendering dependency). If a literal PDF is ever required,
   `report.py`'s own docstring names where a `weasyprint` call would go.
