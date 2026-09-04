# sar-oilspill

SAR satellite oil-spill detection + vessel attribution. Smart India Hackathon.
Python 3.11, Windows. **Read this instead of re-auditing the codebase.**

Pipeline: ingestion → preprocessing → detection → characterization → alerts →
[AIS → attribution] ∥ [drift] → dashboard.

## Status

| Step | Module | State |
|---|---|---|
| 1.x ingestion | `src/ingestion/` | **done** — CDSE catalogue + local disk source |
| 2.x preprocessing | `src/preprocessing/pipeline.py` | **done** — calibrate → despeckle → geocode → mask land → tile |
| 3.1 training | `src/detection/{dataset,model,train}.py` | **done, never trained** — no GPU, no dataset here |
| 3.2 inference | `src/detection/infer.py` | not built — next |
| 3.3+ | look-alike filter, characterization, alerts, AIS, attribution, drift, dashboard | not built |

`pytest` → 147 passing, ~7s, fully offline. Run it before believing anything here.

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
.venv/Scripts/python.exe -m pytest                      # 147 tests, offline
.venv/Scripts/python.exe -m src.detection.train --smoke-test   # end-to-end, no data/GPU
```

```python
from src.ingestion import LocalSceneSource, load_aoi
from src.preprocessing.pipeline import run_pipeline

scene = LocalSceneSource(root="D:/SIH/01_Train_Val_Oil_Spill_images").scenes()[0]
run_pipeline(scene)          # -> data/processed/<scene_id>/ + tile_index.parquet
```

## Working agreement

The user drives this step by step and says **"stop here"** at the end of each
brief. Respect it — do not start the next pipeline stage unasked. Commit only
when asked; push only when asked (they are separate requests).

## Open questions

1. Where are the MKLab labels / is that dataset being downloaded at all?
2. Can acquisition timestamps be recovered for the 1200-tile set? Without them
   AIS attribution cannot work — it needs the SAR acquisition time ±6 h.
3. Which machine has a GPU for the real training run?
