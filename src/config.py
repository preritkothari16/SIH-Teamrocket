"""Typed application settings for sar-oilspill.

Values are loaded from ``configs/config.yaml`` and may be overridden by
environment variables (prefix ``SAROIL_``, ``__`` between nesting levels)::

    SAROIL_DETECTION__PROB_THRESHOLD=0.65
    SAROIL_AIS__SEARCH_WINDOW_HOURS=12

Secrets are never stored in the YAML file. It only records the *names* of the
environment variables that hold each credential; call
``settings.credentials.resolve()`` to read the actual values from the process
environment (typically populated from a local ``.env``).

Usage::

    from src.config import get_settings

    settings = get_settings()
    settings.detection.prob_threshold
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
ENV_PREFIX = "SAROIL_"
CONFIG_FILE_ENV = ENV_PREFIX + "CONFIG_FILE"


# --------------------------------------------------------------------------- #
# YAML settings source
# --------------------------------------------------------------------------- #
class YamlSettingsSource(PydanticBaseSettingsSource):
    """Feed a YAML file into the pydantic-settings source chain."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self.path = Path(path)

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        # The whole file is loaded at once in __call__; nothing to do per field.
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return {}
        with self.path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError(
                str(self.path) + " must contain a YAML mapping at the top level"
            )
        return data


# --------------------------------------------------------------------------- #
# Config sections
# --------------------------------------------------------------------------- #
class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths are resolved against the repo root."""

    aoi_geojson: Path = Path("configs/aoi.geojson")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    ais_dir: Path = Path("data/ais")
    models_dir: Path = Path("models")

    def resolve(self, path: Path) -> Path:
        """Absolute version of one of these paths."""
        path = Path(path)
        return path if path.is_absolute() else (REPO_ROOT / path)

    @property
    def absolute(self) -> Dict[str, Path]:
        """All configured paths, resolved to absolute."""
        return {name: self.resolve(value) for name, value in self}


class CredentialsConfig(BaseModel):
    """Names of the env vars holding each credential, never the secrets."""

    cdse_client_id_env: str = "CDSE_CLIENT_ID"
    cdse_client_secret_env: str = "CDSE_CLIENT_SECRET"
    cds_api_key_env: str = "CDS_API_KEY"
    cmems_username_env: str = "CMEMS_USERNAME"
    cmems_password_env: str = "CMEMS_PASSWORD"

    def resolve(self, required: bool = False) -> Dict[str, Optional[str]]:
        """Read the actual secrets out of the environment.

        Returns a mapping of logical name -> value, with ``None`` where the
        variable is unset. With ``required=True`` a missing value raises.
        """
        resolved: Dict[str, Optional[str]] = {}
        missing: List[str] = []
        for field_name, env_name in self:
            value = os.environ.get(env_name)
            if value is None:
                missing.append(env_name)
            resolved[field_name.removesuffix("_env")] = value
        if required and missing:
            raise RuntimeError(
                "Missing credential environment variables: "
                + ", ".join(missing)
                + " (copy .env.example to .env and fill it in)"
            )
        return resolved


class IngestionConfig(BaseModel):
    """Satellite product search (step 1.x)."""

    collection: str = "SENTINEL-1"
    product_type: str = "GRD"
    sensor_mode: str = "IW"
    polarisation: str = "VV"
    max_products_per_query: int = Field(default=10, gt=0)
    search_lookback_days: int = Field(default=3, gt=0)


class PreprocessingConfig(BaseModel):
    """SAR preprocessing (step 2.x).

    Tile size and overlap deliberately live on :class:`DetectionConfig`: tiles
    exist to feed the segmentation model, so the model's config owns them.
    """

    target_resolution_m: float = Field(default=40.0, gt=0)
    calibrate_to: str = "sigma0"
    output_crs: str = "EPSG:4326"

    # None means "decide from the pixel values" (dB data is signed and small).
    input_is_db: Optional[bool] = None
    calibration_constant_db: float = 0.0
    number_of_looks: float = Field(default=4.4, gt=0)

    speckle_filter: str = "lee"
    speckle_filter_size: int = Field(default=7, gt=0)

    coastline_path: Path = Path("configs/coastline/ne_10m_land_aoi.geojson")
    coastline_url: str = (
        "https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip"
    )
    land_buffer_m: float = Field(default=500.0, ge=0.0)
    nodata: float = float("nan")
    tile_min_valid_fraction: float = Field(default=0.1, ge=0.0, le=1.0)

    @field_validator("speckle_filter")
    @classmethod
    def _known_filter(cls, v: str) -> str:
        allowed = {"lee", "refined_lee", "none"}
        if v.lower() not in allowed:
            raise ValueError(f"speckle_filter must be one of {sorted(allowed)}")
        return v.lower()

    @field_validator("speckle_filter_size")
    @classmethod
    def _odd_window(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("speckle_filter_size must be odd so the window has a centre")
        return v


class DetectionConfig(BaseModel):
    """Segmentation and look-alike rejection thresholds (step 3.x)."""

    # ``model_`` is a protected attribute prefix in pydantic v2; free it up.
    model_config = {"protected_namespaces": ()}

    model_arch: str = "unet"
    encoder: str = "resnet34"
    tile_size: int = Field(default=512, gt=0)
    tile_overlap: int = Field(default=64, ge=0)
    prob_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_spill_area_km2: float = Field(default=0.05, ge=0.0)
    dark_patch_max_sigma0_db: float = -22.0
    lookalike_min_contrast_db: float = Field(default=3.0, ge=0.0)
    lookalike_min_wind_speed_ms: float = Field(default=3.0, ge=0.0)
    lookalike_max_wind_speed_ms: float = Field(default=12.0, ge=0.0)

    # inference (step 1.4)
    checkpoint_path: Path = Path("models/best.pt")
    batch_size: int = Field(default=8, gt=0)
    input_db_min: float = -35.0
    input_db_max: float = 0.0

    # look-alike rejection (step 1.5)
    min_blob_pixels: int = Field(default=50, gt=0)
    lookalike_min_elongation: float = Field(default=1.8, ge=1.0)
    lookalike_min_edge_gradient: float = Field(default=0.5, ge=0.0)
    lookalike_ring_pixels: int = Field(default=15, gt=0)

    @field_validator("tile_overlap")
    @classmethod
    def _overlap_fits_tile(cls, v: int, info: Any) -> int:
        tile = info.data.get("tile_size")
        if tile is not None and v >= tile:
            raise ValueError("tile_overlap must be smaller than tile_size")
        return v

    @field_validator("input_db_max")
    @classmethod
    def _db_window_ordered(cls, v: float, info: Any) -> float:
        low = info.data.get("input_db_min")
        if low is not None and v <= low:
            raise ValueError("input_db_max must exceed input_db_min")
        return v

    @field_validator("lookalike_max_wind_speed_ms")
    @classmethod
    def _wind_band_ordered(cls, v: float, info: Any) -> float:
        low = info.data.get("lookalike_min_wind_speed_ms")
        if low is not None and v <= low:
            raise ValueError(
                "lookalike_max_wind_speed_ms must exceed lookalike_min_wind_speed_ms"
            )
        return v


class TrainingConfig(BaseModel):
    """Segmentation training (step 3.1).

    ``class_names`` doubles as the label-index order for the MKLab /
    Krestenitis 5-class scheme, so reordering it silently relabels the data.
    """

    dataset_dir: Path = Path("data/raw/mklab")
    num_classes: int = Field(default=5, gt=1)
    class_names: List[str] = Field(
        default_factory=lambda: ["sea", "oil_spill", "look_alike", "ship", "land"]
    )
    in_channels: int = Field(default=3, gt=0)
    encoder_weights: Optional[str] = "imagenet"

    image_size: int = Field(default=512, gt=0)
    batch_size: int = Field(default=8, gt=0)
    epochs: int = Field(default=40, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    val_split: float = Field(default=0.15, gt=0, lt=1)
    num_workers: int = Field(default=0, ge=0)
    seed: int = 42

    loss: str = "dice_ce"
    dice_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    class_weights: Optional[List[float]] = None
    max_class_weight: float = Field(default=50.0, gt=0)

    amp: bool = True
    checkpoint_metric: str = "oil_spill"
    early_stopping_patience: int = Field(default=10, gt=0)

    @field_validator("loss")
    @classmethod
    def _known_loss(cls, v: str) -> str:
        allowed = {"dice_ce", "ce"}
        if v.lower() not in allowed:
            raise ValueError(f"loss must be one of {sorted(allowed)}")
        return v.lower()

    @model_validator(mode="after")
    def _names_match_classes(self) -> "TrainingConfig":
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but "
                f"num_classes is {self.num_classes}"
            )
        if self.checkpoint_metric not in self.class_names:
            raise ValueError(
                f"checkpoint_metric {self.checkpoint_metric!r} is not one of "
                f"class_names {self.class_names}"
            )
        if self.class_weights is not None and len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights must have {self.num_classes} entries, "
                f"got {len(self.class_weights)}"
            )
        return self


class CharacterizationConfig(BaseModel):
    """Slick area / thickness / volume estimation (step 3.x)."""

    thickness_classes: List[str] = Field(
        default_factory=lambda: ["sheen", "rainbow", "thick"]
    )
    volume_estimate_thickness_um: float = Field(default=1.0, gt=0)
    simplify_tolerance_m: float = Field(default=30.0, ge=0.0)
    min_polygon_area_km2: float = Field(default=0.01, ge=0.0)
    spills_dirname: str = "spills"


class AlertsConfig(BaseModel):
    """Alert manager gating and severity banding (step 4.x)."""

    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_area_km2: float = Field(default=0.1, ge=0.0)
    severity_area_km2: Dict[str, float] = Field(
        default_factory=lambda: {
            "low": 0.1,
            "medium": 5.0,
            "high": 20.0,
            "critical": 100.0,
        }
    )
    coastline_proximity_km: float = Field(default=25.0, ge=0.0)

    @field_validator("severity_area_km2")
    @classmethod
    def _bands_increase(cls, v: Dict[str, float]) -> Dict[str, float]:
        bounds = list(v.values())
        if bounds != sorted(bounds):
            raise ValueError(
                "severity_area_km2 bands must be listed in increasing area order"
            )
        return v


class AISConfig(BaseModel):
    """AIS retrieval and vessel attribution (step 2.x).

    ``search_window_hours`` is one-sided: a vessel must have been near the
    spill at or before the SAR acquisition time, not after it, so the window
    is ``[acquisition_time - search_window_hours, acquisition_time]``.
    """

    search_window_hours: float = Field(default=48.0, gt=0)
    search_radius_km: float = Field(default=25.0, gt=0)
    interpolation_minutes: float = Field(default=15.0, gt=0)
    min_moving_speed_knots: float = Field(default=0.5, ge=0.0)
    min_track_points: int = Field(default=3, gt=0)
    max_candidate_vessels: int = Field(default=50, gt=0)


class AttributionConfig(BaseModel):
    """Vessel attribution scoring (step 2.3).

    Combines four independent pieces of evidence into one 0-1 score per
    candidate vessel - see :mod:`src.attribution.scoring` for what each one
    measures and why. ``weight_*`` must sum to 1.0.
    """

    spatial_scale_km: float = Field(default=10.0, gt=0)
    temporal_optimal_lag_hours: float = Field(default=6.0, ge=0)
    temporal_scale_hours: float = Field(default=8.0, gt=0)

    type_priors: Dict[str, float] = Field(
        default_factory=lambda: {
            "tanker": 1.0,
            "cargo": 0.8,
            "passenger": 0.4,
            "fishing": 0.3,
            "other": 0.3,
            "leisure": 0.2,
        }
    )
    default_type_prior: float = Field(default=0.3, ge=0.0, le=1.0)

    weight_spatial: float = Field(default=0.3, ge=0.0)
    weight_temporal: float = Field(default=0.2, ge=0.0)
    weight_alignment: float = Field(default=0.35, ge=0.0)
    weight_type: float = Field(default=0.15, ge=0.0)

    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "AttributionConfig":
        total = (
            self.weight_spatial + self.weight_temporal
            + self.weight_alignment + self.weight_type
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"attribution weight_* fields must sum to 1.0, got {total}")
        return self


class DriftConfig(BaseModel):
    """Lagrangian drift forecast (step 5.x)."""

    forecast_hours: float = Field(default=48.0, gt=0)
    timestep_minutes: float = Field(default=30.0, gt=0)
    wind_drift_factor: float = Field(default=0.03, ge=0.0, le=1.0)
    n_particles: int = Field(default=1000, gt=0)


class APIConfig(BaseModel):
    """FastAPI / dashboard binding."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Root settings object: the single entry point for pipeline config."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
    )

    project_name: str = "sar-oilspill"

    paths: PathsConfig = Field(default_factory=PathsConfig)
    credentials: CredentialsConfig = Field(default_factory=CredentialsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    characterization: CharacterizationConfig = Field(
        default_factory=CharacterizationConfig
    )
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    ais: AISConfig = Field(default_factory=AISConfig)
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        # Precedence, highest first: init args > env vars > .env > config.yaml.
        yaml_path = Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlSettingsSource(settings_cls, yaml_path),
        )


class config_file_override:
    """Context manager pointing the YAML source at a different file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._previous: Optional[str] = None

    def __enter__(self) -> "config_file_override":
        self._previous = os.environ.get(CONFIG_FILE_ENV)
        os.environ[CONFIG_FILE_ENV] = str(self.path)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._previous is None:
            os.environ.pop(CONFIG_FILE_ENV, None)
        else:
            os.environ[CONFIG_FILE_ENV] = self._previous


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Build a fresh (uncached) Settings, optionally from a specific YAML file."""
    if config_path is None:
        return Settings()
    with config_file_override(config_path):
        return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Use ``get_settings.cache_clear()`` to reload."""
    return load_settings()


__all__ = [
    "REPO_ROOT",
    "DEFAULT_CONFIG_PATH",
    "CONFIG_FILE_ENV",
    "Settings",
    "PathsConfig",
    "CredentialsConfig",
    "IngestionConfig",
    "PreprocessingConfig",
    "DetectionConfig",
    "TrainingConfig",
    "CharacterizationConfig",
    "AlertsConfig",
    "AISConfig",
    "AttributionConfig",
    "DriftConfig",
    "APIConfig",
    "config_file_override",
    "load_settings",
    "get_settings",
]
