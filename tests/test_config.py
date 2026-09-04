"""Config scaffold tests: configs/config.yaml loads, validates, and overrides."""

from __future__ import annotations

import pytest
import yaml

from src.config import (
    CONFIG_FILE_ENV,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    Settings,
    get_settings,
    load_settings,
)


@pytest.fixture
def settings() -> Settings:
    return load_settings()


def test_config_yaml_exists() -> None:
    assert DEFAULT_CONFIG_PATH.is_file(), f"missing {DEFAULT_CONFIG_PATH}"


def test_config_yaml_is_a_mapping() -> None:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict) and data


def test_settings_load_and_validate(settings: Settings) -> None:
    assert isinstance(settings, Settings)
    assert settings.project_name == "sar-oilspill"


def test_yaml_keys_all_map_onto_the_model() -> None:
    """extra='forbid' means an unknown YAML key would raise here."""
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert set(data) <= set(Settings.model_fields)


def test_paths_section(settings: Settings) -> None:
    paths = settings.paths
    assert paths.aoi_geojson.suffix == ".geojson"
    for name in ("data_dir", "raw_dir", "processed_dir", "ais_dir", "models_dir"):
        assert paths.resolve(getattr(paths, name)).is_dir(), f"{name} missing on disk"
    assert paths.absolute["raw_dir"].is_absolute()


def test_credentials_are_env_var_names_not_secrets(settings: Settings) -> None:
    creds = settings.credentials
    assert creds.cdse_client_id_env == "CDSE_CLIENT_ID"
    assert creds.cdse_client_secret_env == "CDSE_CLIENT_SECRET"
    assert creds.cds_api_key_env == "CDS_API_KEY"
    assert creds.cmems_username_env == "CMEMS_USERNAME"
    assert creds.cmems_password_env == "CMEMS_PASSWORD"


def test_credentials_resolve_reads_the_environment(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CDSE_CLIENT_ID", "dummy-id")
    assert settings.credentials.resolve()["cdse_client_id"] == "dummy-id"


def test_credentials_resolve_required_raises_when_unset(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    for env_name in (
        "CDSE_CLIENT_ID",
        "CDSE_CLIENT_SECRET",
        "CDS_API_KEY",
        "CMEMS_USERNAME",
        "CMEMS_PASSWORD",
    ):
        monkeypatch.delenv(env_name, raising=False)
    with pytest.raises(RuntimeError, match="Missing credential"):
        settings.credentials.resolve(required=True)


def test_threshold_sections(settings: Settings) -> None:
    assert 0.0 <= settings.detection.prob_threshold <= 1.0
    assert settings.detection.tile_overlap < settings.detection.tile_size
    assert (
        settings.detection.lookalike_min_wind_speed_ms
        < settings.detection.lookalike_max_wind_speed_ms
    )
    assert 0.0 <= settings.alerts.min_confidence <= 1.0
    assert settings.alerts.severity_area_km2
    assert settings.ais.search_window_hours > 0
    assert settings.drift.forecast_hours > 0


def test_env_var_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAROIL_DETECTION__PROB_THRESHOLD", "0.77")
    monkeypatch.setenv("SAROIL_AIS__SEARCH_WINDOW_HOURS", "12")
    reloaded = load_settings()
    assert reloaded.detection.prob_threshold == pytest.approx(0.77)
    assert reloaded.ais.search_window_hours == pytest.approx(12.0)


def test_out_of_range_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAROIL_DETECTION__PROB_THRESHOLD", "1.5")
    with pytest.raises(Exception):
        load_settings()


def test_unknown_yaml_key_is_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project_name: x\nnot_a_real_section: 1\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_FILE_ENV, str(bad))
    with pytest.raises(Exception):
        load_settings()


def test_alternate_config_file_is_honoured(tmp_path) -> None:
    alt = tmp_path / "alt.yaml"
    alt.write_text("project_name: alt-project\n", encoding="utf-8")
    assert load_settings(alt).project_name == "alt-project"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_repo_root_points_at_the_repo() -> None:
    assert (REPO_ROOT / "requirements.txt").is_file()
    assert (REPO_ROOT / "src" / "config.py").is_file()
