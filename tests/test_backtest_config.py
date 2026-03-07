"""Tests for M4 backtest config loader (config.py)."""

from datetime import date
from pathlib import Path

import pytest
import yaml

from trade_analysis.backtester.config import (
    BacktestConfig,
    WalkForwardConfig,
    load_backtest_config,
)
from trade_analysis.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "backtest.yaml"
    p.write_text(yaml.dump(data, default_flow_style=False))
    return p


def _minimal_config() -> dict:
    return {
        "backtest": {
            "date_range": {
                "start": "2020-01-01",
                "end": "2024-12-31",
            },
        },
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadBacktestConfig:
    """Test loading backtest config from YAML."""

    def test_load_real_config(self):
        """Load the actual config/backtest.yaml."""
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        assert isinstance(cfg, BacktestConfig)

    def test_date_range(self):
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        assert cfg.start_date == date(2020, 1, 1)
        assert cfg.end_date == date(2024, 12, 31)

    def test_initial_capital(self):
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        assert cfg.initial_capital == 100000.0

    def test_max_open_positions(self):
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        assert cfg.max_open_positions == 1

    def test_walk_forward_config(self):
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        assert cfg.walk_forward is not None
        assert cfg.walk_forward.in_sample_years == 3
        assert cfg.walk_forward.out_of_sample_years == 1
        assert cfg.walk_forward.anchored is True


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    """Test default values for optional fields."""

    def test_minimal_config_loads(self, tmp_path):
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_backtest_config(p)

        assert cfg.initial_capital == 100000.0
        assert cfg.max_open_positions == 1
        assert cfg.walk_forward is None

    def test_walk_forward_none_when_absent(self, tmp_path):
        p = _write_yaml(tmp_path, _minimal_config())
        cfg = load_backtest_config(p)
        assert cfg.walk_forward is None

    def test_walk_forward_anchored_defaults_true(self, tmp_path):
        data = _minimal_config()
        data["backtest"]["walk_forward"] = {
            "in_sample_years": 2,
            "out_of_sample_years": 1,
        }
        p = _write_yaml(tmp_path, data)
        cfg = load_backtest_config(p)
        assert cfg.walk_forward is not None
        assert cfg.walk_forward.anchored is True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestConfigErrors:
    """Test config validation and error handling."""

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_backtest_config(tmp_path / "nonexistent.yaml")

    def test_missing_backtest_key(self, tmp_path):
        p = _write_yaml(tmp_path, {"other": {}})
        with pytest.raises(ConfigError, match="backtest"):
            load_backtest_config(p)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "backtest.yaml"
        p.write_text("")
        with pytest.raises(ConfigError, match="backtest"):
            load_backtest_config(p)

    def test_missing_date_range(self, tmp_path):
        p = _write_yaml(tmp_path, {"backtest": {}})
        with pytest.raises(ConfigError, match="date_range"):
            load_backtest_config(p)

    def test_missing_start_date(self, tmp_path):
        data = {"backtest": {"date_range": {"end": "2024-12-31"}}}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="start"):
            load_backtest_config(p)

    def test_missing_end_date(self, tmp_path):
        data = {"backtest": {"date_range": {"start": "2020-01-01"}}}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="end"):
            load_backtest_config(p)

    def test_invalid_date_format(self, tmp_path):
        data = {"backtest": {"date_range": {"start": "not-a-date", "end": "2024-12-31"}}}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="Invalid date"):
            load_backtest_config(p)

    def test_start_after_end(self, tmp_path):
        data = {"backtest": {"date_range": {"start": "2025-01-01", "end": "2024-12-31"}}}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="before"):
            load_backtest_config(p)

    def test_walk_forward_missing_is_years(self, tmp_path):
        data = _minimal_config()
        data["backtest"]["walk_forward"] = {"out_of_sample_years": 1}
        p = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="in_sample_years"):
            load_backtest_config(p)


# ---------------------------------------------------------------------------
# Frozen
# ---------------------------------------------------------------------------


class TestConfigFrozen:
    """Verify config immutability."""

    def test_backtest_config_is_frozen(self):
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        with pytest.raises(AttributeError):
            cfg.initial_capital = 999  # type: ignore[misc]

    def test_walk_forward_config_is_frozen(self):
        cfg = load_backtest_config(Path("config/backtest.yaml"))
        assert cfg.walk_forward is not None
        with pytest.raises(AttributeError):
            cfg.walk_forward.in_sample_years = 99  # type: ignore[misc]
