"""Tests for M5 grid configuration loading."""

import pytest
import yaml

from trade_analysis.grid.config import GridConfig, load_grid_config
from trade_analysis.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_grid_yaml(tmp_path, content: dict) -> "Path":
    """Write a grid config YAML and return its path."""
    path = tmp_path / "grid.yaml"
    with open(path, "w") as f:
        yaml.dump(content, f)
    return path


def _minimal_grid_yaml() -> dict:
    """Minimal valid grid config."""
    return {
        "grid": {
            "target": {
                "symbol": "AAPL",
                "asset_class": "stock",
                "timeframe": "Daily",
            },
            "parameters": {
                "rsi_period": [10, 14, 20],
            },
        }
    }


# ===========================================================================
# Load from real config
# ===========================================================================


class TestLoadRealConfig:
    """Test loading the real config/grid.yaml."""

    def test_loads_default_config(self):
        """config/grid.yaml should load without error."""
        config = load_grid_config()
        assert isinstance(config, GridConfig)

    def test_default_config_has_parameters(self):
        """Default config should have parameter ranges."""
        config = load_grid_config()
        assert len(config.parameters) > 0

    def test_default_config_target(self):
        """Default config should have AAPL target."""
        config = load_grid_config()
        assert config.symbol == "AAPL"
        assert config.asset_class == "stock"
        assert config.timeframe == "Daily"


# ===========================================================================
# Load from custom YAML
# ===========================================================================


class TestLoadCustomConfig:
    """Test loading custom grid configs."""

    def test_loads_minimal_config(self, tmp_path):
        """Minimal config with one parameter loads correctly."""
        path = _write_grid_yaml(tmp_path, _minimal_grid_yaml())
        config = load_grid_config(path)
        assert config.symbol == "AAPL"
        assert config.parameters["rsi_period"] == [10, 14, 20]

    def test_defaults_applied(self, tmp_path):
        """Default values for optional fields."""
        path = _write_grid_yaml(tmp_path, _minimal_grid_yaml())
        config = load_grid_config(path)
        assert config.min_trades == 30
        assert config.rank_by == "total_r"
        assert config.timeframe == "Daily"

    def test_custom_rank_by(self, tmp_path):
        """Custom rank_by is respected."""
        data = _minimal_grid_yaml()
        data["grid"]["rank_by"] = "profit_factor"
        path = _write_grid_yaml(tmp_path, data)
        config = load_grid_config(path)
        assert config.rank_by == "profit_factor"

    def test_custom_min_trades(self, tmp_path):
        """Custom min_trades is respected."""
        data = _minimal_grid_yaml()
        data["grid"]["min_trades"] = 50
        path = _write_grid_yaml(tmp_path, data)
        config = load_grid_config(path)
        assert config.min_trades == 50

    def test_multiple_parameters(self, tmp_path):
        """Config with multiple parameters."""
        data = _minimal_grid_yaml()
        data["grid"]["parameters"]["trend_ma_period"] = [10, 20, 30]
        path = _write_grid_yaml(tmp_path, data)
        config = load_grid_config(path)
        assert len(config.parameters) == 2

    def test_frozen(self, tmp_path):
        """GridConfig should be immutable."""
        path = _write_grid_yaml(tmp_path, _minimal_grid_yaml())
        config = load_grid_config(path)
        with pytest.raises(AttributeError):
            config.symbol = "MSFT"


# ===========================================================================
# Validation errors
# ===========================================================================


class TestConfigValidation:
    """Test config validation and error handling."""

    def test_missing_file_raises(self, tmp_path):
        """Missing YAML file raises ConfigError."""
        with pytest.raises(ConfigError, match="not found"):
            load_grid_config(tmp_path / "nonexistent.yaml")

    def test_missing_grid_key_raises(self, tmp_path):
        """YAML without 'grid' key raises."""
        path = _write_grid_yaml(tmp_path, {"other": {}})
        with pytest.raises(ConfigError, match="grid"):
            load_grid_config(path)

    def test_missing_target_raises(self, tmp_path):
        """Missing target section raises."""
        path = _write_grid_yaml(tmp_path, {
            "grid": {"parameters": {"rsi_period": [10]}}
        })
        with pytest.raises(ConfigError, match="target"):
            load_grid_config(path)

    def test_missing_symbol_raises(self, tmp_path):
        """Missing symbol raises."""
        path = _write_grid_yaml(tmp_path, {
            "grid": {
                "target": {"asset_class": "stock"},
                "parameters": {"rsi_period": [10]},
            }
        })
        with pytest.raises(ConfigError, match="symbol"):
            load_grid_config(path)

    def test_missing_asset_class_raises(self, tmp_path):
        """Missing asset_class raises."""
        path = _write_grid_yaml(tmp_path, {
            "grid": {
                "target": {"symbol": "AAPL"},
                "parameters": {"rsi_period": [10]},
            }
        })
        with pytest.raises(ConfigError, match="asset_class"):
            load_grid_config(path)

    def test_no_parameters_raises(self, tmp_path):
        """Empty parameters raises."""
        data = _minimal_grid_yaml()
        data["grid"]["parameters"] = {}
        path = _write_grid_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="at least one parameter"):
            load_grid_config(path)

    def test_parameter_not_list_raises(self, tmp_path):
        """Parameter value that's not a list raises."""
        data = _minimal_grid_yaml()
        data["grid"]["parameters"]["rsi_period"] = 14  # scalar, not list
        path = _write_grid_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="list"):
            load_grid_config(path)

    def test_empty_parameter_list_raises(self, tmp_path):
        """Empty parameter value list raises."""
        data = _minimal_grid_yaml()
        data["grid"]["parameters"]["rsi_period"] = []
        path = _write_grid_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="at least one value"):
            load_grid_config(path)

    def test_invalid_rank_by_raises(self, tmp_path):
        """Invalid rank_by value raises."""
        data = _minimal_grid_yaml()
        data["grid"]["rank_by"] = "sharpe_ratio"
        path = _write_grid_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="rank_by"):
            load_grid_config(path)
