"""Tests for M5 parameter grid generation and config modification."""

import pytest

from trade_analysis.grid.parameters import (
    ALL_KNOWN_PARAMS,
    generate_parameter_grid,
    apply_params_to_config,
)
from trade_analysis.exceptions import ConfigError
from trade_analysis.signals.engine import load_signal_config


# ===========================================================================
# Grid generation
# ===========================================================================


class TestGenerateParameterGrid:
    """Test parameter combination generation."""

    def test_single_param(self):
        """Single parameter produces one combo per value."""
        combos = generate_parameter_grid({"rsi_period": [10, 14, 20]})
        assert len(combos) == 3
        assert combos[0] == {"rsi_period": 10}
        assert combos[1] == {"rsi_period": 14}
        assert combos[2] == {"rsi_period": 20}

    def test_two_params_cartesian_product(self):
        """Two parameters produce cartesian product."""
        combos = generate_parameter_grid({
            "rsi_period": [10, 14],
            "trend_ma_period": [20, 50],
        })
        assert len(combos) == 4  # 2 × 2
        # Check all combos exist
        expected = [
            {"rsi_period": 10, "trend_ma_period": 20},
            {"rsi_period": 10, "trend_ma_period": 50},
            {"rsi_period": 14, "trend_ma_period": 20},
            {"rsi_period": 14, "trend_ma_period": 50},
        ]
        assert combos == expected

    def test_three_params(self):
        """Three parameters produce correct product count."""
        combos = generate_parameter_grid({
            "rsi_period": [10, 14],
            "trend_ma_period": [20, 50],
            "atr_period": [10, 14, 20],
        })
        assert len(combos) == 2 * 2 * 3  # 12

    def test_single_value_param(self):
        """Parameter with single value still generates combos."""
        combos = generate_parameter_grid({
            "rsi_period": [14],
            "trend_ma_period": [20, 50],
        })
        assert len(combos) == 2
        assert all(c["rsi_period"] == 14 for c in combos)

    def test_empty_params(self):
        """Empty parameters dict returns single empty combo."""
        combos = generate_parameter_grid({})
        assert combos == [{}]

    def test_all_combos_are_dicts(self):
        """Each combination is a dict."""
        combos = generate_parameter_grid({"rsi_period": [10, 14]})
        for combo in combos:
            assert isinstance(combo, dict)

    def test_large_grid(self):
        """Larger grid produces correct count."""
        combos = generate_parameter_grid({
            "rsi_period": [10, 12, 14, 16, 18, 20],
            "trend_ma_period": [10, 20, 30, 40, 50],
        })
        assert len(combos) == 6 * 5  # 30


# ===========================================================================
# Config modification — direct params
# ===========================================================================


class TestApplyDirectParams:
    """Test applying direct SignalEngineConfig parameters."""

    @pytest.fixture
    def base_config(self):
        return load_signal_config()

    def test_modify_rsi_period(self, base_config):
        """Modify rsi_period on config."""
        modified = apply_params_to_config(
            base_config, {"rsi_period": 20}, "stock"
        )
        assert modified.rsi_period == 20
        assert modified is not base_config

    def test_modify_regime_ma_period(self, base_config):
        """Modify regime_ma_period on config."""
        modified = apply_params_to_config(
            base_config, {"regime_ma_period": 150}, "stock"
        )
        assert modified.regime_ma_period == 150

    def test_modify_atr_period(self, base_config):
        """Modify atr_period on config."""
        modified = apply_params_to_config(
            base_config, {"atr_period": 21}, "stock"
        )
        assert modified.atr_period == 21

    def test_original_unchanged(self, base_config):
        """Base config should not be modified (frozen)."""
        original_rsi = base_config.rsi_period
        apply_params_to_config(base_config, {"rsi_period": 99}, "stock")
        assert base_config.rsi_period == original_rsi

    def test_multiple_direct_params(self, base_config):
        """Multiple direct params applied at once."""
        modified = apply_params_to_config(
            base_config,
            {"rsi_period": 20, "regime_ma_period": 100, "atr_period": 21},
            "stock",
        )
        assert modified.rsi_period == 20
        assert modified.regime_ma_period == 100
        assert modified.atr_period == 21

    def test_empty_params_returns_same(self, base_config):
        """Empty params returns the same config object."""
        result = apply_params_to_config(base_config, {}, "stock")
        assert result is base_config


# ===========================================================================
# Config modification — bucket params
# ===========================================================================


class TestApplyBucketParams:
    """Test applying bucket-specific parameters."""

    @pytest.fixture
    def base_config(self):
        return load_signal_config()

    def test_modify_trend_ma_period_stock(self, base_config):
        """trend_ma_period for stock should modify bucket A."""
        modified = apply_params_to_config(
            base_config, {"trend_ma_period": 30}, "stock"
        )
        assert modified.bucket_a.trend_ma_period == 30
        # Bucket B should be unchanged
        assert modified.bucket_b.trend_ma_period == base_config.bucket_b.trend_ma_period

    def test_modify_trend_ma_period_index(self, base_config):
        """trend_ma_period for index should modify bucket B."""
        modified = apply_params_to_config(
            base_config, {"trend_ma_period": 30}, "index"
        )
        assert modified.bucket_b.trend_ma_period == 30
        # Bucket A should be unchanged
        assert modified.bucket_a.trend_ma_period == base_config.bucket_a.trend_ma_period

    def test_modify_target_r_multiple(self, base_config):
        """target_r_multiple modifies the correct bucket."""
        modified = apply_params_to_config(
            base_config, {"target_r_multiple": 3.0}, "stock"
        )
        assert modified.bucket_a.target_r_multiple == 3.0

    def test_mixed_direct_and_bucket(self, base_config):
        """Both direct and bucket params applied together."""
        modified = apply_params_to_config(
            base_config,
            {"rsi_period": 20, "trend_ma_period": 30},
            "stock",
        )
        assert modified.rsi_period == 20
        assert modified.bucket_a.trend_ma_period == 30


# ===========================================================================
# Validation
# ===========================================================================


class TestParamValidation:
    """Test parameter name validation."""

    @pytest.fixture
    def base_config(self):
        return load_signal_config()

    def test_unknown_param_raises(self, base_config):
        """Unknown parameter name raises ConfigError."""
        with pytest.raises(ConfigError, match="Unknown"):
            apply_params_to_config(
                base_config, {"nonexistent_param": 42}, "stock"
            )

    def test_known_params_set_is_populated(self):
        """ALL_KNOWN_PARAMS should contain expected entries."""
        assert "rsi_period" in ALL_KNOWN_PARAMS
        assert "trend_ma_period" in ALL_KNOWN_PARAMS
        assert "regime_ma_period" in ALL_KNOWN_PARAMS
        assert "target_r_multiple" in ALL_KNOWN_PARAMS
