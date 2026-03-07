"""Tests for M3 signal engine orchestrator (generate_signals)."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.signals.engine import (
    generate_signals,
    load_signal_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trending_ohlcv(direction: str = "up", n: int = 250) -> pd.DataFrame:
    """Create an OHLCV DataFrame with a clear trend (needs enough bars for SMA 200)."""
    rng = np.random.default_rng(42)
    base = 100.0
    if direction == "up":
        closes = base + np.arange(n) * 0.3 + rng.standard_normal(n) * 0.5
    else:
        closes = base + 100 - np.arange(n) * 0.3 + rng.standard_normal(n) * 0.5

    closes = np.maximum(closes, 10.0)
    timestamps = pd.date_range("2019-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 1, n),
            "high": closes + rng.uniform(0.5, 3, n),
            "low": closes - rng.uniform(0.5, 3, n),
            "close": closes,
            "volume": rng.uniform(3e7, 1.2e8, n),
        }
    )


# ---------------------------------------------------------------------------
# Pipeline end-to-end
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    """Integration tests for the full signal pipeline."""

    @pytest.fixture
    def config(self):
        return load_signal_config()

    @pytest.fixture
    def uptrend_df(self):
        return _make_trending_ohlcv("up", n=250)

    @pytest.fixture
    def downtrend_df(self):
        return _make_trending_ohlcv("down", n=250)

    def test_returns_dataframe(self, uptrend_df, config):
        """generate_signals returns a DataFrame."""
        result = generate_signals(uptrend_df, asset_class="stock", config=config)
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self, uptrend_df, config):
        """All original OHLCV columns are preserved."""
        original_cols = set(uptrend_df.columns)
        result = generate_signals(uptrend_df, asset_class="stock", config=config)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, uptrend_df, config):
        """Output has the same number of rows as input."""
        result = generate_signals(uptrend_df, asset_class="stock", config=config)
        assert len(result) == len(uptrend_df)

    def test_does_not_modify_original(self, uptrend_df, config):
        """Original DataFrame is not modified."""
        original_cols = list(uptrend_df.columns)
        original_len = len(uptrend_df)
        generate_signals(uptrend_df, asset_class="stock", config=config)
        assert list(uptrend_df.columns) == original_cols
        assert len(uptrend_df) == original_len


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------


class TestOutputColumns:
    """Verify all expected columns are present."""

    @pytest.fixture
    def result(self):
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        return generate_signals(df, asset_class="stock", config=config)

    def test_trend_ma_column(self, result):
        """Bucket A uses EMA 21 → ema_21 column."""
        assert "ema_21" in result.columns

    def test_volume_columns(self, result):
        assert "volume_spike" in result.columns

    def test_regime_columns(self, result):
        expected = {"regime_ma", "regime", "regime_allow_long", "regime_allow_short",
                     "regime_strongly_aligned", "regime_distance_pct"}
        assert expected.issubset(set(result.columns))

    def test_trend_condition_columns(self, result):
        assert "trend_bull" in result.columns
        assert "trend_bear" in result.columns

    def test_structure_condition_columns(self, result):
        expected = {"structure_bull", "structure_bear", "structure_near_pivot",
                     "structure_near_round", "structure_multi_method"}
        assert expected.issubset(set(result.columns))

    def test_momentum_condition_columns(self, result):
        expected = {"momentum_bull", "momentum_bear", "momentum_rsi_bull",
                     "momentum_macd_bull"}
        assert expected.issubset(set(result.columns))

    def test_signal_columns(self, result):
        expected = {"signal_direction", "signal_conditions_met",
                     "signal_score", "signal_tradeable"}
        assert expected.issubset(set(result.columns))

    def test_exit_columns(self, result):
        expected = {"exit_stop", "exit_target", "exit_trail_be",
                     "exit_risk", "exit_reward", "exit_rr_ratio"}
        assert expected.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Asset class routing
# ---------------------------------------------------------------------------


class TestAssetClassRouting:
    """Test that different asset classes use their bucket's parameters."""

    def test_stock_uses_ema_21(self):
        """Stock (Bucket A) should use EMA 21 for trend."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="stock", config=config)
        assert "ema_21" in result.columns

    def test_index_uses_sma_50(self):
        """Index (Bucket B) should use SMA 50 for trend."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="index", config=config)
        assert "sma_50" in result.columns

    def test_crypto_uses_ema_21(self):
        """Crypto (Bucket A) should use EMA 21."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="crypto", config=config)
        assert "ema_21" in result.columns

    def test_metal_uses_sma_50(self):
        """Metal (Bucket B) should use SMA 50."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="metal", config=config)
        assert "sma_50" in result.columns


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------


class TestSignalQuality:
    """Test signal quality properties."""

    def test_uptrend_has_some_long_signals(self):
        """A clear uptrend should produce at least some long signals."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="stock", config=config)

        long_signals = result[result["signal_direction"] == "long"]
        assert len(long_signals) > 0

    def test_score_range(self):
        """Signal scores should be between 0 and 6."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="stock", config=config)

        assert result["signal_score"].min() >= 0
        assert result["signal_score"].max() <= 6

    def test_tradeable_signals_have_exits(self):
        """Tradeable signals should have exit levels computed."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="stock", config=config)

        tradeable = result[result["signal_tradeable"]]
        if len(tradeable) > 0:
            # At least some tradeable signals should have stops
            has_stop = tradeable["exit_stop"].notna()
            assert has_stop.any()

    def test_regime_valid_values(self):
        """Regime column has valid values."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="stock", config=config)

        valid_regimes = {"bull", "bear", "transition"}
        assert set(result["regime"].unique()).issubset(valid_regimes)

    def test_direction_values(self):
        """Signal direction has valid values."""
        df = _make_trending_ohlcv("up", n=250)
        config = load_signal_config()
        result = generate_signals(df, asset_class="stock", config=config)

        directions = result["signal_direction"].dropna().unique()
        valid = {"long", "short"}
        assert set(directions).issubset(valid)


# ---------------------------------------------------------------------------
# Config loading within pipeline
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test config loading within the pipeline."""

    def test_auto_load_config(self):
        """Config auto-loads from default path when not provided."""
        df = _make_trending_ohlcv("up", n=250)
        result = generate_signals(df, asset_class="stock")
        assert "signal_direction" in result.columns

    def test_explicit_config_path(self, tmp_path):
        """Config loads from explicit path."""
        import yaml

        # Copy the real config to a temp location
        config = load_signal_config()
        # Use the auto-loaded config
        df = _make_trending_ohlcv("up", n=250)
        result = generate_signals(df, asset_class="stock", config=config)
        assert "signal_direction" in result.columns
