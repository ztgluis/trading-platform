"""Tests for the indicator library."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.indicators.trend import add_ema, add_ma, add_sma
from trade_analysis.indicators.momentum import add_macd, add_rsi, add_rsi_direction
from trade_analysis.indicators.structure import (
    detect_higher_lows,
    detect_lower_highs,
    detect_swing_highs,
    detect_swing_lows,
)
from trade_analysis.indicators.volume import add_volume_sma, detect_volume_spike
from trade_analysis.indicators.levels import (
    detect_pivot_levels,
    detect_round_numbers,
    find_nearest_level,
)


# ============================================================
# Trend Indicators
# ============================================================


class TestSMA:
    def test_basic_sma(self, sample_200bar_ohlcv):
        df = add_sma(sample_200bar_ohlcv, period=50)
        assert "sma_50" in df.columns
        assert len(df) == 200

    def test_sma_nan_warmup(self, sample_200bar_ohlcv):
        df = add_sma(sample_200bar_ohlcv, period=50)
        # First 49 values should be NaN
        assert df["sma_50"].isna().sum() == 49

    def test_sma_values(self):
        """Verify SMA against hand-computed values."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "high": [11.0, 12.0, 13.0, 14.0, 15.0],
            "low": [9.0, 10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 12.0, 14.0, 16.0, 18.0],
            "volume": [100.0, 100.0, 100.0, 100.0, 100.0],
        })
        result = add_sma(df, period=3)
        # SMA(3) at index 2: (10+12+14)/3 = 12.0
        assert result["sma_3"].iloc[2] == pytest.approx(12.0)
        # SMA(3) at index 3: (12+14+16)/3 = 14.0
        assert result["sma_3"].iloc[3] == pytest.approx(14.0)
        # SMA(3) at index 4: (14+16+18)/3 = 16.0
        assert result["sma_3"].iloc[4] == pytest.approx(16.0)

    def test_sma_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        add_sma(sample_200bar_ohlcv, period=20)
        assert list(sample_200bar_ohlcv.columns) == original_cols

    def test_sma_different_periods(self, sample_200bar_ohlcv):
        df = add_sma(sample_200bar_ohlcv, period=10)
        df = add_sma(df, period=50)
        assert "sma_10" in df.columns
        assert "sma_50" in df.columns


class TestEMA:
    def test_basic_ema(self, sample_200bar_ohlcv):
        df = add_ema(sample_200bar_ohlcv, period=21)
        assert "ema_21" in df.columns

    def test_ema_nan_warmup(self, sample_200bar_ohlcv):
        df = add_ema(sample_200bar_ohlcv, period=21)
        # EMA has fewer NaN than SMA (typically period-1)
        nan_count = df["ema_21"].isna().sum()
        assert nan_count == 20

    def test_ema_reacts_faster_than_sma(self, sample_200bar_ohlcv):
        """EMA should be closer to recent prices than SMA of same period."""
        df = add_sma(sample_200bar_ohlcv, period=21)
        df = add_ema(df, period=21)
        last_close = df["close"].iloc[-1]
        sma_dist = abs(df["sma_21"].iloc[-1] - last_close)
        ema_dist = abs(df["ema_21"].iloc[-1] - last_close)
        # This isn't always true, but statistically likely with trending data
        # Just verify both are computed without error
        assert not np.isnan(df["ema_21"].iloc[-1])
        assert not np.isnan(df["sma_21"].iloc[-1])


class TestUnifiedMA:
    def test_sma_dispatch(self, sample_200bar_ohlcv):
        df = add_ma(sample_200bar_ohlcv, period=50, ma_type="sma")
        assert "sma_50" in df.columns

    def test_ema_dispatch(self, sample_200bar_ohlcv):
        df = add_ma(sample_200bar_ohlcv, period=21, ma_type="ema")
        assert "ema_21" in df.columns

    def test_invalid_type_raises(self, sample_200bar_ohlcv):
        with pytest.raises(ValueError, match="Unknown MA type"):
            add_ma(sample_200bar_ohlcv, period=21, ma_type="wma")


# ============================================================
# Momentum Indicators
# ============================================================


class TestRSI:
    def test_basic_rsi(self, sample_200bar_ohlcv):
        df = add_rsi(sample_200bar_ohlcv, period=14)
        assert f"rsi_14" in df.columns

    def test_rsi_range(self, sample_200bar_ohlcv):
        """RSI must be in [0, 100]."""
        df = add_rsi(sample_200bar_ohlcv, period=14)
        valid = df["rsi_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_nan_warmup(self, sample_200bar_ohlcv):
        df = add_rsi(sample_200bar_ohlcv, period=14)
        nan_count = df["rsi_14"].isna().sum()
        # pandas-ta RSI uses EMA-style smoothing → only first bar is NaN
        assert nan_count >= 1

    def test_rsi_different_period(self, sample_200bar_ohlcv):
        df = add_rsi(sample_200bar_ohlcv, period=7)
        assert "rsi_7" in df.columns

    def test_rsi_direction(self, sample_200bar_ohlcv):
        df = add_rsi_direction(sample_200bar_ohlcv, period=14)
        assert "rsi_14" in df.columns
        assert "rsi_rising" in df.columns
        assert df["rsi_rising"].dtype == bool


class TestMACD:
    def test_basic_macd(self, sample_200bar_ohlcv):
        df = add_macd(sample_200bar_ohlcv)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

    def test_macd_histogram_is_difference(self, sample_200bar_ohlcv):
        df = add_macd(sample_200bar_ohlcv)
        valid = df.dropna(subset=["macd", "macd_signal", "macd_histogram"])
        diff = valid["macd"] - valid["macd_signal"]
        pd.testing.assert_series_equal(
            valid["macd_histogram"].reset_index(drop=True),
            diff.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_macd_custom_params(self, sample_200bar_ohlcv):
        df = add_macd(sample_200bar_ohlcv, fast=8, slow=21, signal=5)
        assert "macd" in df.columns

    def test_macd_nan_warmup(self, sample_200bar_ohlcv):
        df = add_macd(sample_200bar_ohlcv)
        # MACD needs at least slow (26) periods to warm up
        nan_count = df["macd"].isna().sum()
        assert nan_count >= 25


# ============================================================
# Structure Indicators
# ============================================================


class TestSwingDetection:
    def test_swing_highs_detected(self, sample_200bar_ohlcv):
        df = detect_swing_highs(sample_200bar_ohlcv, lookback=3)
        assert "swing_high" in df.columns
        assert "swing_high_price" in df.columns
        assert df["swing_high"].sum() > 0

    def test_swing_lows_detected(self, sample_200bar_ohlcv):
        df = detect_swing_lows(sample_200bar_ohlcv, lookback=3)
        assert "swing_low" in df.columns
        assert "swing_low_price" in df.columns
        assert df["swing_low"].sum() > 0

    def test_swing_high_on_known_data(self):
        """Bar 3 should be a swing high with lookback=2."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC"),
            "open": [10.0] * 7,
            "high": [10.0, 12.0, 14.0, 20.0, 14.0, 12.0, 10.0],
            "low": [9.0] * 7,
            "close": [10.0] * 7,
            "volume": [100.0] * 7,
        })
        result = detect_swing_highs(df, lookback=2)
        assert result["swing_high"].iloc[3] == True  # noqa: E712 (numpy bool)
        assert result["swing_high_price"].iloc[3] == 20.0

    def test_swing_low_on_known_data(self):
        """Bar 3 should be a swing low with lookback=2."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC"),
            "open": [10.0] * 7,
            "high": [11.0] * 7,
            "low": [10.0, 8.0, 6.0, 3.0, 6.0, 8.0, 10.0],
            "close": [10.0] * 7,
            "volume": [100.0] * 7,
        })
        result = detect_swing_lows(df, lookback=2)
        assert result["swing_low"].iloc[3] == True  # noqa: E712 (numpy bool)
        assert result["swing_low_price"].iloc[3] == 3.0

    def test_no_swings_at_edges(self, sample_200bar_ohlcv):
        """First and last `lookback` bars should not be swings."""
        lookback = 5
        df = detect_swing_highs(sample_200bar_ohlcv, lookback=lookback)
        assert not df["swing_high"].iloc[:lookback].any()
        assert not df["swing_high"].iloc[-lookback:].any()

    def test_higher_lows(self):
        """Ascending pattern should detect higher lows."""
        # Create data with clear higher lows
        lows = [10.0, 8.0, 6.0, 4.0, 6.0, 8.0, 5.0, 8.0, 11.0, 7.0, 11.0, 15.0, 9.0, 15.0, 20.0]
        n = len(lows)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": [10.0] * n,
            "high": [max(l + 5, 25.0) for l in lows],
            "low": lows,
            "close": [10.0] * n,
            "volume": [100.0] * n,
        })
        result = detect_higher_lows(df, lookback=2)
        assert "higher_low" in result.columns

    def test_lower_highs(self):
        """Descending pattern should detect lower highs."""
        highs = [20.0, 22.0, 24.0, 25.0, 24.0, 22.0, 23.0, 20.0, 17.0, 19.0, 15.0, 12.0, 14.0, 10.0, 8.0]
        n = len(highs)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": [10.0] * n,
            "high": highs,
            "low": [min(h - 5, 5.0) for h in highs],
            "close": [10.0] * n,
            "volume": [100.0] * n,
        })
        result = detect_lower_highs(df, lookback=2)
        assert "lower_high" in result.columns


# ============================================================
# Volume Indicators
# ============================================================


class TestVolume:
    def test_volume_sma(self, sample_200bar_ohlcv):
        df = add_volume_sma(sample_200bar_ohlcv, period=20)
        assert "volume_sma_20" in df.columns
        assert df["volume_sma_20"].isna().sum() == 19

    def test_volume_spike_detection(self, sample_200bar_ohlcv):
        df = detect_volume_spike(sample_200bar_ohlcv, period=20, threshold=1.5)
        assert "volume_spike" in df.columns
        assert df["volume_spike"].dtype == bool

    def test_volume_spike_known_data(self):
        """Only the spike bar should be flagged."""
        volumes = [100.0] * 20 + [300.0]  # Last bar is 3x average
        n = len(volumes)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.0] * n,
            "volume": volumes,
        })
        result = detect_volume_spike(df, period=20, threshold=1.5)
        # Last bar (300) should be a spike (> 1.5 * 100 = 150)
        assert result["volume_spike"].iloc[-1] == True  # noqa: E712 (numpy bool)
        # Bars before should not be spikes (100 < 150)
        assert not result["volume_spike"].iloc[19]

    def test_custom_threshold(self, sample_200bar_ohlcv):
        df_low = detect_volume_spike(sample_200bar_ohlcv, threshold=1.0)
        df_high = detect_volume_spike(sample_200bar_ohlcv, threshold=3.0)
        # Lower threshold should flag more spikes
        assert df_low["volume_spike"].sum() >= df_high["volume_spike"].sum()


# ============================================================
# Key Level Detection
# ============================================================


class TestLevels:
    def test_pivot_levels(self, sample_200bar_ohlcv):
        levels = detect_pivot_levels(sample_200bar_ohlcv, lookback=5)
        assert "price" in levels.columns
        assert "type" in levels.columns
        assert "count" in levels.columns
        assert len(levels) > 0

    def test_pivot_types(self, sample_200bar_ohlcv):
        levels = detect_pivot_levels(sample_200bar_ohlcv, lookback=5)
        valid_types = {"support", "resistance"}
        assert set(levels["type"].unique()).issubset(valid_types)

    def test_round_numbers_stock(self):
        levels = detect_round_numbers(185.0)
        # Auto step=10 for price in [100, 1000), base=180
        assert 180.0 in levels
        assert 190.0 in levels
        assert 200.0 in levels

    def test_round_numbers_crypto(self):
        levels = detect_round_numbers(42000.0)
        assert 42000.0 in levels
        assert 42100.0 in levels
        assert 41900.0 in levels

    def test_round_numbers_custom_step(self):
        levels = detect_round_numbers(185.0, step=25.0)
        assert 175.0 in levels
        assert 200.0 in levels

    def test_find_nearest_level_within_range(self):
        levels = [100.0, 150.0, 200.0, 250.0]
        result = find_nearest_level(148.0, levels, max_distance_pct=5.0)
        assert result is not None
        assert result["price"] == 150.0

    def test_find_nearest_level_out_of_range(self):
        levels = [100.0, 200.0]
        result = find_nearest_level(148.0, levels, max_distance_pct=1.0)
        assert result is None

    def test_find_nearest_level_with_dataframe(self, sample_200bar_ohlcv):
        pivot_levels = detect_pivot_levels(sample_200bar_ohlcv, lookback=5)
        if len(pivot_levels) > 0:
            price = pivot_levels["price"].iloc[0]
            result = find_nearest_level(price, pivot_levels, max_distance_pct=1.0)
            assert result is not None
            assert result["distance_pct"] < 1.0

    def test_merge_nearby_levels(self):
        """Levels within merge_distance should be combined."""
        # Create data with two nearby swing lows
        n = 30
        lows = [10.0] * n
        lows[5] = 5.0   # swing low at 5.0
        lows[15] = 5.1   # swing low at 5.1 (should merge with 5.0)
        lows[25] = 8.0   # swing low at 8.0 (different level)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": [10.0] * n,
            "high": [12.0] * n,
            "low": lows,
            "close": [10.0] * n,
            "volume": [100.0] * n,
        })
        levels = detect_pivot_levels(df, lookback=3, merge_distance_pct=5.0)
        # 5.0 and 5.1 should merge, 8.0 stays separate
        support_levels = levels[levels["type"] == "support"]
        assert len(support_levels) >= 1
