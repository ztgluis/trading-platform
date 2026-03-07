"""Tests for M3 condition evaluators (conditions.py)."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.signals.conditions import (
    evaluate_momentum_condition,
    evaluate_structure_condition,
    evaluate_trend_condition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame."""
    n = len(closes)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    c = np.array(closes, dtype=float)
    h = np.array(highs, dtype=float) if highs else c * 1.005
    lo = np.array(lows, dtype=float) if lows else c * 0.995
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": c * 0.999,
            "high": h,
            "low": lo,
            "close": c,
            "volume": np.full(n, 1e6),
        }
    )


def _make_trending_ohlcv(direction: str = "up", n: int = 200) -> pd.DataFrame:
    """Create a clear trending OHLCV dataset."""
    rng = np.random.default_rng(42)
    base = 100.0
    if direction == "up":
        closes = base + np.arange(n) * 0.5 + rng.standard_normal(n) * 0.3
    else:
        closes = base - np.arange(n) * 0.5 + rng.standard_normal(n) * 0.3

    closes = np.maximum(closes, 10.0)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 1, n),
            "high": closes + rng.uniform(0.5, 2, n),
            "low": closes - rng.uniform(0.5, 2, n),
            "close": closes,
            "volume": rng.uniform(1e6, 5e6, n),
        }
    )


# ===========================================================================
# Trend Condition
# ===========================================================================


class TestTrendCondition:
    """Test evaluate_trend_condition."""

    def test_output_columns(self):
        """Adds trend_bull and trend_bear columns."""
        df = _make_ohlcv([100, 101, 102, 103, 104])
        df["sma_3"] = df["close"].rolling(3).mean()
        result = evaluate_trend_condition(df, ma_column="sma_3")

        assert "trend_bull" in result.columns
        assert "trend_bear" in result.columns

    def test_bull_when_above_ma(self):
        """Close above MA → trend_bull is True."""
        closes = [100, 100, 100, 110, 110]
        df = _make_ohlcv(closes)
        df["sma_3"] = df["close"].rolling(3).mean()
        result = evaluate_trend_condition(df, ma_column="sma_3")

        # Last bar: close=110, SMA(3)= (100+110+110)/3 ≈ 106.7 → bull
        assert result["trend_bull"].iloc[-1] == True  # noqa: E712

    def test_bear_when_below_ma(self):
        """Close below MA → trend_bear is True."""
        closes = [100, 100, 100, 90, 90]
        df = _make_ohlcv(closes)
        df["sma_3"] = df["close"].rolling(3).mean()
        result = evaluate_trend_condition(df, ma_column="sma_3")

        # Last bar: close=90, SMA(3)= (100+90+90)/3 ≈ 93.3 → bear
        assert result["trend_bear"].iloc[-1] == True  # noqa: E712

    def test_nan_warmup_is_false(self):
        """During MA warmup (NaN), both trend_bull and trend_bear are False."""
        closes = [100, 101, 102, 103, 104]
        df = _make_ohlcv(closes)
        df["sma_3"] = df["close"].rolling(3).mean()
        result = evaluate_trend_condition(df, ma_column="sma_3")

        # First 2 bars: SMA(3) is NaN → both False
        assert result["trend_bull"].iloc[0] == False  # noqa: E712
        assert result["trend_bear"].iloc[0] == False  # noqa: E712

    def test_mutual_exclusivity(self):
        """trend_bull and trend_bear are mutually exclusive (except at MA)."""
        df = _make_trending_ohlcv("up", n=50)
        df["sma_10"] = df["close"].rolling(10).mean()
        result = evaluate_trend_condition(df, ma_column="sma_10")

        # No bar should have both True
        both_true = result["trend_bull"] & result["trend_bear"]
        assert both_true.sum() == 0

    def test_does_not_modify_original(self):
        """Original DataFrame is not modified."""
        df = _make_ohlcv([100, 101, 102])
        df["sma_3"] = df["close"].rolling(3).mean()
        original_cols = list(df.columns)
        evaluate_trend_condition(df, ma_column="sma_3")
        assert list(df.columns) == original_cols


# ===========================================================================
# Structure Condition
# ===========================================================================


class TestStructureCondition:
    """Test evaluate_structure_condition."""

    def test_output_columns(self, sample_200bar_ohlcv):
        """Adds all structure columns."""
        result = evaluate_structure_condition(sample_200bar_ohlcv)
        expected = {
            "structure_bull",
            "structure_bear",
            "structure_near_pivot",
            "structure_near_round",
            "structure_multi_method",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        """Original DataFrame is not modified."""
        original_cols = list(sample_200bar_ohlcv.columns)
        evaluate_structure_condition(sample_200bar_ohlcv)
        assert list(sample_200bar_ohlcv.columns) == original_cols

    def test_structure_bull_requires_higher_low(self):
        """Bullish structure needs a higher low pattern."""
        # Create data with clear higher lows:
        # Low at bar 3 = 90, low at bar 10 = 95 (higher low)
        n = 20
        closes = [100.0] * n
        highs = [102.0] * n
        lows = [98.0] * n

        # Create swing lows at specific bars (lookback=2)
        # Bar 3: low dip
        lows[2] = 97.0
        lows[3] = 90.0  # swing low
        lows[4] = 97.0

        # Bar 10: higher swing low
        lows[9] = 97.0
        lows[10] = 95.0  # higher swing low
        lows[11] = 97.0

        df = _make_ohlcv(closes, highs, lows)
        result = evaluate_structure_condition(
            df, swing_lookback=2, level_proximity_pct=10.0
        )

        # higher_low should be detected at bar 10
        assert result["higher_low"].iloc[10] == True  # noqa: E712

    def test_structure_bear_requires_lower_high(self):
        """Bearish structure needs a lower high pattern."""
        n = 20
        closes = [100.0] * n
        highs = [102.0] * n
        lows = [98.0] * n

        # Bar 3: swing high
        highs[2] = 103.0
        highs[3] = 110.0  # swing high
        highs[4] = 103.0

        # Bar 10: lower swing high
        highs[9] = 103.0
        highs[10] = 105.0  # lower high
        highs[11] = 103.0

        df = _make_ohlcv(closes, highs, lows)
        result = evaluate_structure_condition(
            df, swing_lookback=2, level_proximity_pct=10.0
        )

        assert result["lower_high"].iloc[10] == True  # noqa: E712

    def test_near_pivot_flag(self, sample_200bar_ohlcv):
        """structure_near_pivot is True when close is near a pivot level."""
        result = evaluate_structure_condition(sample_200bar_ohlcv)
        # With 200 bars, there should be at least some pivots
        assert result["structure_near_pivot"].any()

    def test_near_round_flag(self, sample_200bar_ohlcv):
        """structure_near_round is True when close is near a round number."""
        result = evaluate_structure_condition(sample_200bar_ohlcv)
        # Round numbers should almost always be nearby
        assert result["structure_near_round"].any()

    def test_multi_method_subset_of_both(self, sample_200bar_ohlcv):
        """multi_method implies both near_pivot and near_round."""
        result = evaluate_structure_condition(sample_200bar_ohlcv)
        multi = result[result["structure_multi_method"]]
        if len(multi) > 0:
            assert (multi["structure_near_pivot"] == True).all()  # noqa: E712
            assert (multi["structure_near_round"] == True).all()  # noqa: E712


# ===========================================================================
# Momentum Condition
# ===========================================================================


class TestMomentumCondition:
    """Test evaluate_momentum_condition."""

    def test_output_columns(self, sample_200bar_ohlcv):
        """Adds all momentum columns."""
        result = evaluate_momentum_condition(sample_200bar_ohlcv)
        expected = {
            "momentum_rsi_bull",
            "momentum_rsi_bear",
            "momentum_macd_bull",
            "momentum_macd_bear",
            "momentum_bull",
            "momentum_bear",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        """Original DataFrame is not modified."""
        original_cols = list(sample_200bar_ohlcv.columns)
        evaluate_momentum_condition(sample_200bar_ohlcv)
        assert list(sample_200bar_ohlcv.columns) == original_cols

    def test_rsi_bull_above_threshold(self):
        """RSI > 50 and rising → momentum_rsi_bull."""
        df = _make_trending_ohlcv("up", n=100)
        result = evaluate_momentum_condition(df, rsi_bull_threshold=50)

        # In a strong uptrend, RSI should be above 50 for many bars
        assert result["momentum_rsi_bull"].any()

    def test_rsi_bear_below_threshold(self):
        """RSI < 50 and falling → momentum_rsi_bear."""
        df = _make_trending_ohlcv("down", n=100)
        result = evaluate_momentum_condition(df, rsi_bear_threshold=50)

        assert result["momentum_rsi_bear"].any()

    def test_macd_bull_positive_histogram(self):
        """MACD histogram > 0 → momentum_macd_bull."""
        df = _make_trending_ohlcv("up", n=100)
        result = evaluate_momentum_condition(df)

        assert result["momentum_macd_bull"].any()

    def test_macd_bear_negative_histogram(self):
        """MACD histogram < 0 → momentum_macd_bear."""
        df = _make_trending_ohlcv("down", n=100)
        result = evaluate_momentum_condition(df)

        assert result["momentum_macd_bear"].any()

    def test_overall_bull_is_rsi_or_macd(self, sample_200bar_ohlcv):
        """momentum_bull = momentum_rsi_bull OR momentum_macd_bull."""
        result = evaluate_momentum_condition(sample_200bar_ohlcv)

        expected = result["momentum_rsi_bull"] | result["momentum_macd_bull"]
        pd.testing.assert_series_equal(
            result["momentum_bull"],
            expected,
            check_names=False,
        )

    def test_overall_bear_is_rsi_or_macd(self, sample_200bar_ohlcv):
        """momentum_bear = momentum_rsi_bear OR momentum_macd_bear."""
        result = evaluate_momentum_condition(sample_200bar_ohlcv)

        expected = result["momentum_rsi_bear"] | result["momentum_macd_bear"]
        pd.testing.assert_series_equal(
            result["momentum_bear"],
            expected,
            check_names=False,
        )

    def test_idempotent_with_existing_rsi(self, sample_200bar_ohlcv):
        """If RSI column already exists, uses it instead of recomputing."""
        import pandas_ta as ta

        df = sample_200bar_ohlcv.copy()
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        result = evaluate_momentum_condition(df, rsi_period=14)
        assert "momentum_rsi_bull" in result.columns

    def test_idempotent_with_existing_macd(self, sample_200bar_ohlcv):
        """If MACD columns already exist, uses them instead of recomputing."""
        import pandas_ta as ta

        df = sample_200bar_ohlcv.copy()
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd_df["MACD_12_26_9"]
        df["macd_signal"] = macd_df["MACDs_12_26_9"]
        df["macd_histogram"] = macd_df["MACDh_12_26_9"]

        result = evaluate_momentum_condition(df)
        assert "momentum_macd_bull" in result.columns

    def test_nan_warmup_is_false(self):
        """During RSI/MACD warmup, momentum flags are False."""
        df = _make_ohlcv([100 + i * 0.5 for i in range(50)])
        result = evaluate_momentum_condition(df)

        # First few bars should be False (before RSI/MACD have values)
        assert result["momentum_bull"].iloc[0] == False  # noqa: E712
        assert result["momentum_bear"].iloc[0] == False  # noqa: E712

    def test_custom_thresholds(self):
        """Custom RSI thresholds work correctly."""
        df = _make_trending_ohlcv("up", n=100)

        # With very high bull threshold, fewer bars should be bullish
        result_high = evaluate_momentum_condition(df, rsi_bull_threshold=70)
        result_low = evaluate_momentum_condition(df, rsi_bull_threshold=30)

        bull_count_high = result_high["momentum_rsi_bull"].sum()
        bull_count_low = result_low["momentum_rsi_bull"].sum()

        assert bull_count_low >= bull_count_high
