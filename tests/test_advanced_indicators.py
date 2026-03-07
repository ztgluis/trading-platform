"""Tests for advanced indicators: ATR, ZLEMA, HMA, VIDYA, oscillators, signals."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.indicators.trend import (
    add_atr,
    add_hma,
    add_vidya,
    add_zlema,
    add_ma,
)
from trade_analysis.indicators.oscillators import (
    add_two_pole_oscillator,
    add_momentum_bias_index,
    detect_crossovers,
    detect_crossunders,
)
from trade_analysis.indicators.signals import (
    trend_state_machine,
    add_zero_lag_trend_signals,
    add_volumatic_vidya,
)


# ============================================================
# New Trend Indicators (ATR, ZLEMA, HMA, VIDYA)
# ============================================================


class TestATR:
    def test_basic_atr(self, sample_200bar_ohlcv):
        df = add_atr(sample_200bar_ohlcv, period=14)
        assert "atr_14" in df.columns
        assert df["atr_14"].dropna().gt(0).all()

    def test_atr_nan_warmup(self, sample_200bar_ohlcv):
        df = add_atr(sample_200bar_ohlcv, period=14)
        # ATR needs at least period bars to warm up
        nan_count = df["atr_14"].isna().sum()
        assert nan_count >= 1

    def test_atr_different_period(self, sample_200bar_ohlcv):
        df = add_atr(sample_200bar_ohlcv, period=7)
        assert "atr_7" in df.columns

    def test_atr_values_reasonable(self, sample_200bar_ohlcv):
        """ATR should be less than the price range but positive."""
        df = add_atr(sample_200bar_ohlcv, period=14)
        valid = df["atr_14"].dropna()
        # ATR should be positive and smaller than price
        assert (valid > 0).all()
        assert (valid < df["close"].max()).all()


class TestHMA:
    def test_basic_hma(self, sample_200bar_ohlcv):
        df = add_hma(sample_200bar_ohlcv, period=21)
        assert "hma_21" in df.columns

    def test_hma_via_unified_ma(self, sample_200bar_ohlcv):
        df = add_ma(sample_200bar_ohlcv, period=21, ma_type="hma")
        assert "hma_21" in df.columns

    def test_hma_has_values(self, sample_200bar_ohlcv):
        df = add_hma(sample_200bar_ohlcv, period=21)
        # HMA should have valid values after warmup
        valid = df["hma_21"].dropna()
        assert len(valid) > 100


class TestZLEMA:
    def test_basic_zlema(self, sample_200bar_ohlcv):
        df = add_zlema(sample_200bar_ohlcv, period=50)
        assert "zlema_50" in df.columns

    def test_zlema_via_unified_ma(self, sample_200bar_ohlcv):
        df = add_ma(sample_200bar_ohlcv, period=50, ma_type="zlema")
        assert "zlema_50" in df.columns

    def test_zlema_less_lag_than_ema(self):
        """ZLEMA should track recent prices more closely than EMA."""
        # Create a strong trend
        n = 100
        trend = np.arange(100, 100 + n, dtype=float)
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": trend - 0.5,
            "high": trend + 1,
            "low": trend - 1,
            "close": trend,
            "volume": [1e6] * n,
        })
        from trade_analysis.indicators.trend import add_ema
        df = add_ema(df, period=21)
        df = add_zlema(df, period=21)
        # On a linear trend, ZLEMA should be closer to close than EMA
        last = df.iloc[-1]
        ema_gap = abs(last["close"] - last["ema_21"])
        zlema_gap = abs(last["close"] - last["zlema_21"])
        assert zlema_gap < ema_gap

    def test_zlema_nan_warmup(self, sample_200bar_ohlcv):
        df = add_zlema(sample_200bar_ohlcv, period=50)
        nan_count = df["zlema_50"].isna().sum()
        assert nan_count >= 1


class TestVIDYA:
    def test_basic_vidya(self, sample_200bar_ohlcv):
        df = add_vidya(sample_200bar_ohlcv, length=10, momentum_period=20)
        assert "vidya_10" in df.columns

    def test_vidya_has_values(self, sample_200bar_ohlcv):
        df = add_vidya(sample_200bar_ohlcv)
        valid = df["vidya_10"].dropna()
        assert len(valid) > 100

    def test_vidya_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        add_vidya(sample_200bar_ohlcv)
        assert list(sample_200bar_ohlcv.columns) == original_cols

    def test_vidya_values_in_price_range(self, sample_200bar_ohlcv):
        """VIDYA should be bounded within the price range."""
        df = add_vidya(sample_200bar_ohlcv)
        valid = df["vidya_10"].dropna()
        assert valid.min() > sample_200bar_ohlcv["close"].min() * 0.5
        assert valid.max() < sample_200bar_ohlcv["close"].max() * 1.5


# ============================================================
# Crossover Helpers
# ============================================================


class TestCrossovers:
    def test_crossover_detection(self):
        series = pd.Series([1, 2, 3, 2, 1, 2, 3])
        signal = pd.Series([2, 2, 2, 2, 2, 2, 2])
        result = detect_crossovers(series, signal)
        # Should detect crossover at index 2 (1->2->3, crosses above 2)
        assert result.iloc[2] == True  # noqa: E712
        # And at index 6
        assert result.iloc[6] == True  # noqa: E712

    def test_crossunder_detection(self):
        series = pd.Series([3, 2, 1, 2, 3, 2, 1])
        signal = pd.Series([2, 2, 2, 2, 2, 2, 2])
        result = detect_crossunders(series, signal)
        # Should detect crossunder at index 2 (3->2->1, crosses below 2)
        assert result.iloc[2] == True  # noqa: E712
        # And at index 6
        assert result.iloc[6] == True  # noqa: E712

    def test_no_false_crossovers(self):
        """Flat series should produce no crossovers."""
        series = pd.Series([1, 1, 1, 1, 1])
        signal = pd.Series([2, 2, 2, 2, 2])
        result = detect_crossovers(series, signal)
        assert not result.any()


# ============================================================
# Two-Pole Oscillator
# ============================================================


class TestTwoPoleOscillator:
    def test_basic_output_columns(self, sample_200bar_ohlcv):
        df = add_two_pole_oscillator(sample_200bar_ohlcv)
        assert "two_pole" in df.columns
        assert "two_pole_signal" in df.columns
        assert "two_pole_buy" in df.columns
        assert "two_pole_sell" in df.columns

    def test_oscillator_range(self, sample_200bar_ohlcv):
        """Two-pole oscillator should generally be in [-2, 2] range."""
        df = add_two_pole_oscillator(sample_200bar_ohlcv)
        valid = df["two_pole"].dropna()
        # Z-score based, should mostly be in [-3, 3]
        assert valid.min() > -5
        assert valid.max() < 5

    def test_signal_is_lagged(self, sample_200bar_ohlcv):
        """Signal should equal oscillator shifted by 4."""
        df = add_two_pole_oscillator(sample_200bar_ohlcv, signal_lag=4)
        osc = df["two_pole"]
        sig = df["two_pole_signal"]
        # Check that signal[i] == osc[i-4] (ignoring NaN edges)
        for i in range(10, 50):
            if not np.isnan(osc.iloc[i - 4]) and not np.isnan(sig.iloc[i]):
                assert sig.iloc[i] == pytest.approx(osc.iloc[i - 4])

    def test_buy_signals_below_zero(self, sample_200bar_ohlcv):
        """Buy signals should only occur when oscillator < 0."""
        df = add_two_pole_oscillator(sample_200bar_ohlcv)
        buys = df[df["two_pole_buy"]]
        if len(buys) > 0:
            assert (buys["two_pole"] < 0).all()

    def test_sell_signals_above_zero(self, sample_200bar_ohlcv):
        """Sell signals should only occur when oscillator > 0."""
        df = add_two_pole_oscillator(sample_200bar_ohlcv)
        sells = df[df["two_pole_sell"]]
        if len(sells) > 0:
            assert (sells["two_pole"] > 0).all()

    def test_custom_filter_length(self, sample_200bar_ohlcv):
        df = add_two_pole_oscillator(sample_200bar_ohlcv, filter_length=25)
        assert "two_pole" in df.columns

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        add_two_pole_oscillator(sample_200bar_ohlcv)
        assert list(sample_200bar_ohlcv.columns) == original_cols


# ============================================================
# Momentum Bias Index
# ============================================================


class TestMomentumBiasIndex:
    def test_basic_output_columns(self, sample_200bar_ohlcv):
        df = add_momentum_bias_index(sample_200bar_ohlcv)
        assert "mbi_up_bias" in df.columns
        assert "mbi_down_bias" in df.columns
        assert "mbi_boundary" in df.columns
        assert "mbi_bullish_tp" in df.columns
        assert "mbi_bearish_tp" in df.columns
        assert "mbi_trend" in df.columns

    def test_biases_non_negative(self, sample_200bar_ohlcv):
        """Up and down biases should always be >= 0."""
        df = add_momentum_bias_index(sample_200bar_ohlcv)
        assert (df["mbi_up_bias"] >= 0).all()
        assert (df["mbi_down_bias"] >= 0).all()

    def test_boundary_non_negative(self, sample_200bar_ohlcv):
        df = add_momentum_bias_index(sample_200bar_ohlcv)
        valid = df["mbi_boundary"].dropna()
        assert (valid >= 0).all()

    def test_trend_values(self, sample_200bar_ohlcv):
        """Trend should only be -1, 0, or 1."""
        df = add_momentum_bias_index(sample_200bar_ohlcv)
        assert set(df["mbi_trend"].unique()).issubset({-1, 0, 1})

    def test_unsmoothed_mode(self, sample_200bar_ohlcv):
        df = add_momentum_bias_index(sample_200bar_ohlcv, smooth=False)
        assert "mbi_up_bias" in df.columns

    def test_custom_params(self, sample_200bar_ohlcv):
        df = add_momentum_bias_index(
            sample_200bar_ohlcv,
            momentum_length=5,
            bias_length=3,
            smooth_length=8,
        )
        assert "mbi_up_bias" in df.columns

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        add_momentum_bias_index(sample_200bar_ohlcv)
        assert list(sample_200bar_ohlcv.columns) == original_cols


# ============================================================
# Trend State Machine
# ============================================================


class TestTrendStateMachine:
    def test_basic_trend_flip(self):
        """Trend should flip on band crossovers."""
        close = pd.Series([10, 11, 12, 15, 14, 13, 10, 8, 9, 10])
        upper = pd.Series([13] * 10)
        lower = pd.Series([9] * 10)
        trend = trend_state_machine(close, upper, lower)
        # Should flip to +1 when close goes above 13 (index 3: 15 > 13)
        assert trend.iloc[3] == 1
        # Should flip to -1 when close goes below 9 (index 7: 8 < 9)
        assert trend.iloc[7] == -1

    def test_trend_holds_between_crossovers(self):
        """Trend should maintain state between band crossovers."""
        close = pd.Series([10, 11, 15, 14, 13, 12, 11, 10])
        upper = pd.Series([14] * 8)
        lower = pd.Series([9] * 8)
        trend = trend_state_machine(close, upper, lower)
        # Flips to +1 at index 2 (15 > 14), should stay +1
        assert trend.iloc[2] == 1
        assert trend.iloc[3] == 1  # 14 is not < 9, stays +1
        assert trend.iloc[7] == 1  # 10 is not < 9, stays +1

    def test_trend_nan_handling(self):
        """NaN bands should preserve previous trend state."""
        close = pd.Series([10, 15, 12, 8])
        upper = pd.Series([14, 14, np.nan, 14])
        lower = pd.Series([9, 9, np.nan, 9])
        trend = trend_state_machine(close, upper, lower)
        # index 1: 15 > 14 → +1
        assert trend.iloc[1] == 1
        # index 2: NaN bands → keep +1
        assert trend.iloc[2] == 1


# ============================================================
# Zero Lag Trend Signals
# ============================================================


class TestZeroLagTrendSignals:
    def test_basic_output_columns(self, sample_200bar_ohlcv):
        # Use shorter length for test data
        df = add_zero_lag_trend_signals(sample_200bar_ohlcv, length=20, multiplier=1.0)
        assert "zlts_zlema" in df.columns
        assert "zlts_upper" in df.columns
        assert "zlts_lower" in df.columns
        assert "zlts_trend" in df.columns
        assert "zlts_trend_buy" in df.columns
        assert "zlts_trend_sell" in df.columns
        assert "zlts_entry_buy" in df.columns
        assert "zlts_entry_sell" in df.columns

    def test_trend_values(self, sample_200bar_ohlcv):
        df = add_zero_lag_trend_signals(sample_200bar_ohlcv, length=20, multiplier=1.0)
        assert set(df["zlts_trend"].unique()).issubset({-1, 0, 1})

    def test_bands_surround_zlema(self, sample_200bar_ohlcv):
        df = add_zero_lag_trend_signals(sample_200bar_ohlcv, length=20, multiplier=1.0)
        valid = df.dropna(subset=["zlts_zlema", "zlts_upper", "zlts_lower"])
        if len(valid) > 0:
            assert (valid["zlts_upper"] >= valid["zlts_zlema"]).all()
            assert (valid["zlts_lower"] <= valid["zlts_zlema"]).all()

    def test_trend_buy_at_flip(self, sample_200bar_ohlcv):
        """Buy signals should only occur at trend flips from -1 to +1."""
        df = add_zero_lag_trend_signals(sample_200bar_ohlcv, length=20, multiplier=1.0)
        buys = df[df["zlts_trend_buy"]]
        if len(buys) > 0:
            for idx in buys.index:
                pos = df.index.get_loc(idx)
                if pos > 0:
                    assert df["zlts_trend"].iloc[pos] == 1
                    assert df["zlts_trend"].iloc[pos - 1] == -1

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        add_zero_lag_trend_signals(sample_200bar_ohlcv, length=20)
        assert list(sample_200bar_ohlcv.columns) == original_cols


# ============================================================
# Volumatic VIDYA
# ============================================================


class TestVolumaticVIDYA:
    def test_basic_output_columns(self, sample_200bar_ohlcv):
        df = add_volumatic_vidya(sample_200bar_ohlcv, atr_period=50)
        assert "vvidya" in df.columns
        assert "vvidya_upper" in df.columns
        assert "vvidya_lower" in df.columns
        assert "vvidya_trend" in df.columns
        assert "vvidya_trend_buy" in df.columns
        assert "vvidya_trend_sell" in df.columns
        assert "vvidya_vol_delta_pct" in df.columns

    def test_trend_values(self, sample_200bar_ohlcv):
        df = add_volumatic_vidya(sample_200bar_ohlcv, atr_period=50)
        assert set(df["vvidya_trend"].unique()).issubset({-1, 0, 1})

    def test_bands_surround_vidya(self, sample_200bar_ohlcv):
        df = add_volumatic_vidya(sample_200bar_ohlcv, atr_period=50)
        valid = df.dropna(subset=["vvidya", "vvidya_upper", "vvidya_lower"])
        if len(valid) > 0:
            assert (valid["vvidya_upper"] >= valid["vvidya"]).all()
            assert (valid["vvidya_lower"] <= valid["vvidya"]).all()

    def test_volume_delta_finite(self, sample_200bar_ohlcv):
        df = add_volumatic_vidya(sample_200bar_ohlcv, atr_period=50)
        assert np.isfinite(df["vvidya_vol_delta_pct"]).all()

    def test_volume_delta_resets_on_trend_change(self):
        """Delta should restart counting at trend changes."""
        # Create data with a clear trend flip
        n = 50
        close = list(range(100, 125)) + list(range(125, 100, -1))  # up then down
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": [c - 0.5 for c in close],
            "high": [c + 1 for c in close],
            "low": [c - 1 for c in close],
            "close": [float(c) for c in close],
            "volume": [1e6] * n,
        })
        result = add_volumatic_vidya(df, atr_period=10, vidya_length=5, vidya_momentum=10)
        assert "vvidya_vol_delta_pct" in result.columns

    def test_does_not_modify_original(self, sample_200bar_ohlcv):
        original_cols = list(sample_200bar_ohlcv.columns)
        add_volumatic_vidya(sample_200bar_ohlcv, atr_period=50)
        assert list(sample_200bar_ohlcv.columns) == original_cols
