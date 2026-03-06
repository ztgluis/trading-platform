"""Tests for timeframe aggregation."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.models.ohlcv import Timeframe, validate_ohlcv
from trade_analysis.transforms.timeframe import (
    aggregate_timeframe,
    get_source_timeframe,
    needs_aggregation,
)


class TestAggregateTimeframe:
    def test_1h_to_4h(self, sample_1h_ohlcv):
        result = aggregate_timeframe(sample_1h_ohlcv, Timeframe.H4)
        # Resample aligns to clock boundaries (0,4,8,12,...), so count varies
        assert len(result) >= 10
        assert validate_ohlcv(result) is True

    def test_4h_aggregation_values(self):
        """Test aggregation with clock-aligned timestamps."""
        timestamps = pd.date_range("2024-01-02 00:00", periods=8, freq="h", tz="UTC")
        rng = np.random.default_rng(42)
        closes = 185.0 + rng.standard_normal(8).cumsum() * 0.5
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 0.5, 8),
            "high": closes + rng.uniform(0.2, 1.5, 8),
            "low": closes - rng.uniform(0.2, 1.5, 8),
            "close": closes,
            "volume": rng.uniform(1e6, 5e6, 8),
        })
        result = aggregate_timeframe(df, Timeframe.H4)
        assert len(result) == 2
        # First 4H bar aggregates bars 0-3
        first_4 = df.iloc[:4]
        first_4h = result.iloc[0]
        assert first_4h["open"] == pytest.approx(first_4["open"].iloc[0])
        assert first_4h["high"] == pytest.approx(first_4["high"].max())
        assert first_4h["low"] == pytest.approx(first_4["low"].min())
        assert first_4h["close"] == pytest.approx(first_4["close"].iloc[-1])
        assert first_4h["volume"] == pytest.approx(first_4["volume"].sum())

    def test_daily_to_weekly(self, sample_daily_ohlcv):
        # 10 business days → should produce at least 1 complete week
        result = aggregate_timeframe(sample_daily_ohlcv, Timeframe.WEEKLY)
        assert len(result) >= 1
        assert validate_ohlcv(result) is True

    def test_daily_to_monthly(self):
        # Create 60 days of data (2 months)
        timestamps = pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC")
        rng = np.random.default_rng(42)
        closes = 185.0 + rng.standard_normal(60).cumsum()
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 1, 60),
            "high": closes + rng.uniform(1, 3, 60),
            "low": closes - rng.uniform(1, 3, 60),
            "close": closes,
            "volume": rng.uniform(5e7, 1e8, 60),
        })
        result = aggregate_timeframe(df, Timeframe.MONTHLY)
        assert len(result) >= 2
        assert validate_ohlcv(result) is True

    def test_preserves_attrs(self, sample_1h_ohlcv):
        sample_1h_ohlcv.attrs["symbol"] = "AAPL"
        result = aggregate_timeframe(sample_1h_ohlcv, Timeframe.H4)
        assert result.attrs["symbol"] == "AAPL"
        assert result.attrs["timeframe"] == "4H"

    def test_invalid_target_raises(self, sample_daily_ohlcv):
        with pytest.raises(ValueError, match="No resample rule"):
            aggregate_timeframe(sample_daily_ohlcv, Timeframe.H1)

    def test_custom_week_end_day(self, sample_daily_ohlcv):
        result = aggregate_timeframe(
            sample_daily_ohlcv, Timeframe.WEEKLY, week_end_day="SUN"
        )
        assert len(result) >= 1
        assert validate_ohlcv(result) is True


class TestNeedsAggregation:
    def test_4h_needs_aggregation_from_1h(self):
        supported = [Timeframe.H1, Timeframe.DAILY]
        assert needs_aggregation(Timeframe.H4, supported) is True

    def test_daily_does_not_need_aggregation(self):
        supported = [Timeframe.H1, Timeframe.DAILY, Timeframe.WEEKLY]
        assert needs_aggregation(Timeframe.DAILY, supported) is False

    def test_weekly_from_daily(self):
        supported = [Timeframe.DAILY]
        assert needs_aggregation(Timeframe.WEEKLY, supported) is True


class TestGetSourceTimeframe:
    def test_4h_source_is_1h(self):
        assert get_source_timeframe(Timeframe.H4) == Timeframe.H1

    def test_weekly_source_is_daily(self):
        assert get_source_timeframe(Timeframe.WEEKLY) == Timeframe.DAILY

    def test_monthly_source_is_daily(self):
        assert get_source_timeframe(Timeframe.MONTHLY) == Timeframe.DAILY

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            get_source_timeframe(Timeframe.H1)
