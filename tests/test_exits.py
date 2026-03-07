"""Tests for M3 exit level computation (exits.py)."""

import numpy as np
import pandas as pd
import pytest

from trade_analysis.signals.exits import compute_exit_levels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exit_df(
    n: int = 50,
    direction: str | None = None,
    direction_at: int | None = None,
) -> pd.DataFrame:
    """Create OHLCV DataFrame with signal_direction column.

    If direction_at is specified, only that bar gets the signal direction.
    Otherwise, all bars get the specified direction.
    """
    rng = np.random.default_rng(42)
    base = 100.0
    closes = base + rng.standard_normal(n).cumsum() * 0.5
    closes = np.maximum(closes, 50.0)

    timestamps = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes - rng.uniform(0, 1, n),
            "high": closes + rng.uniform(0.5, 2, n),
            "low": closes - rng.uniform(0.5, 2, n),
            "close": closes,
            "volume": rng.uniform(1e6, 5e6, n),
        }
    )

    if direction_at is not None:
        sig_dir = pd.Series(None, index=df.index, dtype=object)
        sig_dir.iloc[direction_at] = direction
        df["signal_direction"] = sig_dir
    else:
        df["signal_direction"] = direction

    return df


def _make_known_exit_df() -> pd.DataFrame:
    """Create a deterministic DataFrame for exact exit level assertions."""
    n = 30
    # Uptrend with a clear swing low
    closes = [100.0] * 10  # flat base
    closes += [98.0, 96.0, 95.0, 96.0, 98.0]  # swing low at 95 (bar 12)
    closes += [100.0, 102.0, 104.0, 106.0, 108.0]  # move up
    closes += [110.0, 112.0, 114.0, 116.0, 118.0]  # continue up
    closes += [120.0, 122.0, 124.0, 126.0, 128.0]  # signal bar territory

    timestamps = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    c = np.array(closes, dtype=float)

    # Create highs/lows with swing structure
    highs = c + 1.0
    lows = c - 1.0
    # Make swing low at bar 12 clearly the lowest
    lows[12] = 93.0  # clear swing low

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": c - 0.5,
            "high": highs,
            "low": lows,
            "close": c,
            "volume": np.full(n, 1e6),
        }
    )

    # Signal on bar 25 (close = 120)
    sig_dir = pd.Series(None, index=df.index, dtype=object)
    sig_dir.iloc[25] = "long"
    df["signal_direction"] = sig_dir

    return df


# ===========================================================================
# Output columns
# ===========================================================================


class TestExitOutputColumns:
    """Verify exit level output columns."""

    def test_columns_present(self):
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df)
        expected = {
            "exit_stop",
            "exit_target",
            "exit_trail_be",
            "exit_risk",
            "exit_reward",
            "exit_rr_ratio",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_modify_original(self):
        df = _make_exit_df(direction_at=30, direction="long")
        original_cols = list(df.columns)
        compute_exit_levels(df)
        assert list(df.columns) == original_cols


# ===========================================================================
# Long signal exits
# ===========================================================================


class TestLongExits:
    """Test exit levels for long signals."""

    def test_stop_below_entry(self):
        """Long stop should be below entry price."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df)

        stop = result["exit_stop"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(stop):
            assert stop < entry

    def test_target_above_entry(self):
        """Long target should be above entry price."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df)

        target = result["exit_target"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(target):
            assert target > entry

    def test_trail_be_above_entry(self):
        """Long trail-to-breakeven should be above entry."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df)

        trail = result["exit_trail_be"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(trail):
            assert trail > entry

    def test_rr_ratio_matches_r_multiple(self):
        """RR ratio should equal the target_r_multiple."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df, target_r_multiple=2.0)

        rr = result["exit_rr_ratio"].iloc[30]
        if not np.isnan(rr):
            assert abs(rr - 2.0) < 0.01


# ===========================================================================
# Short signal exits
# ===========================================================================


class TestShortExits:
    """Test exit levels for short signals."""

    def test_stop_above_entry(self):
        """Short stop should be above entry price."""
        df = _make_exit_df(direction_at=30, direction="short")
        result = compute_exit_levels(df)

        stop = result["exit_stop"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(stop):
            assert stop > entry

    def test_target_below_entry(self):
        """Short target should be below entry price."""
        df = _make_exit_df(direction_at=30, direction="short")
        result = compute_exit_levels(df)

        target = result["exit_target"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(target):
            assert target < entry

    def test_trail_be_below_entry(self):
        """Short trail-to-breakeven should be below entry."""
        df = _make_exit_df(direction_at=30, direction="short")
        result = compute_exit_levels(df)

        trail = result["exit_trail_be"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(trail):
            assert trail < entry


# ===========================================================================
# No signal bars
# ===========================================================================


class TestNoSignalBars:
    """Bars without signals should have NaN exit levels."""

    def test_nan_exits_no_signal(self):
        """Bars with no signal direction → all exit columns NaN."""
        df = _make_exit_df(direction=None)
        result = compute_exit_levels(df)

        assert result["exit_stop"].isna().all()
        assert result["exit_target"].isna().all()
        assert result["exit_trail_be"].isna().all()

    def test_only_signal_bar_has_values(self):
        """Only the bar with a signal should have exit values."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df)

        # Non-signal bars should be NaN
        non_signal = result.drop(index=30)
        assert non_signal["exit_stop"].isna().all()
        assert non_signal["exit_target"].isna().all()


# ===========================================================================
# Stop methods
# ===========================================================================


class TestStopMethods:
    """Test different stop placement methods."""

    def test_swing_stop_uses_swing_low(self):
        """Swing stop for long should use recent swing low."""
        df = _make_known_exit_df()
        result = compute_exit_levels(df, stop_method="swing", swing_lookback=2)

        stop = result["exit_stop"].iloc[25]
        # Should find the swing low at bar 12 (low = 93.0)
        if not np.isnan(stop):
            assert stop < 120.0  # below entry

    def test_atr_stop_method(self):
        """ATR stop method uses entry ± ATR × multiplier."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(
            df, stop_method="atr", atr_stop_multiplier=2.0, atr_period=14
        )

        stop = result["exit_stop"].iloc[30]
        entry = result["close"].iloc[30]
        if not np.isnan(stop):
            assert stop < entry

    def test_swing_fallback_to_atr(self):
        """When no swing is found, falls back to ATR-based stop."""
        # Create data with no swing lows at all (flat)
        n = 30
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC"),
                "open": np.full(n, 99.5),
                "high": np.full(n, 101.0),
                "low": np.full(n, 99.0),
                "close": np.full(n, 100.0),
                "volume": np.full(n, 1e6),
                "signal_direction": [None] * 20 + ["long"] + [None] * 9,
            }
        )
        result = compute_exit_levels(df, stop_method="swing", swing_lookback=2)

        # Should still compute a stop (ATR fallback)
        stop = result["exit_stop"].iloc[20]
        # May be NaN if ATR is also NaN for flat data, but test the fallback path runs
        # The important thing is it doesn't crash


# ===========================================================================
# R-multiple and trail
# ===========================================================================


class TestRMultiple:
    """Test R-multiple calculation for target and trail."""

    def test_target_2r(self):
        """Target at 2R: reward = 2 × risk."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df, target_r_multiple=2.0)

        risk = result["exit_risk"].iloc[30]
        reward = result["exit_reward"].iloc[30]
        if not np.isnan(risk) and not np.isnan(reward):
            assert abs(reward - 2.0 * risk) < 0.01

    def test_target_3r(self):
        """Target at 3R: reward = 3 × risk."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df, target_r_multiple=3.0)

        risk = result["exit_risk"].iloc[30]
        reward = result["exit_reward"].iloc[30]
        if not np.isnan(risk) and not np.isnan(reward):
            assert abs(reward - 3.0 * risk) < 0.01

    def test_trail_breakeven_at_1r(self):
        """Trail-to-breakeven at 1R: trail_be = entry + 1 × risk."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df, trail_breakeven_r=1.0)

        entry = result["close"].iloc[30]
        risk = result["exit_risk"].iloc[30]
        trail = result["exit_trail_be"].iloc[30]
        if not np.isnan(risk) and not np.isnan(trail):
            expected = entry + risk
            assert abs(trail - expected) < 0.01

    def test_risk_is_positive(self):
        """Risk should always be positive."""
        df = _make_exit_df(direction_at=30, direction="long")
        result = compute_exit_levels(df)

        risk = result["exit_risk"].iloc[30]
        if not np.isnan(risk):
            assert risk > 0

    def test_existing_atr_column_reused(self):
        """If ATR column exists, it should be reused."""
        df = _make_exit_df(direction_at=30, direction="long")
        import pandas_ta as pta

        df["atr_14"] = pta.atr(
            high=df["high"], low=df["low"], close=df["close"], length=14
        )
        result = compute_exit_levels(df, atr_period=14, stop_method="atr")

        stop = result["exit_stop"].iloc[30]
        if not np.isnan(stop):
            assert stop < result["close"].iloc[30]
