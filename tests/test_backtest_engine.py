"""Tests for M4 backtester engine (engine.py)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.backtester.engine import Backtester
from trade_analysis.signals.engine import load_signal_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bt_config() -> BacktestConfig:
    return BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=100000.0,
        max_open_positions=1,
        walk_forward=None,
    )


def _make_signal_df(
    n: int = 50,
    tradeable_at: list[int] | None = None,
    direction: str = "long",
    entry_price: float = 100.0,
    stop: float = 95.0,
    target: float = 110.0,
    trail_be: float = 105.0,
    regime: str = "bull",
    score: int = 4,
) -> pd.DataFrame:
    """Create a synthetic signal-enriched DataFrame.

    By default, no bars are tradeable. Use tradeable_at=[bar_indices]
    to make specific bars generate signals.
    """
    timestamps = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(42)

    closes = np.full(n, entry_price)
    highs = closes + rng.uniform(0.5, 2, n)
    lows = closes - rng.uniform(0.5, 2, n)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes - 0.5,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 1e6),
            # Signal columns
            "signal_tradeable": False,
            "signal_direction": None,
            "signal_score": 0,
            "signal_conditions_met": 0,
            "regime": regime,
            "regime_allow_long": True,
            "regime_allow_short": True,
            "regime_strongly_aligned": False,
            # Exit levels
            "exit_stop": np.nan,
            "exit_target": np.nan,
            "exit_trail_be": np.nan,
            "exit_risk": np.nan,
            # Trend MA (needed for bucket routing)
            "ema_21": closes,
            # Condition columns
            "trend_bull": True,
            "trend_bear": False,
            "structure_bull": True,
            "structure_bear": False,
            "structure_multi_method": False,
            "momentum_bull": True,
            "momentum_bear": False,
        }
    )

    if tradeable_at:
        for i in tradeable_at:
            df.loc[df.index[i], "signal_tradeable"] = True
            df.loc[df.index[i], "signal_direction"] = direction
            df.loc[df.index[i], "signal_score"] = score
            df.loc[df.index[i], "signal_conditions_met"] = 2
            df.loc[df.index[i], "exit_stop"] = stop
            df.loc[df.index[i], "exit_target"] = target
            df.loc[df.index[i], "exit_trail_be"] = trail_be
            df.loc[df.index[i], "exit_risk"] = abs(entry_price - stop)

    return df


# ===========================================================================
# Entry logic
# ===========================================================================


class TestEntryLogic:
    """Test backtester entry conditions."""

    def test_enters_on_tradeable_signal(self):
        """Position opened when signal_tradeable=True."""
        df = _make_signal_df(tradeable_at=[10])
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        # Should have 1 trade (entered at bar 10, force-closed at end)
        assert len(result.trades) == 1
        assert result.trades[0].entry_price == 100.0

    def test_no_entry_when_not_tradeable(self):
        """No trades when no bars are tradeable."""
        df = _make_signal_df()  # no tradeable bars
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert len(result.trades) == 0

    def test_no_entry_when_position_open(self):
        """Second tradeable signal while in a trade is skipped."""
        df = _make_signal_df(tradeable_at=[10, 15])
        # Make highs/lows stay between stop and target (no exit)
        df["high"] = 102.0
        df["low"] = 98.0
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        # Only 1 trade (second signal ignored, end-of-data close)
        assert len(result.trades) == 1

    def test_no_entry_when_exit_levels_nan(self):
        """NaN exit levels prevent entry."""
        df = _make_signal_df(n=30)
        df.loc[df.index[10], "signal_tradeable"] = True
        df.loc[df.index[10], "signal_direction"] = "long"
        df.loc[df.index[10], "signal_score"] = 4
        # exit_stop and exit_target remain NaN
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert len(result.trades) == 0

    def test_entry_price_is_close(self):
        """Entry price equals the bar's close."""
        df = _make_signal_df(tradeable_at=[10], entry_price=150.0)
        df["close"] = 150.0
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert result.trades[0].entry_price == 150.0

    def test_entry_captures_signal_metadata(self):
        """Position captures signal score, regime, direction."""
        df = _make_signal_df(
            tradeable_at=[10],
            direction="long",
            score=5,
            regime="bull",
        )
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.direction == "long"
        assert trade.entry_signal_score == 5
        assert trade.entry_regime == "bull"


# ===========================================================================
# Exit logic — Stop loss
# ===========================================================================


class TestStopLoss:
    """Test stop loss exit conditions."""

    def test_stop_hit_long(self):
        """Low touches stop → trade closed at stop price."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=110.0
        )
        # Bar 15: low drops to stop level
        df.loc[df.index[15], "low"] = 94.0
        df.loc[df.index[15], "high"] = 101.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "stop"
        assert trade.exit_price == 95.0
        assert trade.pnl_r == pytest.approx(-1.0)

    def test_stop_hit_short(self):
        """High touches stop → short trade closed at stop price."""
        df = _make_signal_df(
            tradeable_at=[10],
            direction="short",
            entry_price=100.0,
            stop=105.0,
            target=90.0,
            trail_be=95.0,
        )
        # Bar 15: high rises to stop level
        df.loc[df.index[15], "high"] = 106.0
        df.loc[df.index[15], "low"] = 99.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "stop"
        assert trade.exit_price == 105.0
        assert trade.pnl_r == pytest.approx(-1.0)

    def test_pnl_r_negative_one_on_stop(self):
        """Stop loss hit → pnl_r = -1.0 (lost 1R)."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=110.0
        )
        df.loc[df.index[12], "low"] = 94.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert result.trades[0].pnl_r == pytest.approx(-1.0)


# ===========================================================================
# Exit logic — Target
# ===========================================================================


class TestTarget:
    """Test target exit conditions."""

    def test_target_hit_long(self):
        """High reaches target → closed at target price."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=110.0
        )
        # Keep safe from stop
        df["low"] = 98.0
        # Bar 20: high reaches target
        df.loc[df.index[20], "high"] = 112.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "target"
        assert trade.exit_price == 110.0
        assert trade.pnl_r == pytest.approx(2.0)  # (110-100)/5 = 2R

    def test_target_hit_short(self):
        """Low reaches target → short trade closed at target."""
        df = _make_signal_df(
            tradeable_at=[10],
            direction="short",
            entry_price=100.0,
            stop=105.0,
            target=90.0,
            trail_be=95.0,
        )
        df["high"] = 102.0  # safe from stop
        df.loc[df.index[20], "low"] = 89.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "target"
        assert trade.exit_price == 90.0
        assert trade.pnl_r == pytest.approx(2.0)  # (100-90)/5 = 2R

    def test_pnl_r_positive_on_target(self):
        """Target hit → pnl_r = target_r_multiple."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=115.0
        )
        df["low"] = 98.0
        df.loc[df.index[20], "high"] = 116.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert result.trades[0].pnl_r == pytest.approx(3.0)  # (115-100)/5 = 3R


# ===========================================================================
# Exit logic — Same-bar conflict
# ===========================================================================


class TestSameBarConflict:
    """Test priority when stop and target are both hit on same bar."""

    def test_stop_priority_over_target(self):
        """When both stop and target hit on same bar, stop wins (conservative)."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=110.0
        )
        # Bar 15: wide range hits both
        df.loc[df.index[15], "low"] = 93.0
        df.loc[df.index[15], "high"] = 112.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "stop"
        assert trade.exit_price == 95.0


# ===========================================================================
# Exit logic — Trail to breakeven
# ===========================================================================


class TestTrailToBreakeven:
    """Test trail-to-breakeven stop update."""

    def test_trail_updates_stop_long(self):
        """After price reaches trail level, stop moves to entry."""
        df = _make_signal_df(
            tradeable_at=[10],
            entry_price=100.0,
            stop=95.0,
            target=110.0,
            trail_be=105.0,
        )
        df["low"] = 98.0  # safe from original stop

        # Bar 15: high reaches trail level (105)
        df.loc[df.index[15], "high"] = 106.0
        # Bar 20: low drops to entry price (now the stop)
        df.loc[df.index[20], "low"] = 99.5  # just below entry (100)

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "trail_stop"
        assert trade.exit_price == 100.0  # stopped at breakeven
        assert trade.pnl_r == pytest.approx(0.0)

    def test_trail_updates_stop_short(self):
        """Short: after price drops to trail level, stop moves to entry."""
        df = _make_signal_df(
            tradeable_at=[10],
            direction="short",
            entry_price=100.0,
            stop=105.0,
            target=90.0,
            trail_be=95.0,
        )
        df["high"] = 102.0  # safe from original stop

        # Bar 15: low reaches trail level (95)
        df.loc[df.index[15], "low"] = 94.0
        # Bar 20: high reaches entry price (now the stop)
        df.loc[df.index[20], "high"] = 101.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "trail_stop"
        assert trade.exit_price == 100.0

    def test_trail_does_not_fire_before_level(self):
        """Stop stays at original level until trail price reached."""
        df = _make_signal_df(
            n=25,  # short enough to hit end_of_data before max_hold (20 bars)
            tradeable_at=[10],
            entry_price=100.0,
            stop=95.0,
            target=110.0,
            trail_be=105.0,
        )
        df["low"] = 98.0
        df["high"] = 103.0  # never reaches 105

        # Bar 20: low drops below entry (but stop should still be at 95)
        df.loc[df.index[20], "low"] = 99.0  # below entry but above stop

        bt = Backtester(_make_bt_config(), load_signal_config())
        # Use index (Bucket B, max_hold_weeks=None) to avoid max_hold exit
        result = bt.run(df, "^GSPC", "index", "Daily")

        trade = result.trades[0]
        # Should be end_of_data (stop not hit, target not hit, no max_hold)
        assert trade.exit_reason == "end_of_data"


# ===========================================================================
# Exit logic — Max hold
# ===========================================================================


class TestMaxHold:
    """Test max hold period exit."""

    def test_max_hold_exit(self):
        """Position closed at close price after max_hold_bars."""
        df = _make_signal_df(
            n=100,
            tradeable_at=[10],
            entry_price=100.0,
            stop=90.0,  # far from price
            target=120.0,  # far from price
            trail_be=110.0,
        )
        df["low"] = 98.0  # safe from stop
        df["high"] = 102.0  # safe from target/trail

        bt = Backtester(_make_bt_config(), load_signal_config())
        # Bucket A (stock) has max_hold_weeks=4, Daily = 5 bars/week = 20 bars
        result = bt.run(df, "AAPL", "stock", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "max_hold"
        assert trade.duration_bars == 20  # 4 weeks × 5 bars

    def test_max_hold_none_no_forced_exit(self):
        """Bucket B (max_hold_weeks=None) never force-closes."""
        df = _make_signal_df(
            n=200,
            tradeable_at=[10],
            entry_price=100.0,
            stop=90.0,
            target=120.0,
            trail_be=110.0,
        )
        df["low"] = 98.0
        df["high"] = 102.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "^GSPC", "index", "Weekly")

        trade = result.trades[0]
        # Bucket B has no max_hold → should be end_of_data
        assert trade.exit_reason == "end_of_data"


# ===========================================================================
# End of data
# ===========================================================================


class TestEndOfData:
    """Test end-of-data handling."""

    def test_open_position_closed_at_end(self):
        """Position open at last bar is force-closed."""
        df = _make_signal_df(
            tradeable_at=[10],
            entry_price=100.0,
            stop=90.0,
            target=120.0,
        )
        df["low"] = 98.0
        df["high"] = 102.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        # Use index (Bucket B, max_hold_weeks=None) so max_hold doesn't fire first
        result = bt.run(df, "^GSPC", "index", "Daily")

        trade = result.trades[0]
        assert trade.exit_reason == "end_of_data"


# ===========================================================================
# Multiple trades
# ===========================================================================


class TestMultipleTrades:
    """Test multiple sequential trades."""

    def test_second_trade_after_first_closes(self):
        """Can open a new trade after a previous one closes."""
        df = _make_signal_df(
            n=50,
            tradeable_at=[10, 25],
            entry_price=100.0,
            stop=95.0,
            target=110.0,
        )
        df["low"] = 98.0
        df["high"] = 102.0

        # First trade: hit stop at bar 15
        df.loc[df.index[15], "low"] = 94.0
        # Second trade entry at bar 25 (after first closed)
        # Second trade: hit target at bar 35
        df.loc[df.index[35], "high"] = 111.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert len(result.trades) == 2
        assert result.trades[0].exit_reason == "stop"
        assert result.trades[1].exit_reason == "target"


# ===========================================================================
# Duration
# ===========================================================================


class TestDuration:
    """Test trade duration tracking."""

    def test_duration_bars(self):
        """duration_bars is exit_bar_index - entry_bar_index."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=110.0
        )
        df["low"] = 98.0
        df.loc[df.index[20], "high"] = 112.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        assert result.trades[0].duration_bars == 10

    def test_duration_calendar_days(self):
        """duration_calendar_days tracks actual elapsed days."""
        df = _make_signal_df(
            tradeable_at=[10], entry_price=100.0, stop=95.0, target=110.0
        )
        df["low"] = 98.0
        df.loc[df.index[20], "high"] = 112.0

        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")

        # 10 business days ≈ 14 calendar days (2 weekends)
        assert result.trades[0].duration_calendar_days == 14


# ===========================================================================
# BacktestResult metadata
# ===========================================================================


class TestResultMetadata:
    """Test BacktestResult metadata fields."""

    def test_symbol_preserved(self):
        df = _make_signal_df()
        bt = Backtester(_make_bt_config(), load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")
        assert result.symbol == "AAPL"
        assert result.asset_class == "stock"
        assert result.timeframe == "Daily"

    def test_config_preserved(self):
        config = _make_bt_config()
        df = _make_signal_df()
        bt = Backtester(config, load_signal_config())
        result = bt.run(df, "AAPL", "stock", "Daily")
        assert result.config is config
