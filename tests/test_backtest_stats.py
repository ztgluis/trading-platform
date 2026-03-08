"""Tests for M4 backtest summary statistics (stats.py)."""

from datetime import date

import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.backtester.models import BacktestResult, Trade
from trade_analysis.backtester.stats import (
    compute_backtest_stats,
    format_stats_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    pnl_r: float = 1.5,
    direction: str = "long",
    exit_reason: str = "target",
    entry_score: int = 4,
    entry_regime: str = "bull",
    duration_bars: int = 10,
    duration_days: int = 14,
) -> Trade:
    return Trade(
        symbol="AAPL",
        asset_class="stock",
        timeframe="Daily",
        bucket="A",
        direction=direction,
        entry_timestamp=pd.Timestamp("2021-06-01", tz="UTC"),
        entry_price=150.0,
        entry_signal_score=entry_score,
        entry_regime=entry_regime,
        exit_timestamp=pd.Timestamp("2021-06-15", tz="UTC"),
        exit_price=157.5,
        exit_reason=exit_reason,
        pnl_r=pnl_r,
        pnl_dollar=pnl_r * 5.0,
        duration_bars=duration_bars,
        duration_calendar_days=duration_days,
    )


def _make_result(trades: list[Trade]) -> BacktestResult:
    return BacktestResult(
        trades=trades,
        config=BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        ),
        signal_config_hash="abc123",
        symbol="AAPL",
        asset_class="stock",
        timeframe="Daily",
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
    )


# ===========================================================================
# Empty trades
# ===========================================================================


class TestEmptyTrades:
    """Stats computation on zero trades."""

    def test_empty_trades(self):
        result = _make_result([])
        stats = compute_backtest_stats(result)

        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_r"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["max_drawdown_r"] == 0.0


# ===========================================================================
# All winners / all losers
# ===========================================================================


class TestAllWinnersLosers:
    """Stats for uniform trade outcomes."""

    def test_all_winners(self):
        trades = [_make_trade(pnl_r=2.0) for _ in range(5)]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["win_rate"] == 1.0
        assert stats["profit_factor"] == float("inf")
        assert stats["max_drawdown_r"] == 0.0
        assert stats["max_consecutive_wins"] == 5
        assert stats["max_consecutive_losses"] == 0

    def test_all_losers(self):
        trades = [_make_trade(pnl_r=-1.0) for _ in range(5)]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["win_rate"] == 0.0
        assert stats["profit_factor"] == 0.0
        assert stats["max_consecutive_wins"] == 0
        assert stats["max_consecutive_losses"] == 5


# ===========================================================================
# Known sequence
# ===========================================================================


class TestKnownSequence:
    """Stats on a known sequence of trades."""

    def test_mixed_trades(self):
        """3 wins at +2R, 2 losses at -1R."""
        trades = (
            [_make_trade(pnl_r=2.0, exit_reason="target") for _ in range(3)]
            + [_make_trade(pnl_r=-1.0, exit_reason="stop") for _ in range(2)]
        )
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["total_trades"] == 5
        assert stats["win_rate"] == pytest.approx(0.6)
        assert stats["avg_r"] == pytest.approx(0.8)  # (6-2)/5
        assert stats["total_r"] == pytest.approx(4.0)

    def test_profit_factor(self):
        """3 wins at +2R, 2 losses at -1R → PF = 6/2 = 3.0."""
        trades = (
            [_make_trade(pnl_r=2.0) for _ in range(3)]
            + [_make_trade(pnl_r=-1.0) for _ in range(2)]
        )
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["profit_factor"] == pytest.approx(3.0)


# ===========================================================================
# Drawdown
# ===========================================================================


class TestMaxDrawdown:
    """Test max drawdown calculation."""

    def test_max_drawdown_known_sequence(self):
        """Sequence [+2, -1, -1, +3, -2] → max_dd = 2.0 (after two losses)."""
        trades = [
            _make_trade(pnl_r=2.0),
            _make_trade(pnl_r=-1.0),
            _make_trade(pnl_r=-1.0),
            _make_trade(pnl_r=3.0),
            _make_trade(pnl_r=-2.0),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["max_drawdown_r"] == pytest.approx(2.0)

    def test_max_drawdown_monotonic_up(self):
        """Monotonically increasing equity → max_dd = 0."""
        trades = [_make_trade(pnl_r=1.0) for _ in range(5)]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["max_drawdown_r"] == 0.0


# ===========================================================================
# Consecutive wins/losses
# ===========================================================================


class TestConsecutive:
    """Test consecutive wins and losses."""

    def test_consecutive_wins(self):
        """W W W L W W → max_consec_wins = 3."""
        trades = [
            _make_trade(pnl_r=2.0),
            _make_trade(pnl_r=1.0),
            _make_trade(pnl_r=0.5),
            _make_trade(pnl_r=-1.0),
            _make_trade(pnl_r=1.0),
            _make_trade(pnl_r=1.0),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["max_consecutive_wins"] == 3

    def test_consecutive_losses(self):
        """W L L L W → max_consec_losses = 3."""
        trades = [
            _make_trade(pnl_r=1.0),
            _make_trade(pnl_r=-1.0),
            _make_trade(pnl_r=-0.5),
            _make_trade(pnl_r=-1.0),
            _make_trade(pnl_r=2.0),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["max_consecutive_losses"] == 3


# ===========================================================================
# Duration
# ===========================================================================


class TestDurationStats:
    """Test duration statistics."""

    def test_avg_duration(self):
        trades = [
            _make_trade(duration_bars=5, duration_days=7),
            _make_trade(duration_bars=15, duration_days=21),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["avg_duration_bars"] == pytest.approx(10.0)
        assert stats["avg_duration_days"] == pytest.approx(14.0)

    def test_longest_shortest(self):
        trades = [
            _make_trade(duration_bars=5),
            _make_trade(duration_bars=20),
            _make_trade(duration_bars=10),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["longest_trade_bars"] == 20
        assert stats["shortest_trade_bars"] == 5


# ===========================================================================
# Breakdowns
# ===========================================================================


class TestBreakdowns:
    """Test stat breakdowns by various dimensions."""

    def test_by_direction(self):
        trades = [
            _make_trade(pnl_r=2.0, direction="long"),
            _make_trade(pnl_r=-1.0, direction="short"),
            _make_trade(pnl_r=1.0, direction="long"),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert "long" in stats["by_direction"]
        assert "short" in stats["by_direction"]
        assert stats["by_direction"]["long"]["total_trades"] == 2
        assert stats["by_direction"]["short"]["total_trades"] == 1

    def test_by_regime(self):
        trades = [
            _make_trade(entry_regime="bull"),
            _make_trade(entry_regime="bull"),
            _make_trade(entry_regime="bear"),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["by_regime"]["bull"]["total_trades"] == 2
        assert stats["by_regime"]["bear"]["total_trades"] == 1

    def test_by_exit_reason(self):
        trades = [
            _make_trade(exit_reason="target"),
            _make_trade(exit_reason="stop"),
            _make_trade(exit_reason="target"),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["by_exit_reason"]["target"]["total_trades"] == 2
        assert stats["by_exit_reason"]["stop"]["total_trades"] == 1

    def test_by_signal_score(self):
        trades = [
            _make_trade(entry_score=3),
            _make_trade(entry_score=4),
            _make_trade(entry_score=4),
        ]
        stats = compute_backtest_stats(_make_result(trades))

        assert stats["by_signal_score"]["3"]["total_trades"] == 1
        assert stats["by_signal_score"]["4"]["total_trades"] == 2


# ===========================================================================
# Sufficient trades
# ===========================================================================


class TestSufficientTrades:
    """Test minimum trade count flag."""

    def test_sufficient_30(self):
        trades = [_make_trade() for _ in range(30)]
        stats = compute_backtest_stats(_make_result(trades))
        assert stats["sufficient_trades"] is True

    def test_insufficient_29(self):
        trades = [_make_trade() for _ in range(29)]
        stats = compute_backtest_stats(_make_result(trades))
        assert stats["sufficient_trades"] is False


# ===========================================================================
# Format report
# ===========================================================================


class TestFormatReport:
    """Test human-readable stats report."""

    def test_format_report(self):
        trades = [
            _make_trade(pnl_r=2.0, direction="long", exit_reason="target"),
            _make_trade(pnl_r=-1.0, direction="short", exit_reason="stop"),
        ]
        stats = compute_backtest_stats(_make_result(trades))
        report = format_stats_report(stats)

        assert "BACKTEST SUMMARY" in report
        assert "Win Rate" in report
        assert "Total Trades" in report
        assert "BY DIRECTION" in report

    def test_format_report_empty(self):
        stats = compute_backtest_stats(_make_result([]))
        report = format_stats_report(stats)
        assert "Total Trades:       0" in report
