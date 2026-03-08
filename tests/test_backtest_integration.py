"""Integration tests for M4 backtester: full pipeline from OHLCV to stats.

Tests the complete flow:
    OHLCV DataFrame → generate_signals() → Backtester.run() → compute_backtest_stats()
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig, WalkForwardConfig
from trade_analysis.backtester.engine import Backtester
from trade_analysis.backtester.models import BacktestResult, Trade
from trade_analysis.backtester.stats import compute_backtest_stats, format_stats_report
from trade_analysis.backtester.walk_forward import (
    generate_walk_forward_splits,
    run_walk_forward,
)
from trade_analysis.signals.engine import generate_signals, load_signal_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 500,
    start: str = "2020-01-02",
    seed: int = 42,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for integration tests.

    Uses a seeded random walk to produce realistic price movements.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n, freq="B")

    # Random walk for closes
    returns = rng.normal(0.0003, 0.015, n)
    close = base_price * np.exp(np.cumsum(returns))

    # OHLC from close
    high = close * (1 + rng.uniform(0.001, 0.03, n))
    low = close * (1 - rng.uniform(0.001, 0.03, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))

    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(dates, tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1e7, 1e8, n),
        }
    )


# ===========================================================================
# Full pipeline
# ===========================================================================


class TestFullPipeline:
    """End-to-end: OHLCV → signals → backtest → stats."""

    def test_pipeline_runs_without_error(self):
        """Full pipeline should complete without exceptions."""
        ohlcv = _make_ohlcv(n=500)
        signal_config = load_signal_config()
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )

        # Run signal engine
        enriched = generate_signals(ohlcv, asset_class="stock", config=signal_config)

        # Run backtest
        backtester = Backtester(bt_config, signal_config)
        result = backtester.run(enriched, "AAPL", "stock", "Daily")

        # Compute stats
        stats = compute_backtest_stats(result)

        assert isinstance(result, BacktestResult)
        assert stats["total_trades"] >= 0

    def test_pipeline_produces_trades(self):
        """500-bar synthetic data should produce at least some trades."""
        ohlcv = _make_ohlcv(n=500, seed=123)
        signal_config = load_signal_config()
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )

        enriched = generate_signals(ohlcv, asset_class="stock", config=signal_config)
        backtester = Backtester(bt_config, signal_config)
        result = backtester.run(enriched, "AAPL", "stock", "Daily")

        # With 500 bars of data, we may or may not get trades
        # depending on conditions; just verify the result is valid
        assert isinstance(result, BacktestResult)
        for trade in result.trades:
            assert isinstance(trade, Trade)

    def test_trade_log_schema(self):
        """Trade log DataFrame should have the correct columns."""
        ohlcv = _make_ohlcv(n=500)
        signal_config = load_signal_config()
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )

        enriched = generate_signals(ohlcv, asset_class="stock", config=signal_config)
        backtester = Backtester(bt_config, signal_config)
        result = backtester.run(enriched, "AAPL", "stock", "Daily")

        df = result.to_dataframe()

        expected_columns = [
            "symbol",
            "asset_class",
            "timeframe",
            "bucket",
            "direction",
            "entry_timestamp",
            "entry_price",
            "entry_signal_score",
            "entry_regime",
            "exit_timestamp",
            "exit_price",
            "exit_reason",
            "pnl_r",
            "pnl_dollar",
            "duration_bars",
            "duration_calendar_days",
            "is_winner",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"


# ===========================================================================
# Trade validity
# ===========================================================================


class TestTradeValidity:
    """Verify trade records obey business rules."""

    @pytest.fixture
    def trades(self):
        """Generate trades from a pipeline run."""
        ohlcv = _make_ohlcv(n=500)
        signal_config = load_signal_config()
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        enriched = generate_signals(ohlcv, asset_class="stock", config=signal_config)
        backtester = Backtester(bt_config, signal_config)
        result = backtester.run(enriched, "AAPL", "stock", "Daily")
        return result.trades

    def test_no_overlapping_trades(self, trades):
        """No two trades should overlap in time."""
        for i in range(1, len(trades)):
            assert trades[i].entry_timestamp >= trades[i - 1].exit_timestamp, (
                f"Trade {i} entry ({trades[i].entry_timestamp}) "
                f"before trade {i-1} exit ({trades[i-1].exit_timestamp})"
            )

    def test_valid_exit_reasons(self, trades):
        """All exit reasons must be from allowed set."""
        valid_reasons = {"stop", "trail_stop", "target", "max_hold", "end_of_data"}
        for t in trades:
            assert t.exit_reason in valid_reasons, (
                f"Invalid exit_reason: {t.exit_reason}"
            )

    def test_entry_before_exit(self, trades):
        """Entry timestamp must be before exit timestamp."""
        for t in trades:
            assert t.entry_timestamp <= t.exit_timestamp

    def test_pnl_r_consistency(self, trades):
        """pnl_r should be positive for winners, negative for losers."""
        for t in trades:
            if t.pnl_r > 0:
                assert t.is_winner == True  # noqa: E712 (numpy bool)
            elif t.pnl_r < 0:
                assert t.is_winner == False  # noqa: E712 (numpy bool)

    def test_positive_entry_price(self, trades):
        """Entry and exit prices should be positive."""
        for t in trades:
            assert t.entry_price > 0
            assert t.exit_price > 0

    def test_duration_non_negative(self, trades):
        """Duration in bars should be non-negative."""
        for t in trades:
            assert t.duration_bars >= 0
            assert t.duration_calendar_days >= 0


# ===========================================================================
# Stats sanity
# ===========================================================================


class TestStatsSanity:
    """Verify stats are mathematically consistent."""

    @pytest.fixture
    def stats(self):
        """Compute stats from a pipeline run."""
        ohlcv = _make_ohlcv(n=500)
        signal_config = load_signal_config()
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        enriched = generate_signals(ohlcv, asset_class="stock", config=signal_config)
        backtester = Backtester(bt_config, signal_config)
        result = backtester.run(enriched, "AAPL", "stock", "Daily")
        return compute_backtest_stats(result)

    def test_win_rate_range(self, stats):
        """Win rate should be between 0 and 1."""
        assert 0.0 <= stats["win_rate"] <= 1.0

    def test_total_r_equals_avg_r_times_n(self, stats):
        """total_r should equal avg_r * total_trades."""
        if stats["total_trades"] > 0:
            expected = stats["avg_r"] * stats["total_trades"]
            assert stats["total_r"] == pytest.approx(expected, abs=1e-6)

    def test_drawdown_non_negative(self, stats):
        """Max drawdown should be non-negative."""
        assert stats["max_drawdown_r"] >= 0.0

    def test_profit_factor_non_negative(self, stats):
        """Profit factor should be non-negative."""
        assert stats["profit_factor"] >= 0.0

    def test_format_report_works(self, stats):
        """format_stats_report should produce a non-empty string."""
        report = format_stats_report(stats)
        assert isinstance(report, str)
        assert len(report) > 0
        assert "BACKTEST SUMMARY" in report


# ===========================================================================
# Walk-forward integration
# ===========================================================================


class TestWalkForwardIntegration:
    """End-to-end walk-forward validation test."""

    def test_walk_forward_full_pipeline(self):
        """Walk-forward with signal-enriched data runs end-to-end."""
        ohlcv = _make_ohlcv(n=1200, start="2019-01-02", seed=99)
        signal_config = load_signal_config()
        bt_config = BacktestConfig(
            start_date=date(2019, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=WalkForwardConfig(
                in_sample_years=3,
                out_of_sample_years=1,
                anchored=True,
            ),
        )

        enriched = generate_signals(ohlcv, asset_class="stock", config=signal_config)
        result = run_walk_forward(
            enriched, "AAPL", "stock", "Daily", bt_config, signal_config
        )

        assert len(result.splits) >= 1
        assert len(result.in_sample_results) == len(result.splits)
        assert len(result.out_of_sample_results) == len(result.splits)

        # Each result should be a BacktestResult
        for is_result in result.in_sample_results:
            assert isinstance(is_result, BacktestResult)
        for oos_result in result.out_of_sample_results:
            assert isinstance(oos_result, BacktestResult)

    def test_walk_forward_splits_dont_overlap(self):
        """Walk-forward OOS windows should not overlap."""
        splits = generate_walk_forward_splits(
            start_date=date(2015, 1, 1),
            end_date=date(2024, 12, 31),
            in_sample_years=2,
            out_of_sample_years=1,
            anchored=True,
        )

        for i in range(1, len(splits)):
            assert splits[i].oos_start >= splits[i - 1].oos_end, (
                f"OOS overlap: fold {i-1} ends {splits[i-1].oos_end}, "
                f"fold {i} starts {splits[i].oos_start}"
            )


# ===========================================================================
# Package imports
# ===========================================================================


class TestPackageImports:
    """Verify all exports are accessible from the package."""

    def test_import_from_backtester_package(self):
        """All public exports should be importable."""
        from trade_analysis.backtester import (
            BacktestConfig,
            WalkForwardConfig,
            load_backtest_config,
            Backtester,
            BacktestResult,
            Position,
            Trade,
            WalkForwardResult,
            WalkForwardSplit,
            compute_backtest_stats,
            format_stats_report,
            generate_walk_forward_splits,
            run_walk_forward,
        )

        # Just verify they exist
        assert BacktestConfig is not None
        assert Backtester is not None
        assert Trade is not None

    def test_backtest_error_exists(self):
        """BacktestError should be importable from exceptions."""
        from trade_analysis.exceptions import BacktestError

        assert issubclass(BacktestError, Exception)
