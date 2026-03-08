"""Tests for M5 grid runner and grid result."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.grid.config import GridConfig
from trade_analysis.grid.runner import GridResult, GridRunner
from trade_analysis.signals.engine import load_signal_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for grid tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-02", periods=n, freq="B")
    returns = rng.normal(0.0003, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.03, n))
    low = close * (1 - rng.uniform(0.001, 0.03, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex(dates, tz="UTC"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(1e7, 1e8, n),
    })


def _small_grid_config(**overrides) -> GridConfig:
    """Create a small grid config for fast tests."""
    defaults = {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "Daily",
        "parameters": {"rsi_period": [10, 14]},
        "min_trades": 0,  # no filtering for basic tests
        "rank_by": "total_r",
    }
    defaults.update(overrides)
    return GridConfig(**defaults)


def _bt_config() -> BacktestConfig:
    return BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        initial_capital=100000.0,
        max_open_positions=1,
        walk_forward=None,
    )


# ===========================================================================
# GridRunner
# ===========================================================================


class TestGridRunner:
    """Test the grid runner pipeline."""

    def test_runs_without_error(self):
        """Grid runner should complete without exceptions."""
        grid_config = _small_grid_config()
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        assert isinstance(result, GridResult)

    def test_produces_correct_combo_count(self):
        """Number of results matches number of parameter combos."""
        grid_config = _small_grid_config(
            parameters={"rsi_period": [10, 14, 20]}
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        assert result.total_combos == 3

    def test_two_params_combo_count(self):
        """Two params produce correct cartesian product count."""
        grid_config = _small_grid_config(
            parameters={"rsi_period": [10, 14], "atr_period": [10, 21]}
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        assert result.total_combos == 4

    def test_results_contain_param_columns(self):
        """Results DataFrame should include parameter columns."""
        grid_config = _small_grid_config(
            parameters={"rsi_period": [10, 14]}
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        df = result.to_dataframe()
        assert "rsi_period" in df.columns

    def test_results_contain_stat_columns(self):
        """Results DataFrame should include stat columns."""
        grid_config = _small_grid_config()
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        df = result.to_dataframe()
        for col in ["total_trades", "win_rate", "avg_r", "total_r", "profit_factor"]:
            assert col in df.columns, f"Missing stat column: {col}"

    def test_different_params_can_produce_different_results(self):
        """Different parameter values should potentially produce different stats."""
        grid_config = _small_grid_config(
            parameters={"rsi_period": [5, 25]}
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=500))
        df = result.to_dataframe()
        # The two rows may or may not have different results,
        # but params should differ
        assert df.iloc[0]["rsi_period"] != df.iloc[1]["rsi_period"]

    def test_bucket_param_sweep(self):
        """Bucket-specific params (trend_ma_period) should work."""
        grid_config = _small_grid_config(
            parameters={"trend_ma_period": [10, 30]}
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        assert result.total_combos == 2


# ===========================================================================
# GridResult
# ===========================================================================


class TestGridResult:
    """Test GridResult methods."""

    @pytest.fixture
    def sample_result(self):
        """Run a small grid to get a real result."""
        grid_config = _small_grid_config(
            parameters={"rsi_period": [10, 14, 20]},
            min_trades=0,
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        return runner.run(_make_ohlcv(n=500))

    def test_to_dataframe(self, sample_result):
        """to_dataframe should return a DataFrame with all combos."""
        df = sample_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_rank_returns_sorted(self, sample_result):
        """rank() should return sorted DataFrame with rank column."""
        ranked = sample_result.rank()
        if not ranked.empty:
            assert "rank" in ranked.columns
            assert ranked["rank"].iloc[0] == 1
            # Should be sorted descending
            values = ranked["total_r"].tolist()
            assert values == sorted(values, reverse=True)

    def test_top_n(self, sample_result):
        """top_n should return at most N rows."""
        top = sample_result.top_n(2)
        assert len(top) <= 2

    def test_sufficient_only_with_min_trades(self):
        """sufficient_only filters by min_trades."""
        grid_config = _small_grid_config(
            parameters={"rsi_period": [10, 14]},
            min_trades=9999,  # impossibly high
        )
        runner = GridRunner(grid_config, _bt_config(), load_signal_config())
        result = runner.run(_make_ohlcv(n=300))
        df = result.sufficient_only()
        assert len(df) == 0  # nothing should pass

    def test_rank_by_custom_metric(self, sample_result):
        """rank(by='win_rate') should sort by win_rate."""
        ranked = sample_result.rank(by="win_rate")
        if len(ranked) >= 2:
            values = ranked["win_rate"].tolist()
            assert values == sorted(values, reverse=True)

    def test_format_report(self, sample_result):
        """format_report should produce non-empty string."""
        report = sample_result.format_report()
        assert isinstance(report, str)
        assert "GRID SWEEP RESULTS" in report
        assert "Total combinations" in report

    def test_empty_result(self):
        """Empty GridResult should handle gracefully."""
        result = GridResult(rows=[], grid_config=None)
        assert result.total_combos == 0
        assert result.to_dataframe().empty
        assert result.rank().empty
