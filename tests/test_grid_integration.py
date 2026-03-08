"""Integration tests for M5 grid runner: full pipeline from OHLCV to ranked results.

Tests the complete flow:
    OHLCV DataFrame → GridRunner.run() → GridResult → robustness analysis
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.grid.config import GridConfig
from trade_analysis.grid.robustness import analyze_robustness, find_robust_zones
from trade_analysis.grid.runner import GridResult, GridRunner
from trade_analysis.signals.engine import load_signal_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
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


# ===========================================================================
# Full pipeline
# ===========================================================================


class TestFullPipeline:
    """End-to-end: OHLCV → grid sweep → ranked results."""

    @pytest.fixture
    def grid_result(self):
        """Run a small grid sweep."""
        grid_config = GridConfig(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            parameters={"rsi_period": [10, 14, 20]},
            min_trades=0,
            rank_by="total_r",
        )
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        signal_config = load_signal_config()
        runner = GridRunner(grid_config, bt_config, signal_config)
        return runner.run(_make_ohlcv(n=500))

    def test_pipeline_completes(self, grid_result):
        """Full pipeline should complete without exceptions."""
        assert isinstance(grid_result, GridResult)
        assert grid_result.total_combos == 3

    def test_results_have_correct_param_values(self, grid_result):
        """Each result row should have the correct parameter values."""
        df = grid_result.to_dataframe()
        rsi_values = sorted(df["rsi_period"].tolist())
        assert rsi_values == [10, 14, 20]

    def test_results_have_stats(self, grid_result):
        """Each result row should have backtest stats."""
        df = grid_result.to_dataframe()
        for _, row in df.iterrows():
            assert "total_trades" in row
            assert "win_rate" in row
            assert "total_r" in row

    def test_ranking_works(self, grid_result):
        """Ranking should produce sorted results."""
        ranked = grid_result.rank()
        if len(ranked) >= 2:
            assert ranked.iloc[0]["total_r"] >= ranked.iloc[1]["total_r"]

    def test_format_report_produces_output(self, grid_result):
        """format_report should produce readable output."""
        report = grid_result.format_report()
        assert "GRID SWEEP RESULTS" in report
        assert "Total combinations" in report


# ===========================================================================
# Multi-parameter sweep
# ===========================================================================


class TestMultiParamSweep:
    """Test sweeping multiple parameters together."""

    def test_rsi_and_ma_sweep(self):
        """Sweep RSI period and MA period (PRD exit criteria)."""
        grid_config = GridConfig(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            parameters={
                "rsi_period": [10, 14, 20],
                "trend_ma_period": [10, 30],
            },
            min_trades=0,
            rank_by="total_r",
        )
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        signal_config = load_signal_config()
        runner = GridRunner(grid_config, bt_config, signal_config)
        result = runner.run(_make_ohlcv(n=500))

        assert result.total_combos == 6  # 3 × 2
        df = result.to_dataframe()
        assert "rsi_period" in df.columns
        assert "trend_ma_period" in df.columns


# ===========================================================================
# Robustness on real results
# ===========================================================================


class TestRobustnessIntegration:
    """Test robustness analysis on actual grid results."""

    def test_robustness_on_grid_results(self):
        """analyze_robustness should work on real grid output."""
        grid_config = GridConfig(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            parameters={"rsi_period": [10, 12, 14, 16, 18, 20]},
            min_trades=0,
            rank_by="total_r",
        )
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        signal_config = load_signal_config()
        runner = GridRunner(grid_config, bt_config, signal_config)
        result = runner.run(_make_ohlcv(n=500))

        df = result.to_dataframe()
        robustness = analyze_robustness(df, "rsi_period", "total_r")
        assert len(robustness) == 6
        assert "is_robust" in robustness.columns
        assert "is_isolated_peak" in robustness.columns

    def test_find_robust_zones_on_results(self):
        """find_robust_zones should work on real grid output."""
        grid_config = GridConfig(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            parameters={"rsi_period": [10, 12, 14, 16, 18, 20]},
            min_trades=0,
            rank_by="total_r",
        )
        bt_config = BacktestConfig(
            start_date=date(2020, 1, 1),
            end_date=date(2024, 12, 31),
            initial_capital=100000.0,
            max_open_positions=1,
            walk_forward=None,
        )
        signal_config = load_signal_config()
        runner = GridRunner(grid_config, bt_config, signal_config)
        result = runner.run(_make_ohlcv(n=500))

        zones = find_robust_zones(result.to_dataframe(), "total_r")
        # zones is a dict — may or may not have zones depending on data
        assert isinstance(zones, dict)


# ===========================================================================
# Package imports
# ===========================================================================


class TestPackageImports:
    """Verify all exports are accessible from the package."""

    def test_import_from_grid_package(self):
        """All public exports should be importable."""
        from trade_analysis.grid import (
            GridConfig,
            load_grid_config,
            ALL_KNOWN_PARAMS,
            apply_params_to_config,
            generate_parameter_grid,
            GridResult,
            GridRunner,
            analyze_robustness,
            find_robust_zones,
        )
        assert GridConfig is not None
        assert GridRunner is not None
        assert ALL_KNOWN_PARAMS is not None
