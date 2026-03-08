"""Integration tests for M6 analyzer: grid results → hypothesis evaluation → report."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from trade_analysis.analyzer import (
    HypothesisResult,
    SupabaseClient,
    evaluate_all,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3,
    evaluate_h5,
    format_hypothesis_report,
)
from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.grid.config import GridConfig
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
# Full pipeline: grid → evaluate → report
# ===========================================================================


class TestFullPipeline:
    """End-to-end: grid sweep → hypothesis evaluation → report."""

    @pytest.fixture
    def grid_result(self):
        """Run a small grid sweep with multiple parameter types."""
        grid_config = GridConfig(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            parameters={
                "trend_ma_period": [10, 30],
                "rsi_period": [10, 14],
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
        return runner.run(_make_ohlcv(n=500))

    def test_evaluate_all_on_real_grid(self, grid_result):
        """evaluate_all should produce 5 results on real grid output."""
        df = grid_result.to_dataframe()
        results = evaluate_all(df)
        assert len(results) == 5
        assert all(isinstance(r, HypothesisResult) for r in results)
        ids = [r.hypothesis_id for r in results]
        assert ids == ["H1", "H2", "H3", "H4", "H5"]

    def test_all_verdicts_valid(self, grid_result):
        """All verdicts should be valid strings."""
        df = grid_result.to_dataframe()
        results = evaluate_all(df)
        valid = {"supported", "refuted", "inconclusive", "not_testable"}
        for r in results:
            assert r.verdict in valid, f"{r.hypothesis_id}: {r.verdict}"

    def test_h4_always_not_testable(self, grid_result):
        """H4 should always be not_testable."""
        df = grid_result.to_dataframe()
        results = evaluate_all(df)
        h4 = [r for r in results if r.hypothesis_id == "H4"][0]
        assert h4.verdict == "not_testable"

    def test_report_includes_all_hypotheses(self, grid_result):
        """Report should contain all 5 hypothesis IDs."""
        df = grid_result.to_dataframe()
        results = evaluate_all(df)
        report = format_hypothesis_report(results)
        for h_id in ["H1", "H2", "H3", "H4", "H5"]:
            assert h_id in report

    def test_report_is_readable(self, grid_result):
        """Report should have header and verdict indicators."""
        df = grid_result.to_dataframe()
        results = evaluate_all(df)
        report = format_hypothesis_report(results)
        assert "HYPOTHESIS EVALUATION REPORT" in report
        # At least one verdict icon should appear
        assert any(icon in report for icon in ["[+]", "[-]", "[?]", "[~]"])


# ===========================================================================
# Supabase client (no-credentials path)
# ===========================================================================


class TestSupabaseGracefulSkip:
    """Verify Supabase gracefully skips without credentials."""

    def test_client_disabled_without_env(self):
        """SupabaseClient should be disabled in test env."""
        client = SupabaseClient()
        assert client.enabled is False


# ===========================================================================
# Package imports
# ===========================================================================


class TestPackageImports:
    """Verify all exports are accessible from the package."""

    def test_import_from_analyzer_package(self):
        """All public exports should be importable."""
        from trade_analysis.analyzer import (
            VERDICTS,
            HypothesisResult,
            SupabaseClient,
            compare_groups,
            compare_metrics_by_group,
            evaluate_all,
            evaluate_h1,
            evaluate_h2,
            evaluate_h3,
            evaluate_h4,
            evaluate_h5,
            format_hypothesis_report,
            persist_grid_run,
            persist_hypothesis_results,
        )
        assert HypothesisResult is not None
        assert SupabaseClient is not None
        assert evaluate_all is not None
