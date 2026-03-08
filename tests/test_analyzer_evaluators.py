"""Tests for M6 hypothesis evaluators H1-H5."""

import pandas as pd
import pytest

from trade_analysis.analyzer.evaluators import (
    evaluate_all,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3,
    evaluate_h4,
    evaluate_h5,
)
from trade_analysis.analyzer.hypothesis import HypothesisResult


# ===========================================================================
# Helpers
# ===========================================================================


def _make_grid_df(
    trend_ma_period: list | None = None,
    trend_ma_type: list | None = None,
    rsi_period: list | None = None,
    rsi_bull_threshold: list | None = None,
    avg_r: list | None = None,
    win_rate: list | None = None,
    total_trades: list | None = None,
) -> pd.DataFrame:
    """Build a synthetic grid results DataFrame."""
    data = {}
    n = None
    if trend_ma_period is not None:
        data["trend_ma_period"] = trend_ma_period
        n = len(trend_ma_period)
    if trend_ma_type is not None:
        data["trend_ma_type"] = trend_ma_type
        n = len(trend_ma_type)
    if rsi_period is not None:
        data["rsi_period"] = rsi_period
        n = len(rsi_period)
    if rsi_bull_threshold is not None:
        data["rsi_bull_threshold"] = rsi_bull_threshold
        n = len(rsi_bull_threshold)
    if avg_r is not None:
        data["avg_r"] = avg_r
    elif n:
        data["avg_r"] = [0.1] * n
    if win_rate is not None:
        data["win_rate"] = win_rate
    elif n:
        data["win_rate"] = [0.5] * n
    if total_trades is not None:
        data["total_trades"] = total_trades
    elif n:
        data["total_trades"] = [50] * n
    return pd.DataFrame(data)


# ===========================================================================
# H1: Trend filter impact
# ===========================================================================


class TestEvaluateH1:
    """Test H1: Does applying any trend filter improve results?"""

    def test_supported_when_filter_improves(self):
        """Larger MA periods outperform smallest → supported."""
        df = _make_grid_df(
            trend_ma_period=[5, 20, 50],
            avg_r=[0.0, 0.2, 0.3],
        )
        result = evaluate_h1(df)
        assert result.hypothesis_id == "H1"
        assert result.verdict == "supported"
        assert result.evidence["smallest_period"] == 5

    def test_refuted_when_no_filter_is_better(self):
        """Smallest period outperforms → refuted."""
        df = _make_grid_df(
            trend_ma_period=[5, 20, 50],
            avg_r=[0.4, 0.1, 0.0],
        )
        result = evaluate_h1(df)
        assert result.verdict == "refuted"

    def test_inconclusive_when_similar(self):
        """Similar performance → inconclusive."""
        df = _make_grid_df(
            trend_ma_period=[5, 20, 50],
            avg_r=[0.10, 0.11, 0.12],
        )
        result = evaluate_h1(df)
        assert result.verdict == "inconclusive"

    def test_inconclusive_when_column_missing(self):
        """Missing trend_ma_period → inconclusive."""
        df = _make_grid_df(rsi_period=[10, 14], avg_r=[0.1, 0.2])
        result = evaluate_h1(df)
        assert result.verdict == "inconclusive"
        assert "not swept" in result.summary

    def test_inconclusive_single_value(self):
        """Only one period value → inconclusive."""
        df = _make_grid_df(trend_ma_period=[20], avg_r=[0.1])
        result = evaluate_h1(df)
        assert result.verdict == "inconclusive"


# ===========================================================================
# H2: EMA vs SMA
# ===========================================================================


class TestEvaluateH2:
    """Test H2: Does EMA vs SMA type matter?"""

    def test_refuted_when_type_negligible(self):
        """Negligible difference between types → refuted."""
        df = _make_grid_df(
            trend_ma_type=["ema", "ema", "sma", "sma"],
            avg_r=[0.10, 0.12, 0.11, 0.11],
        )
        result = evaluate_h2(df)
        assert result.hypothesis_id == "H2"
        assert result.verdict == "refuted"

    def test_supported_when_type_matters(self):
        """Large difference between types → supported."""
        df = _make_grid_df(
            trend_ma_type=["ema", "ema", "sma", "sma"],
            avg_r=[0.5, 0.6, 0.1, 0.0],
        )
        result = evaluate_h2(df)
        assert result.verdict == "supported"

    def test_refuted_when_period_dominates(self):
        """Period spread >> type spread → refuted."""
        df = pd.DataFrame({
            "trend_ma_type": ["ema", "ema", "sma", "sma"],
            "trend_ma_period": [10, 50, 10, 50],
            "avg_r": [0.3, 0.1, 0.28, 0.12],
            "win_rate": [0.5] * 4,
            "total_trades": [50] * 4,
        })
        result = evaluate_h2(df)
        assert result.verdict == "refuted"
        assert result.evidence["period_spread"] > result.evidence["type_spread"]

    def test_inconclusive_when_column_missing(self):
        """Missing trend_ma_type → inconclusive."""
        df = _make_grid_df(rsi_period=[10, 14], avg_r=[0.1, 0.2])
        result = evaluate_h2(df)
        assert result.verdict == "inconclusive"


# ===========================================================================
# H3: Best period
# ===========================================================================


class TestEvaluateH3:
    """Test H3: What period produces the best risk-adjusted returns?"""

    def test_finds_best_period(self):
        """Should identify the period with highest avg_r."""
        df = _make_grid_df(
            trend_ma_period=[10, 20, 30, 40, 50],
            avg_r=[0.1, 0.3, 0.5, 0.4, 0.2],
        )
        result = evaluate_h3(df)
        assert result.hypothesis_id == "H3"
        assert result.verdict == "supported"
        assert result.evidence["best_period"] == 30

    def test_robust_when_neighbors_close(self):
        """Neighbors close to best → robust."""
        df = _make_grid_df(
            trend_ma_period=[10, 20, 30, 40, 50],
            avg_r=[0.1, 0.3, 0.35, 0.32, 0.2],
        )
        result = evaluate_h3(df)
        assert result.evidence["is_robust"] is True

    def test_isolated_when_neighbors_far(self):
        """Neighbors far from best → isolated (potential overfit)."""
        df = _make_grid_df(
            trend_ma_period=[10, 20, 30, 40, 50],
            avg_r=[0.0, 0.0, 1.0, 0.0, 0.0],
        )
        result = evaluate_h3(df)
        assert result.evidence["is_robust"] is False

    def test_includes_ranking(self):
        """Evidence should include sorted ranking."""
        df = _make_grid_df(
            trend_ma_period=[10, 20, 30],
            avg_r=[0.1, 0.3, 0.2],
        )
        result = evaluate_h3(df)
        ranking = result.evidence["ranking"]
        assert ranking[0]["period"] == 20
        assert len(ranking) == 3

    def test_inconclusive_when_column_missing(self):
        """Missing trend_ma_period → inconclusive."""
        df = _make_grid_df(rsi_period=[10, 14], avg_r=[0.1, 0.2])
        result = evaluate_h3(df)
        assert result.verdict == "inconclusive"


# ===========================================================================
# H4: Single MA vs crossover
# ===========================================================================


class TestEvaluateH4:
    """Test H4: Single MA vs fast/slow crossover."""

    def test_not_testable(self):
        """H4 should always return not_testable."""
        result = evaluate_h4()
        assert result.hypothesis_id == "H4"
        assert result.verdict == "not_testable"
        assert "crossover" in result.summary.lower()


# ===========================================================================
# H5: RSI threshold impact
# ===========================================================================


class TestEvaluateH5:
    """Test H5: Does RSI > 50 improve win rate or only reduce frequency?"""

    def test_supported_when_wr_improves_trades_drop(self):
        """Higher threshold improves WR but reduces trades → supported."""
        df = _make_grid_df(
            rsi_bull_threshold=[40, 50, 60],
            win_rate=[0.40, 0.50, 0.65],
            total_trades=[100, 80, 50],
            avg_r=[0.1, 0.2, 0.3],
        )
        result = evaluate_h5(df)
        assert result.hypothesis_id == "H5"
        assert result.verdict == "supported"

    def test_supported_when_wr_improves_no_trade_drop(self):
        """Higher threshold improves WR without reducing trades → supported."""
        df = _make_grid_df(
            rsi_bull_threshold=[40, 50, 60],
            win_rate=[0.40, 0.50, 0.65],
            total_trades=[80, 85, 90],
            avg_r=[0.1, 0.2, 0.3],
        )
        result = evaluate_h5(df)
        assert result.verdict == "supported"

    def test_refuted_when_only_reduces_trades(self):
        """Higher threshold reduces trades without improving WR → refuted."""
        df = _make_grid_df(
            rsi_bull_threshold=[40, 50, 60],
            win_rate=[0.50, 0.50, 0.50],
            total_trades=[100, 70, 40],
            avg_r=[0.1, 0.1, 0.1],
        )
        result = evaluate_h5(df)
        assert result.verdict == "refuted"

    def test_falls_back_to_rsi_period(self):
        """Should use rsi_period if rsi_bull_threshold not present."""
        df = _make_grid_df(
            rsi_period=[10, 14, 20],
            win_rate=[0.40, 0.50, 0.65],
            total_trades=[100, 80, 50],
            avg_r=[0.1, 0.2, 0.3],
        )
        result = evaluate_h5(df)
        assert result.evidence["parameter"] == "rsi_period"
        assert result.verdict == "supported"

    def test_inconclusive_when_no_rsi_column(self):
        """No RSI columns → inconclusive."""
        df = _make_grid_df(trend_ma_period=[10, 20], avg_r=[0.1, 0.2])
        result = evaluate_h5(df)
        assert result.verdict == "inconclusive"

    def test_inconclusive_single_value(self):
        """Single RSI value → inconclusive."""
        df = _make_grid_df(
            rsi_bull_threshold=[50],
            win_rate=[0.5],
            total_trades=[80],
            avg_r=[0.1],
        )
        result = evaluate_h5(df)
        assert result.verdict == "inconclusive"


# ===========================================================================
# evaluate_all
# ===========================================================================


class TestEvaluateAll:
    """Test the evaluate_all convenience function."""

    def test_returns_five_results(self):
        """Should return exactly 5 HypothesisResults."""
        df = _make_grid_df(
            trend_ma_period=[10, 20, 30],
            avg_r=[0.1, 0.2, 0.3],
        )
        results = evaluate_all(df)
        assert len(results) == 5
        assert all(isinstance(r, HypothesisResult) for r in results)

    def test_hypothesis_ids(self):
        """Should have H1-H5 in order."""
        df = _make_grid_df(trend_ma_period=[10, 20], avg_r=[0.1, 0.2])
        results = evaluate_all(df)
        ids = [r.hypothesis_id for r in results]
        assert ids == ["H1", "H2", "H3", "H4", "H5"]

    def test_all_verdicts_valid(self):
        """All results should have valid verdicts."""
        df = _make_grid_df(
            trend_ma_period=[10, 20],
            avg_r=[0.1, 0.2],
        )
        results = evaluate_all(df)
        valid = {"supported", "refuted", "inconclusive", "not_testable"}
        for r in results:
            assert r.verdict in valid
