"""Tests for M5 robustness analysis."""

import pandas as pd
import pytest

from trade_analysis.grid.robustness import (
    analyze_robustness,
    find_robust_zones,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_results_df(param_values: list, metric_values: list) -> pd.DataFrame:
    """Create a simple results DataFrame for robustness testing.

    One row per param value with a single metric.
    """
    return pd.DataFrame({
        "rsi_period": param_values,
        "total_r": metric_values,
        "total_trades": [50] * len(param_values),
        "win_rate": [0.5] * len(param_values),
        "avg_r": [v / 50 for v in metric_values],
        "profit_factor": [1.5] * len(param_values),
    })


# ===========================================================================
# analyze_robustness
# ===========================================================================


class TestAnalyzeRobustness:
    """Test single-parameter robustness analysis."""

    def test_flat_performance_is_robust(self):
        """Identical performance across values → all robust."""
        df = _make_results_df([10, 12, 14, 16, 18, 20], [5.0] * 6)
        result = analyze_robustness(df, "rsi_period", "total_r")
        assert len(result) == 6
        assert result["is_robust"].all()
        assert not result["is_isolated_peak"].any()

    def test_isolated_peak_detected(self):
        """One value massively outperforms neighbors → isolated peak."""
        df = _make_results_df(
            [10, 12, 14, 16, 18, 20],
            [1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
        )
        result = analyze_robustness(df, "rsi_period", "total_r")
        # Value at 14 (metric=10) vs neighbors avg 1.0 → isolated peak
        row_14 = result[result["param_value"] == 14].iloc[0]
        assert row_14["is_isolated_peak"] == True  # noqa: E712

    def test_gradual_change_is_robust(self):
        """Gradually changing performance → mostly robust."""
        df = _make_results_df(
            [10, 12, 14, 16, 18, 20],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        )
        result = analyze_robustness(df, "rsi_period", "total_r")
        # Small incremental changes should all be robust
        assert result["is_robust"].all()

    def test_single_value_is_robust(self):
        """Single parameter value → robust by default (no neighbors)."""
        df = _make_results_df([14], [5.0])
        result = analyze_robustness(df, "rsi_period", "total_r")
        assert len(result) == 1
        assert result.iloc[0]["is_robust"] == True  # noqa: E712

    def test_two_values_similar(self):
        """Two similar values → both robust."""
        df = _make_results_df([10, 14], [5.0, 5.5])
        result = analyze_robustness(df, "rsi_period", "total_r")
        assert result["is_robust"].all()

    def test_two_values_different(self):
        """Two very different values → at least one not robust."""
        df = _make_results_df([10, 14], [1.0, 10.0])
        result = analyze_robustness(df, "rsi_period", "total_r")
        assert not result["is_robust"].all()

    def test_returns_sorted_by_param(self):
        """Results should be sorted by parameter value."""
        df = _make_results_df([20, 10, 14], [1.0, 2.0, 3.0])
        result = analyze_robustness(df, "rsi_period", "total_r")
        values = result["param_value"].tolist()
        assert values == sorted(values)

    def test_unknown_param_returns_empty(self):
        """Unknown parameter name returns empty DataFrame."""
        df = _make_results_df([10, 14], [5.0, 5.0])
        result = analyze_robustness(df, "nonexistent", "total_r")
        assert result.empty

    def test_unknown_metric_returns_empty(self):
        """Unknown metric returns empty DataFrame."""
        df = _make_results_df([10, 14], [5.0, 5.0])
        result = analyze_robustness(df, "rsi_period", "nonexistent")
        assert result.empty

    def test_result_columns(self):
        """Result DataFrame has expected columns."""
        df = _make_results_df([10, 14, 20], [1.0, 2.0, 3.0])
        result = analyze_robustness(df, "rsi_period", "total_r")
        expected = {"param_value", "metric_avg", "neighbor_avg", "is_robust", "is_isolated_peak"}
        assert set(result.columns) == expected


# ===========================================================================
# analyze_robustness with multi-param grids
# ===========================================================================


class TestRobustnessMultiParam:
    """Test robustness analysis when multiple parameters are swept."""

    def test_averages_across_other_params(self):
        """metric_avg should average across other parameter values."""
        # Grid: rsi_period=[10,14], atr_period=[10,21]
        # 4 combos total
        df = pd.DataFrame({
            "rsi_period": [10, 10, 14, 14],
            "atr_period": [10, 21, 10, 21],
            "total_r": [2.0, 4.0, 3.0, 5.0],
            "total_trades": [50, 50, 50, 50],
        })
        result = analyze_robustness(df, "rsi_period", "total_r")
        # rsi=10 avg: (2+4)/2 = 3.0, rsi=14 avg: (3+5)/2 = 4.0
        assert len(result) == 2
        assert result.iloc[0]["metric_avg"] == pytest.approx(3.0)
        assert result.iloc[1]["metric_avg"] == pytest.approx(4.0)


# ===========================================================================
# find_robust_zones
# ===========================================================================


class TestFindRobustZones:
    """Test robust zone detection across parameters."""

    def test_finds_zone_in_flat_region(self):
        """Flat performance region should form a robust zone."""
        df = _make_results_df(
            [10, 12, 14, 16, 18, 20],
            [5.0, 5.1, 5.0, 4.9, 5.0, 5.1],
        )
        zones = find_robust_zones(df, "total_r")
        assert "rsi_period" in zones
        assert len(zones["rsi_period"]) >= 1
        # Zone should contain multiple contiguous values
        zone = zones["rsi_period"][0]
        assert len(zone["values"]) >= 2

    def test_no_zone_for_all_different(self):
        """Wildly different values should not form a zone."""
        df = _make_results_df(
            [10, 12, 14, 16, 18, 20],
            [1.0, 10.0, 1.0, 10.0, 1.0, 10.0],
        )
        zones = find_robust_zones(df, "total_r")
        # Alternating values → no contiguous robust zone of length >= 2
        rsi_zones = zones.get("rsi_period", [])
        assert len(rsi_zones) == 0

    def test_zone_has_avg_metric(self):
        """Each zone should report its average metric."""
        df = _make_results_df(
            [10, 12, 14, 16, 18, 20],
            [5.0] * 6,
        )
        zones = find_robust_zones(df, "total_r")
        zone = zones["rsi_period"][0]
        assert "avg_metric" in zone
        assert zone["avg_metric"] == pytest.approx(5.0)

    def test_multiple_zones_possible(self):
        """Two stable plateaus separated by a gap → two zones."""
        # Low plateau [10,12,14], then gap, then high plateau [18,20,22]
        df = _make_results_df(
            [10, 12, 14, 16, 18, 20, 22],
            [2.0, 2.1, 2.0, 8.0, 5.0, 5.1, 5.0],
        )
        zones = find_robust_zones(df, "total_r")
        assert "rsi_period" in zones
        # Should have at least one zone from the stable regions
        assert len(zones["rsi_period"]) >= 1
