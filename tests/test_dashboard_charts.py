"""Tests for dashboard chart builders.

All chart functions return plotly.graph_objects.Figure and are
tested for correct structure without rendering.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from trade_analysis.dashboard.components.charts import (
    build_breakdown_bars,
    build_distribution_histogram,
    build_equity_curve,
    build_param_heatmap,
    build_radar_comparison,
    build_robustness_chart,
    build_single_param_bar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid_df() -> pd.DataFrame:
    """Grid results DataFrame with 2 swept parameters."""
    rows = []
    for rsi in [10, 14, 18]:
        for ma in [20, 30, 40]:
            rows.append({
                "rsi_period": rsi,
                "trend_ma_period": ma,
                "total_r": rsi * 0.1 + ma * 0.05,
                "win_rate": 0.5 + rsi * 0.01,
                "avg_r": 0.2,
                "profit_factor": 1.5,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def trade_df() -> pd.DataFrame:
    """Simple trade log."""
    return pd.DataFrame({
        "pnl_r": [1.0, -0.5, 2.0, -1.0, 0.5, 1.5],
        "exit_reason": ["target", "stop", "target", "stop", "trail_stop", "target"],
    })


@pytest.fixture()
def robustness_df() -> pd.DataFrame:
    """Robustness analysis output."""
    return pd.DataFrame({
        "param_value": [10, 14, 18, 22],
        "metric_avg": [2.0, 2.1, 2.0, 1.5],
        "neighbor_avg": [2.05, 2.0, 1.75, 1.75],
        "is_robust": [True, True, True, False],
        "is_isolated_peak": [False, False, False, True],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParamHeatmap:
    def test_returns_figure(self, grid_df: pd.DataFrame) -> None:
        fig = build_param_heatmap(grid_df, "rsi_period", "trend_ma_period", "total_r")
        assert isinstance(fig, go.Figure)

    def test_heatmap_axes(self, grid_df: pd.DataFrame) -> None:
        fig = build_param_heatmap(grid_df, "rsi_period", "trend_ma_period", "total_r")
        assert fig.layout.xaxis.title.text == "rsi_period"
        assert fig.layout.yaxis.title.text == "trend_ma_period"

    def test_has_heatmap_trace(self, grid_df: pd.DataFrame) -> None:
        fig = build_param_heatmap(grid_df, "rsi_period", "trend_ma_period", "total_r")
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)


class TestSingleParamBar:
    def test_returns_figure(self, grid_df: pd.DataFrame) -> None:
        fig = build_single_param_bar(grid_df, "rsi_period", "total_r")
        assert isinstance(fig, go.Figure)

    def test_bar_count(self, grid_df: pd.DataFrame) -> None:
        fig = build_single_param_bar(grid_df, "rsi_period", "total_r")
        assert len(fig.data[0].x) == 3  # 3 unique rsi values


class TestDistributionHistogram:
    def test_returns_figure(self, grid_df: pd.DataFrame) -> None:
        fig = build_distribution_histogram(grid_df, "total_r")
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, grid_df: pd.DataFrame) -> None:
        fig = build_distribution_histogram(grid_df, "total_r", title="Custom")
        assert fig.layout.title.text == "Custom"


class TestRobustnessChart:
    def test_returns_figure(self, robustness_df: pd.DataFrame) -> None:
        fig = build_robustness_chart(robustness_df, "rsi_period", "total_r")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, robustness_df: pd.DataFrame) -> None:
        fig = build_robustness_chart(robustness_df, "rsi_period", "total_r")
        assert len(fig.data) == 2  # metric line + neighbor avg

    def test_marker_colors(self, robustness_df: pd.DataFrame) -> None:
        fig = build_robustness_chart(robustness_df, "rsi_period", "total_r")
        colors = fig.data[0].marker.color
        assert colors[-1] == "#F44336"  # last point is isolated peak
        assert colors[0] == "#4CAF50"  # first point is robust


class TestEquityCurve:
    def test_returns_figure(self, trade_df: pd.DataFrame) -> None:
        fig = build_equity_curve(trade_df)
        assert isinstance(fig, go.Figure)

    def test_cumulative_values(self, trade_df: pd.DataFrame) -> None:
        fig = build_equity_curve(trade_df)
        y_vals = list(fig.data[0].y)
        expected = [1.0, 0.5, 2.5, 1.5, 2.0, 3.5]
        assert y_vals == pytest.approx(expected)


class TestBreakdownBars:
    def test_returns_figure(self) -> None:
        breakdown = {
            "bull": {"win_rate": 0.65, "avg_r": 0.3},
            "bear": {"win_rate": 0.45, "avg_r": -0.1},
        }
        fig = build_breakdown_bars(breakdown, "win_rate", "Win Rate by Regime")
        assert isinstance(fig, go.Figure)

    def test_handles_missing_metric(self) -> None:
        breakdown = {"bull": {}, "bear": {"win_rate": 0.5}}
        fig = build_breakdown_bars(breakdown, "win_rate", "Test")
        # Groups sorted: bear=0.5, bull=0 (missing defaults to 0)
        values = list(fig.data[0].y)
        assert 0 in values
        assert 0.5 in values


class TestRadarComparison:
    def test_returns_figure(self) -> None:
        rows = [
            {"win_rate": 0.6, "avg_r": 0.2, "profit_factor": 1.5, "total_r": 5.0},
            {"win_rate": 0.55, "avg_r": 0.15, "profit_factor": 1.3, "total_r": 3.0},
        ]
        fig = build_radar_comparison(rows)
        assert isinstance(fig, go.Figure)

    def test_trace_count_matches_rows(self) -> None:
        rows = [
            {"win_rate": 0.6, "avg_r": 0.2, "profit_factor": 1.5, "total_r": 5.0},
            {"win_rate": 0.55, "avg_r": 0.15, "profit_factor": 1.3, "total_r": 3.0},
            {"win_rate": 0.5, "avg_r": 0.1, "profit_factor": 1.1, "total_r": 2.0},
        ]
        fig = build_radar_comparison(rows)
        assert len(fig.data) == 3
