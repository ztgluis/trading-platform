"""Tests for dashboard filter logic (pure functions only)."""

from __future__ import annotations

import pandas as pd
import pytest

from trade_analysis.dashboard.components.filters import (
    FilterState,
    apply_filters,
    apply_trade_filters,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "rsi_period": [10, 14, 18],
        "total_trades": [20, 40, 35],
        "total_r": [2.0, 5.0, 3.0],
        "win_rate": [0.5, 0.65, 0.55],
    })


@pytest.fixture()
def trade_df() -> pd.DataFrame:
    return pd.DataFrame({
        "entry_regime": ["bull", "bear", "bull", "transition", "bear"],
        "direction": ["long", "short", "long", "long", "short"],
        "exit_reason": ["target", "stop", "trail_stop", "stop", "target"],
        "pnl_r": [1.5, -0.5, 0.8, -1.0, 2.0],
    })


# ---------------------------------------------------------------------------
# apply_filters tests
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_no_filters_returns_all(self, grid_df: pd.DataFrame) -> None:
        filters = FilterState()
        result = apply_filters(grid_df, filters)
        assert len(result) == 3

    def test_min_trades_filter(self, grid_df: pd.DataFrame) -> None:
        filters = FilterState(min_trades=30)
        result = apply_filters(grid_df, filters)
        assert len(result) == 2
        assert 20 not in result["total_trades"].values

    def test_min_trades_zero_returns_all(self, grid_df: pd.DataFrame) -> None:
        filters = FilterState(min_trades=0)
        result = apply_filters(grid_df, filters)
        assert len(result) == 3

    def test_min_trades_high_returns_empty(self, grid_df: pd.DataFrame) -> None:
        filters = FilterState(min_trades=100)
        result = apply_filters(grid_df, filters)
        assert len(result) == 0

    def test_empty_df_returns_empty(self) -> None:
        df = pd.DataFrame()
        filters = FilterState(min_trades=30)
        result = apply_filters(df, filters)
        assert result.empty

    def test_preserves_columns(self, grid_df: pd.DataFrame) -> None:
        filters = FilterState(min_trades=30)
        result = apply_filters(grid_df, filters)
        assert set(result.columns) == set(grid_df.columns)


# ---------------------------------------------------------------------------
# apply_trade_filters tests
# ---------------------------------------------------------------------------


class TestApplyTradeFilters:
    def test_empty_filters_returns_all(self, trade_df: pd.DataFrame) -> None:
        result = apply_trade_filters(trade_df, {"regime": [], "direction": [], "exit_reason": []})
        assert len(result) == 5

    def test_filter_by_regime(self, trade_df: pd.DataFrame) -> None:
        result = apply_trade_filters(trade_df, {"regime": ["bull"], "direction": [], "exit_reason": []})
        assert len(result) == 2
        assert all(r == "bull" for r in result["entry_regime"])

    def test_filter_by_direction(self, trade_df: pd.DataFrame) -> None:
        result = apply_trade_filters(trade_df, {"regime": [], "direction": ["short"], "exit_reason": []})
        assert len(result) == 2

    def test_filter_by_exit_reason(self, trade_df: pd.DataFrame) -> None:
        result = apply_trade_filters(trade_df, {"regime": [], "direction": [], "exit_reason": ["target"]})
        assert len(result) == 2

    def test_combined_filters(self, trade_df: pd.DataFrame) -> None:
        result = apply_trade_filters(
            trade_df,
            {"regime": ["bull"], "direction": ["long"], "exit_reason": []},
        )
        assert len(result) == 2

    def test_no_matches(self, trade_df: pd.DataFrame) -> None:
        result = apply_trade_filters(
            trade_df,
            {"regime": ["bull"], "direction": ["short"], "exit_reason": []},
        )
        assert len(result) == 0

    def test_empty_df(self) -> None:
        df = pd.DataFrame()
        result = apply_trade_filters(df, {"regime": ["bull"], "direction": [], "exit_reason": []})
        assert result.empty
