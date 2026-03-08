"""Shared filter widgets and filter-application logic.

Pure filter logic (apply_filters) is independently testable.
Widget functions use Streamlit and are tested manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Filter state
# ---------------------------------------------------------------------------


@dataclass
class FilterState:
    """Holds current filter selections."""

    regimes: list[str] = field(default_factory=list)
    directions: list[str] = field(default_factory=list)
    min_trades: int = 0
    metric: str = "total_r"


# ---------------------------------------------------------------------------
# Filter application (pure — testable)
# ---------------------------------------------------------------------------


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    """Apply filter selections to a grid results DataFrame.

    Filters are applied only when non-empty; an empty filter list means
    'show all'.
    """
    if df.empty:
        return df

    result = df.copy()

    if filters.min_trades > 0 and "total_trades" in result.columns:
        result = result[result["total_trades"] >= filters.min_trades]

    return result


# ---------------------------------------------------------------------------
# Sidebar widgets (Streamlit-dependent)
# ---------------------------------------------------------------------------


def render_sidebar_filters() -> FilterState:
    """Render filter widgets in the sidebar and return current selections."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    metric = st.sidebar.selectbox(
        "Rank by",
        options=["total_r", "avg_r", "profit_factor", "win_rate"],
        index=0,
    )

    min_trades = st.sidebar.number_input(
        "Min trades",
        min_value=0,
        value=30,
        step=5,
    )

    regimes = st.sidebar.multiselect(
        "Regime",
        options=["bull", "bear", "transition"],
        default=[],
    )

    directions = st.sidebar.multiselect(
        "Direction",
        options=["long", "short"],
        default=[],
    )

    return FilterState(
        regimes=regimes,
        directions=directions,
        min_trades=min_trades,
        metric=metric,
    )


def render_trade_filters() -> dict[str, list[str]]:
    """Render trade-level filter widgets. Returns dict of filter selections."""
    col1, col2, col3 = st.columns(3)

    with col1:
        regimes = st.multiselect(
            "Regime",
            options=["bull", "bear", "transition"],
            default=[],
            key="trade_regime",
        )
    with col2:
        directions = st.multiselect(
            "Direction",
            options=["long", "short"],
            default=[],
            key="trade_direction",
        )
    with col3:
        exit_reasons = st.multiselect(
            "Exit reason",
            options=["stop", "trail_stop", "target", "max_hold", "end_of_data"],
            default=[],
            key="trade_exit_reason",
        )

    return {
        "regime": regimes,
        "direction": directions,
        "exit_reason": exit_reasons,
    }


def apply_trade_filters(
    df: pd.DataFrame,
    filters: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply trade-level filters to a trade log DataFrame."""
    if df.empty:
        return df

    result = df.copy()

    for col, values in filters.items():
        col_name = f"entry_{col}" if col == "regime" else col
        if values and col_name in result.columns:
            result = result[result[col_name].isin(values)]

    return result
