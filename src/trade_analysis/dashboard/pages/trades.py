"""Trades page — trade-level breakdown + equity curve."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trade_analysis.dashboard.components.charts import (
    build_breakdown_bars,
    build_equity_curve,
)
from trade_analysis.dashboard.components.labels import col_label
from trade_analysis.dashboard.components.metrics import render_metric_row
from trade_analysis.dashboard.pages.grid_results import _DISPLAY_STATS, _detect_param_cols


def _build_exit_reason_pie(breakdown: dict[str, dict]) -> go.Figure:
    """Pie chart of trade counts by exit reason."""
    labels = sorted(breakdown.keys())
    values = [breakdown[k].get("total_trades", 0) for k in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        hovertemplate="%{label}<br>Trades: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(title="Trades by Exit Reason", height=350)
    return fig


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the trades page."""
    st.header("Trade Analysis")

    if grid_df.empty:
        st.info("No grid results loaded.")
        return

    param_cols = _detect_param_cols(grid_df)
    stat_cols = [c for c in _DISPLAY_STATS if c in grid_df.columns]

    if not param_cols:
        st.warning("No parameter columns found.")
        return

    # ---- Combo selector ----
    ranked = grid_df.sort_values("total_r", ascending=False).reset_index(drop=True)
    combo_labels = []
    for i, (_, row) in enumerate(ranked.iterrows()):
        parts = [f"{col_label(p)}={row[p]}" for p in param_cols if p in row]
        combo_labels.append(f"#{i + 1}: {', '.join(parts)}")

    selected_idx = st.selectbox(
        "Select parameter combination",
        options=list(range(len(combo_labels))),
        format_func=lambda x: combo_labels[x],
        key="trade_combo",
    )

    row = ranked.iloc[selected_idx]

    # ---- Aggregate metrics ----
    _metric_keys = [
        "total_trades", "win_rate", "avg_r", "total_r",
        "profit_factor", "max_drawdown_r",
    ]
    metrics = {}
    for mk in _metric_keys:
        if mk in row:
            val = int(row[mk]) if mk == "total_trades" else row[mk]
            metrics[col_label(mk)] = (val, None)

    if metrics:
        render_metric_row(metrics)

    st.markdown("---")

    # ---- Breakdown charts ----
    breakdown_keys = {
        "by_regime": "Performance by Regime",
        "by_direction": "Performance by Direction",
        "by_exit_reason": "Performance by Exit Reason",
        "by_signal_score": "Performance by Signal Score",
    }

    available_breakdowns = {
        k: v for k, v in breakdown_keys.items()
        if k in row and isinstance(row[k], dict) and row[k]
    }

    if available_breakdowns:
        st.subheader("Breakdowns")

        # 2x2 grid
        keys = list(available_breakdowns.keys())
        for i in range(0, len(keys), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(keys):
                    break
                key = keys[idx]
                breakdown = row[key]
                title = available_breakdowns[key]

                with col:
                    if key == "by_exit_reason":
                        fig = _build_exit_reason_pie(breakdown)
                    else:
                        fig = build_breakdown_bars(breakdown, "avg_r", title)
                    st.plotly_chart(fig, use_container_width=True)

        # Win rate breakdown for regime and direction
        st.subheader("Win Rate Breakdowns")
        wr_cols = st.columns(2)
        for col_idx, key in enumerate(["by_regime", "by_direction"]):
            if key in available_breakdowns:
                with wr_cols[col_idx]:
                    fig = build_breakdown_bars(
                        row[key], "win_rate", f"Win Rate {available_breakdowns[key]}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ---- Equity curve placeholder ----
    # The grid results store aggregated stats, not individual trades.
    # For a real equity curve we'd need to re-run the backtest for this combo
    # or load from backtest_trades table.
    st.subheader("Equity Curve")
    st.caption(
        "To view the equity curve, run a single backtest for this parameter "
        "combination. Grid results contain aggregated stats only."
    )

    # ---- Parameter detail ----
    st.subheader("Parameter Values")
    param_data = {col_label(p): row[p] for p in param_cols if p in row}
    st.json(param_data)

    # ---- Full stats ----
    with st.expander("Full Statistics"):
        all_stats = {col_label(c): row[c] for c in stat_cols if c in row}
        st.json({k: float(v) if isinstance(v, (int, float)) else v for k, v in all_stats.items()})
