"""Grid Results page — ranked table + parameter heatmaps."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_analysis.dashboard.components.charts import (
    build_param_heatmap,
    build_radar_comparison,
    build_single_param_bar,
)

# Columns that are stats (not swept parameters)
_STAT_COLS = {
    "id", "run_id", "created_at", "total_trades", "win_rate", "avg_r",
    "total_r", "profit_factor", "max_drawdown_r", "max_consecutive_wins",
    "max_consecutive_losses", "avg_duration_bars", "avg_duration_days",
    "longest_trade_bars", "shortest_trade_bars", "sufficient_trades",
    "is_robust", "is_isolated_peak", "by_regime", "by_direction",
    "by_signal_score", "by_exit_reason", "rank",
}

_DISPLAY_STATS = [
    "total_trades", "win_rate", "avg_r", "total_r",
    "profit_factor", "max_drawdown_r",
]


def _detect_param_cols(df: pd.DataFrame) -> list[str]:
    """Identify swept parameter columns (everything that's not a stat)."""
    return [c for c in df.columns if c not in _STAT_COLS]


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the grid results page."""
    st.header("Grid Results")

    if grid_df.empty:
        st.info("No grid results loaded.")
        return

    param_cols = _detect_param_cols(grid_df)
    stat_cols = [c for c in _DISPLAY_STATS if c in grid_df.columns]

    # ---- Metric selector ----
    metric = st.selectbox(
        "Rank by metric",
        options=stat_cols,
        index=stat_cols.index("total_r") if "total_r" in stat_cols else 0,
        key="grid_metric",
    )

    # ---- Ranked table ----
    st.subheader("Ranked Results")
    display_cols = param_cols + stat_cols
    ranked = grid_df[display_cols].sort_values(metric, ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "Rank"

    st.dataframe(ranked, use_container_width=True)

    csv = ranked.to_csv()
    st.download_button("Download CSV", csv, "grid_results.csv", "text/csv")

    # ---- Parameter heatmaps ----
    if param_cols:
        st.subheader("Parameter Heatmaps")

        if len(param_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                param_x = st.selectbox("X-axis parameter", param_cols, index=0, key="hm_x")
            with col2:
                other = [p for p in param_cols if p != param_x]
                param_y = st.selectbox("Y-axis parameter", other, index=0, key="hm_y")

            fig = build_param_heatmap(grid_df, param_x, param_y, metric)
            st.plotly_chart(fig, use_container_width=True)
        elif len(param_cols) == 1:
            fig = build_single_param_bar(grid_df, param_cols[0], metric)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Radar comparison ----
    if len(grid_df) >= 2 and param_cols:
        st.subheader("Strategy Comparison")
        st.markdown("Select rows by rank number to compare side-by-side.")

        max_rank = min(len(ranked), 20)
        selected_ranks = st.multiselect(
            "Select ranks to compare (2-5)",
            options=list(range(1, max_rank + 1)),
            default=[1, 2] if max_rank >= 2 else [1],
            key="radar_select",
        )

        if 2 <= len(selected_ranks) <= 5:
            selected_rows = []
            for rank in selected_ranks:
                row = ranked.iloc[rank - 1]
                row_dict = row.to_dict()
                label_parts = [f"{p}={row[p]}" for p in param_cols if p in row]
                row_dict["_label"] = ", ".join(label_parts)
                selected_rows.append(row_dict)

            radar_metrics = [s for s in stat_cols if s != "total_trades"]
            fig = build_radar_comparison(selected_rows, radar_metrics)
            st.plotly_chart(fig, use_container_width=True)
        elif selected_ranks:
            st.caption("Select 2-5 ranks to enable comparison.")
