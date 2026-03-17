"""Overview page — high-level summary of a grid run."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_analysis.dashboard.components.charts import (
    build_distribution_histogram,
)
from trade_analysis.dashboard.components.labels import col_label
from trade_analysis.dashboard.components.metrics import render_metric_row


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the overview page."""
    st.header("Overview")

    if grid_df.empty:
        st.info("No grid results loaded. Select a run or run a fresh sweep.")
        return

    # ---- Key metrics ----
    sufficient = grid_df[grid_df["sufficient_trades"] == True] if "sufficient_trades" in grid_df.columns else grid_df  # noqa: E712
    total_combos = len(grid_df)
    sufficient_combos = len(sufficient)

    best_total_r = grid_df["total_r"].max() if "total_r" in grid_df.columns else 0
    best_win_rate = grid_df["win_rate"].max() if "win_rate" in grid_df.columns else 0

    render_metric_row({
        "Total Combinations": (total_combos, None),
        "Sufficient Trades": (sufficient_combos, f"{sufficient_combos / total_combos:.0%} of total" if total_combos else None),
        "Best Total R-Multiple": (best_total_r, None),
        "Best Win Rate": (best_win_rate, None),
    })

    # ---- Run metadata ----
    if run_meta is not None:
        with st.expander("Run Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Symbol:** {run_meta.get('symbol', 'N/A')}")
                st.markdown(f"**Asset Class:** {run_meta.get('asset_class', 'N/A')}")
                st.markdown(f"**Timeframe:** {run_meta.get('timeframe', 'N/A')}")
            with col2:
                st.markdown(f"**Min Trades:** {run_meta.get('min_trades', 30)}")
                st.markdown(f"**Rank By:** {col_label(run_meta.get('rank_by', 'total_r'))}")
                created = run_meta.get("created_at", "")
                if created:
                    st.markdown(f"**Created:** {str(created)[:19]}")

            params = run_meta.get("parameters", {})
            if params:
                st.markdown("**Swept Parameters:**")
                for name, values in params.items():
                    st.markdown(f"- **{col_label(name)}**: {values}")

    # ---- Distribution charts ----
    if not sufficient.empty and "total_r" in sufficient.columns:
        st.subheader("Distributions (Sufficient-Trade Combos)")
        col1, col2 = st.columns(2)
        with col1:
            fig = build_distribution_histogram(sufficient, "total_r", "Total R Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if "win_rate" in sufficient.columns:
                fig = build_distribution_histogram(sufficient, "win_rate", "Win Rate Distribution")
                st.plotly_chart(fig, use_container_width=True)

    # ---- Top 5 results ----
    if "total_r" in grid_df.columns:
        st.subheader("Top 5 Results")
        display_cols = [
            c for c in grid_df.columns
            if c not in {"id", "run_id", "created_at", "is_robust", "is_isolated_peak"}
        ]
        top5 = grid_df.nlargest(5, "total_r")[display_cols]
        st.dataframe(
            top5.rename(columns=col_label),
            use_container_width=True,
            hide_index=True,
        )
