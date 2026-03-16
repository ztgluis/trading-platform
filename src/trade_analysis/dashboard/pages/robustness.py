"""Robustness page — parameter stability analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_analysis.dashboard.components.charts import build_robustness_chart
from trade_analysis.dashboard.components.labels import col_label
from trade_analysis.dashboard.pages.grid_results import _detect_param_cols


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the robustness page."""
    st.header("Robustness Analysis")

    if grid_df.empty:
        st.info("No grid results loaded.")
        return

    param_cols = _detect_param_cols(grid_df)
    if not param_cols:
        st.warning("No swept parameters detected in the results.")
        return

    # ---- Selectors ----
    col1, col2 = st.columns(2)
    with col1:
        selected_param = st.selectbox(
            "Parameter", param_cols, format_func=col_label, key="robust_param",
        )
    with col2:
        metrics = ["total_r", "avg_r", "profit_factor", "win_rate"]
        available = [m for m in metrics if m in grid_df.columns]
        selected_metric = st.selectbox(
            "Metric", available, format_func=col_label, key="robust_metric",
        )

    # ---- Run robustness analysis ----
    from trade_analysis.grid import analyze_robustness, find_robust_zones

    # Filter to sufficient trades if column exists
    analysis_df = grid_df.copy()
    if "sufficient_trades" in analysis_df.columns:
        sufficient = analysis_df[analysis_df["sufficient_trades"] == True]  # noqa: E712
        if not sufficient.empty:
            analysis_df = sufficient

    try:
        robustness_df = analyze_robustness(analysis_df, selected_param, selected_metric)
    except Exception as exc:
        st.error(f"Robustness analysis failed: {exc}")
        return

    # ---- Robustness chart ----
    st.subheader(f"{col_label(selected_metric)} by {col_label(selected_param)}")
    fig = build_robustness_chart(robustness_df, selected_param, selected_metric)
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown(
        "🟢 **Robust** — within tolerance of neighbors &nbsp;&nbsp; "
        "🔴 **Isolated peak** — significantly outperforms neighbors (potential curve-fit)"
    )

    # ---- Robustness detail table ----
    st.subheader("Detail")
    display_df = robustness_df[
        [c for c in robustness_df.columns if c != "index"]
    ].copy()
    st.dataframe(display_df.rename(columns=col_label), use_container_width=True, hide_index=True)

    # ---- Robust zones (all parameters) ----
    st.subheader("Robust Zones (All Parameters)")
    try:
        zones = find_robust_zones(analysis_df, selected_metric)
    except Exception:
        st.warning("Could not compute robust zones.")
        return

    if not zones:
        st.info("No robust zones found.")
        return

    for param_name, param_zones in zones.items():
        lp = col_label(param_name)
        if not param_zones:
            st.markdown(f"**{lp}**: No robust zones")
            continue
        for zone in param_zones:
            values = zone.get("values", [])
            avg = zone.get("avg_metric", 0)
            st.markdown(
                f"**{lp}**: values `{values}` — "
                f"avg {col_label(selected_metric)} = {avg:.3f}"
            )
