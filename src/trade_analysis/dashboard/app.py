"""Trade Analysis Dashboard — Streamlit entry point.

Run with: streamlit run src/trade_analysis/dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Trade Analysis Dashboard",
    layout="wide",
)

from trade_analysis.dashboard.components.filters import FilterState, render_sidebar_filters
from trade_analysis.dashboard.data_loader import (
    get_supabase_client,
    load_grid_results,
    load_grid_runs,
)
from trade_analysis.dashboard.pages import (
    grid_results,
    hypotheses,
    overview,
    robustness,
    trades,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Trade Analysis")

    sb = get_supabase_client()
    if sb.enabled:
        st.success("Supabase connected", icon=None)
    else:
        st.warning("In-memory mode")

    # ---- Run selector ----
    runs_df = load_grid_runs()
    selected_run_id: int | None = None
    run_meta: pd.Series | None = None

    # Check session state for fresh sweep results
    has_fresh = "fresh_grid_df" in st.session_state and not st.session_state["fresh_grid_df"].empty

    if not runs_df.empty:
        run_options = {}
        for _, row in runs_df.iterrows():
            label = f"{row['symbol']} {row['asset_class']} ({str(row['created_at'])[:10]})"
            run_options[row["id"]] = label

        selected_run_id = st.selectbox(
            "Select Grid Run",
            options=list(run_options.keys()),
            format_func=lambda x: run_options[x],
        )
    elif not has_fresh:
        st.info("No grid runs found. Use the Fresh Sweep page to run one.")

    # ---- Filters ----
    filters = render_sidebar_filters()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

grid_df = pd.DataFrame()

if has_fresh and st.sidebar.checkbox("Use fresh sweep results", value=True):
    grid_df = st.session_state["fresh_grid_df"]
    run_meta = st.session_state.get("fresh_run_meta")
elif selected_run_id is not None:
    grid_df = load_grid_results(selected_run_id)
    if not runs_df.empty:
        run_meta = runs_df[runs_df["id"] == selected_run_id].iloc[0]

# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": overview,
    "Grid Results": grid_results,
    "Hypotheses": hypotheses,
    "Robustness": robustness,
    "Trades": trades,
}

page_name = st.sidebar.radio("Page", list(PAGES.keys()))
page_module = PAGES[page_name]
page_module.render(grid_df, run_meta)
