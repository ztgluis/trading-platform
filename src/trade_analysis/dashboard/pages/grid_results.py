"""Grid Results page — ranked table + parameter heatmaps."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the grid results page."""
    st.header("Grid Results")
    st.info("Coming soon — ranked table and parameter heatmaps.")
