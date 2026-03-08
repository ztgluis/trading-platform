"""Robustness page — parameter stability analysis."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the robustness page."""
    st.header("Robustness")
    st.info("Coming soon — robustness zone analysis and stability charts.")
