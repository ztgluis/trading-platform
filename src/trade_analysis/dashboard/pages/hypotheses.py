"""Hypotheses page — H1-H5 verdict cards + evidence."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the hypotheses page."""
    st.header("Hypotheses")
    st.info("Coming soon — hypothesis verdict cards and evidence display.")
