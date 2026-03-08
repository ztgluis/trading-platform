"""Trades page — trade-level breakdown + equity curve."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the trades page."""
    st.header("Trades")
    st.info("Coming soon — trade breakdowns and equity curve.")
