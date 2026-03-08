"""Live Signals page — recent signals from the webhook runner."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_analysis.dashboard.components.metrics import (
    VERDICT_COLORS,
    render_metric_row,
)
from trade_analysis.dashboard.data_loader import load_signals


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the live signals page."""
    st.header("Live Signals")

    # ---- Filters ----
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol_filter = st.text_input("Symbol filter", value="", key="live_symbol")
    with col2:
        tradeable_only = st.checkbox("Tradeable only", value=True, key="live_tradeable")
    with col3:
        limit = st.selectbox("Show last", options=[25, 50, 100], index=1, key="live_limit")

    auto_refresh = st.checkbox("Auto-refresh (60s cache)", value=True, key="live_auto")
    if auto_refresh:
        st.caption("Data refreshes every 60 seconds from Supabase.")

    # ---- Load signals ----
    signals_df = load_signals(
        limit=limit,
        symbol=symbol_filter if symbol_filter else None,
        tradeable_only=tradeable_only,
    )

    if signals_df.empty:
        st.info(
            "No signals found. Make sure the webhook server is running "
            "and Supabase is configured."
        )
        return

    # ---- Summary metrics ----
    total = len(signals_df)
    tradeable_count = len(signals_df[signals_df["signal_tradeable"] == True]) if "signal_tradeable" in signals_df.columns else 0  # noqa: E712
    long_count = len(signals_df[signals_df["signal_direction"] == "long"]) if "signal_direction" in signals_df.columns else 0
    short_count = len(signals_df[signals_df["signal_direction"] == "short"]) if "signal_direction" in signals_df.columns else 0

    render_metric_row({
        "Total Signals": (total, None),
        "Tradeable": (tradeable_count, None),
        "Long": (long_count, None),
        "Short": (short_count, None),
    })

    st.markdown("---")

    # ---- Signals table ----
    display_cols = [
        c for c in [
            "created_at", "symbol", "timeframe", "signal_direction", "regime",
            "signal_score", "signal_tradeable", "entry_price", "exit_stop",
            "exit_target", "exit_rr_ratio", "conditions_met",
            "trend_confirmed", "structure_confirmed", "momentum_confirmed",
            "volume_spike",
        ]
        if c in signals_df.columns
    ]

    if display_cols:
        display_df = signals_df[display_cols].copy()

        # Format timestamp
        if "created_at" in display_df.columns:
            display_df["created_at"] = pd.to_datetime(display_df["created_at"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(signals_df, use_container_width=True, hide_index=True)

    # ---- Detail expander for latest signal ----
    if not signals_df.empty:
        latest = signals_df.iloc[0]
        with st.expander("Latest Signal Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Symbol:** {latest.get('symbol', 'N/A')}")
                st.markdown(f"**Timeframe:** {latest.get('timeframe', 'N/A')}")
                st.markdown(f"**Direction:** {latest.get('signal_direction', 'N/A')}")
                st.markdown(f"**Score:** {latest.get('signal_score', 0)}")
                st.markdown(f"**Regime:** {latest.get('regime', 'N/A')}")
            with col2:
                st.markdown(f"**Entry:** {latest.get('entry_price', 'N/A')}")
                st.markdown(f"**Stop:** {latest.get('exit_stop', 'N/A')}")
                st.markdown(f"**Target:** {latest.get('exit_target', 'N/A')}")
                st.markdown(f"**R:R:** {latest.get('exit_rr_ratio', 'N/A')}")
                st.markdown(f"**Tradeable:** {latest.get('signal_tradeable', False)}")
