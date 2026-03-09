"""Execution page — order proposals, open positions, kill switch."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from trade_analysis.dashboard.components.metrics import render_metric_row
from trade_analysis.dashboard.data_loader import (
    get_kill_switch_status,
    get_supabase_client,
    load_open_positions,
    load_pending_proposals,
    load_recent_orders,
)


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the execution management page."""
    st.header("Execution")

    sb = get_supabase_client()
    if not sb.enabled:
        st.warning(
            "Supabase is not configured. Execution features require a database. "
            "Set SUPABASE_URL and SUPABASE_KEY in your .env file."
        )
        return

    # ---- Execution mode + kill switch ----
    kill_switch_on = get_kill_switch_status()

    col_mode, col_ks = st.columns(2)
    with col_mode:
        st.markdown("### Execution Mode")
        st.info("DRY RUN — orders are simulated, not sent to Schwab.")

    with col_ks:
        st.markdown("### Kill Switch")
        if kill_switch_on:
            st.error("ENGAGED — all new proposals and executions are blocked.")
        else:
            st.success("Disengaged — proposals and executions are allowed.")

    st.markdown("---")

    # ---- Pending proposals ----
    st.markdown("### Pending Proposals")
    proposals_df = load_pending_proposals()

    if proposals_df.empty:
        st.info("No pending proposals. Proposals are created when tradeable signals fire.")
    else:
        proposal_cols = [
            c for c in [
                "id", "created_at", "symbol", "direction", "entry_price",
                "stop_loss", "target_price", "rr_ratio", "signal_score",
                "regime", "status",
            ]
            if c in proposals_df.columns
        ]
        display_df = proposals_df[proposal_cols].copy() if proposal_cols else proposals_df.copy()

        if "created_at" in display_df.columns:
            display_df["created_at"] = pd.to_datetime(
                display_df["created_at"]
            ).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.caption(
            "To approve/reject proposals, use the API endpoints "
            "(POST /proposals/{id}/approve or /reject) or wait for "
            "dashboard form support."
        )

    st.markdown("---")

    # ---- Open positions ----
    st.markdown("### Open Positions")
    positions_df = load_open_positions()

    if positions_df.empty:
        st.info("No open positions.")
    else:
        pos_cols = [
            c for c in [
                "id", "symbol", "direction", "qty", "avg_entry_price",
                "current_stop", "current_target", "opened_at",
            ]
            if c in positions_df.columns
        ]
        display_pos = positions_df[pos_cols].copy() if pos_cols else positions_df.copy()

        if "opened_at" in display_pos.columns:
            display_pos["opened_at"] = pd.to_datetime(
                display_pos["opened_at"]
            ).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(display_pos, use_container_width=True, hide_index=True)

    # ---- Summary metrics ----
    render_metric_row({
        "Pending Proposals": (len(proposals_df), None),
        "Open Positions": (len(positions_df), None),
        "Kill Switch": ("ENGAGED" if kill_switch_on else "OFF", None),
    })

    st.markdown("---")

    # ---- Recent orders ----
    st.markdown("### Recent Orders")
    orders_df = load_recent_orders(limit=25)

    if orders_df.empty:
        st.info("No orders placed yet.")
    else:
        order_cols = [
            c for c in [
                "id", "placed_at", "symbol", "direction", "qty",
                "order_type", "status", "dry_run",
            ]
            if c in orders_df.columns
        ]
        display_orders = orders_df[order_cols].copy() if order_cols else orders_df.copy()

        if "placed_at" in display_orders.columns:
            display_orders["placed_at"] = pd.to_datetime(
                display_orders["placed_at"]
            ).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(display_orders, use_container_width=True, hide_index=True)
