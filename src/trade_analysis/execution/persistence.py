"""Execution persistence — CRUD for proposals, orders, fills, positions.

Follows the same SupabaseClient patterns as live/persistence.py.
All operations gracefully no-op when Supabase is disabled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from trade_analysis.analyzer.persistence import SupabaseClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order Proposals
# ---------------------------------------------------------------------------


def create_proposal(
    sb: SupabaseClient,
    proposal: dict[str, Any],
) -> int | None:
    """Insert a new order proposal. Returns the proposal id or None."""
    if not sb.enabled:
        logger.info("Supabase disabled — skipping proposal persistence.")
        return None

    response = sb.client.table("order_proposals").insert(proposal).execute()

    if response.data:
        pid = response.data[0].get("id")
        logger.info(
            "Created proposal %s %s %s (id=%s)",
            proposal.get("symbol"),
            proposal.get("direction"),
            proposal.get("entry_price"),
            pid,
        )
        return pid

    return None


def update_proposal_status(
    sb: SupabaseClient,
    proposal_id: int,
    status: str,
    decided_at: datetime | None = None,
) -> bool:
    """Update a proposal's status. Returns True if updated."""
    if not sb.enabled:
        return False

    update = {"status": status}
    if decided_at is not None:
        update["decided_at"] = decided_at.isoformat()

    response = (
        sb.client.table("order_proposals")
        .update(update)
        .eq("id", proposal_id)
        .execute()
    )
    return bool(response.data)


def load_pending_proposals(
    sb: SupabaseClient,
    symbol: str | None = None,
) -> list[dict[str, Any]]:
    """Load proposals awaiting user approval."""
    if not sb.enabled:
        return []

    query = (
        sb.client.table("order_proposals")
        .select("*")
        .eq("status", "pending_approval")
        .order("created_at", desc=True)
    )

    if symbol:
        query = query.eq("symbol", symbol)

    response = query.execute()
    return response.data if response.data else []


def load_proposals(
    sb: SupabaseClient,
    limit: int = 50,
    status_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load proposals with optional status filter."""
    if not sb.enabled:
        return []

    query = (
        sb.client.table("order_proposals")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
    )

    if status_filter:
        query = query.eq("status", status_filter)

    response = query.execute()
    return response.data if response.data else []


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def create_order(
    sb: SupabaseClient,
    order: dict[str, Any],
) -> int | None:
    """Insert a new order record. Returns the order id or None."""
    if not sb.enabled:
        logger.info("Supabase disabled — skipping order persistence.")
        return None

    response = sb.client.table("orders").insert(order).execute()

    if response.data:
        oid = response.data[0].get("id")
        logger.info(
            "Created order %s %s qty=%s (id=%s)",
            order.get("symbol"),
            order.get("direction"),
            order.get("qty"),
            oid,
        )
        return oid

    return None


def update_order_status(
    sb: SupabaseClient,
    order_id: int,
    status: str,
) -> bool:
    """Update an order's status. Returns True if updated."""
    if not sb.enabled:
        return False

    response = (
        sb.client.table("orders")
        .update({"status": status, "updated_at": datetime.now(tz=timezone.utc).isoformat()})
        .eq("id", order_id)
        .execute()
    )
    return bool(response.data)


def load_recent_orders(
    sb: SupabaseClient,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Load recent orders ordered by placed_at DESC."""
    if not sb.enabled:
        return []

    response = (
        sb.client.table("orders")
        .select("*")
        .order("placed_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data if response.data else []


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------


def create_fill(
    sb: SupabaseClient,
    fill: dict[str, Any],
) -> int | None:
    """Insert a fill record. Returns the fill id or None."""
    if not sb.enabled:
        return None

    response = sb.client.table("fills").insert(fill).execute()

    if response.data:
        fid = response.data[0].get("id")
        logger.info(
            "Recorded fill: order=%s price=%s qty=%s (id=%s)",
            fill.get("order_id"),
            fill.get("fill_price"),
            fill.get("fill_qty"),
            fid,
        )
        return fid

    return None


# ---------------------------------------------------------------------------
# Live Positions
# ---------------------------------------------------------------------------


def upsert_position(
    sb: SupabaseClient,
    position: dict[str, Any],
) -> int | None:
    """Insert or update a live position. Returns position id or None."""
    if not sb.enabled:
        return None

    response = sb.client.table("live_positions").insert(position).execute()

    if response.data:
        return response.data[0].get("id")

    return None


def close_position(
    sb: SupabaseClient,
    position_id: int,
    close_reason: str,
) -> bool:
    """Mark a position as closed. Returns True if updated."""
    if not sb.enabled:
        return False

    response = (
        sb.client.table("live_positions")
        .update({
            "closed_at": datetime.now(tz=timezone.utc).isoformat(),
            "close_reason": close_reason,
        })
        .eq("id", position_id)
        .execute()
    )
    return bool(response.data)


def load_open_positions(
    sb: SupabaseClient,
) -> list[dict[str, Any]]:
    """Load all open (unclosed) positions."""
    if not sb.enabled:
        return []

    response = (
        sb.client.table("live_positions")
        .select("*")
        .is_("closed_at", "null")
        .order("opened_at", desc=True)
        .execute()
    )
    return response.data if response.data else []


# ---------------------------------------------------------------------------
# Kill Switch
# ---------------------------------------------------------------------------


def get_kill_switch_status(sb: SupabaseClient) -> bool:
    """Get the current kill switch status. Returns True if engaged."""
    if not sb.enabled:
        return False

    response = (
        sb.client.table("kill_switch")
        .select("enabled")
        .order("toggled_at", desc=True)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0].get("enabled", False)

    return False


def set_kill_switch(
    sb: SupabaseClient,
    enabled: bool,
    reason: str = "",
    toggled_by: str = "user",
) -> bool:
    """Toggle the kill switch by inserting a new state row. Returns True if saved."""
    if not sb.enabled:
        return False

    response = (
        sb.client.table("kill_switch")
        .insert({
            "enabled": enabled,
            "reason": reason,
            "toggled_by": toggled_by,
            "toggled_at": datetime.now(tz=timezone.utc).isoformat(),
        })
        .execute()
    )
    return bool(response.data)
