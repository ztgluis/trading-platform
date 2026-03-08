"""Signal persistence — insert/upsert signals to Supabase.

Follows the same patterns as analyzer/persistence.py.
"""

from __future__ import annotations

import logging
from typing import Any

from trade_analysis.analyzer.persistence import SupabaseClient

logger = logging.getLogger(__name__)


def persist_signal(
    sb: SupabaseClient,
    signal_row: dict[str, Any],
) -> int | None:
    """Insert a signal row to the signals table.

    Uses upsert on the UNIQUE(symbol, timeframe, bar_timestamp, signal_direction)
    constraint to avoid duplicates — if the same signal fires again, the row
    is updated rather than creating a duplicate.

    Returns the signals.id if persisted, None if skipped.
    """
    if not sb.enabled:
        logger.info("Supabase disabled — skipping signal persistence.")
        return None

    response = (
        sb.client.table("signals")
        .upsert(
            signal_row,
            on_conflict="symbol,timeframe,bar_timestamp,signal_direction",
        )
        .execute()
    )

    if response.data:
        signal_id = response.data[0].get("id")
        logger.info(
            "Persisted signal %s %s %s (id=%s)",
            signal_row.get("symbol"),
            signal_row.get("signal_direction"),
            signal_row.get("bar_timestamp"),
            signal_id,
        )
        return signal_id

    return None


def load_recent_signals(
    sb: SupabaseClient,
    limit: int = 50,
    symbol: str | None = None,
    tradeable_only: bool = False,
) -> list[dict[str, Any]]:
    """Load recent signals from the signals table.

    Args:
        sb: SupabaseClient instance.
        limit: Maximum number of signals to return.
        symbol: Optional filter by symbol.
        tradeable_only: If True, only return tradeable signals.

    Returns:
        List of signal dicts ordered by created_at DESC.
    """
    if not sb.enabled:
        return []

    query = sb.client.table("signals").select("*").order("created_at", desc=True).limit(limit)

    if symbol:
        query = query.eq("symbol", symbol)
    if tradeable_only:
        query = query.eq("signal_tradeable", True)

    response = query.execute()
    return response.data if response.data else []
