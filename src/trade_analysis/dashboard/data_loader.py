"""Data loading layer for the dashboard.

Provides cached data loading from Supabase with in-memory fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

from trade_analysis.analyzer.persistence import SupabaseClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supabase client (singleton)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_supabase_client() -> SupabaseClient:
    """Singleton Supabase client cached across reruns."""
    return SupabaseClient()


# ---------------------------------------------------------------------------
# Grid runs
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_grid_runs() -> pd.DataFrame:
    """Load all grid run metadata from Supabase.

    Returns DataFrame with columns: id, symbol, asset_class, timeframe,
    parameters, min_trades, rank_by, total_combos, sufficient_combos, created_at.
    Empty DataFrame if Supabase is disabled.
    """
    sb = get_supabase_client()
    if not sb.enabled:
        return pd.DataFrame()

    response = (
        sb.client.table("grid_runs")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()


@st.cache_data(ttl=300)
def load_grid_results(run_id: int) -> pd.DataFrame:
    """Load grid results for a specific run, expanding JSONB params.

    Returns DataFrame with flattened parameter columns + stat columns.
    """
    sb = get_supabase_client()
    if not sb.enabled:
        return pd.DataFrame()

    response = (
        sb.client.table("grid_results")
        .select("*")
        .eq("run_id", run_id)
        .execute()
    )
    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    return expand_params(df)


def expand_params(df: pd.DataFrame) -> pd.DataFrame:
    """Expand JSONB 'params' column into flat DataFrame columns.

    If 'params' column exists, its dict values are normalized into separate
    columns and the original 'params' column is dropped.
    """
    if df.empty or "params" not in df.columns:
        return df

    params_series = df["params"]
    # Handle None values in params
    params_dicts = [p if isinstance(p, dict) else {} for p in params_series]
    params_df = pd.DataFrame(params_dicts, index=df.index)
    return pd.concat([df.drop(columns=["params"]), params_df], axis=1)


# ---------------------------------------------------------------------------
# Hypothesis results
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_hypothesis_results(grid_run_id: int) -> list[dict[str, Any]]:
    """Load hypothesis results for a grid run.

    Returns list of dicts with hypothesis_id, question, verdict, evidence, summary.
    """
    sb = get_supabase_client()
    if not sb.enabled:
        return []

    response = (
        sb.client.table("hypothesis_results")
        .select("*")
        .eq("grid_run_id", grid_run_id)
        .execute()
    )
    return response.data if response.data else []


# ---------------------------------------------------------------------------
# Backtest trades
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_backtest_trades(run_id: int) -> pd.DataFrame:
    """Load individual trades for a backtest run."""
    sb = get_supabase_client()
    if not sb.enabled:
        return pd.DataFrame()

    response = (
        sb.client.table("backtest_trades")
        .select("*")
        .eq("run_id", run_id)
        .execute()
    )
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()


# ---------------------------------------------------------------------------
# Live signals
# ---------------------------------------------------------------------------


@st.cache_data(ttl=60)
def load_signals(
    limit: int = 50,
    symbol: str | None = None,
    tradeable_only: bool = False,
) -> pd.DataFrame:
    """Load recent signals from the signals table.

    Shorter TTL (60s) for near-real-time display.
    """
    sb = get_supabase_client()
    if not sb.enabled:
        return pd.DataFrame()

    query = sb.client.table("signals").select("*").order("created_at", desc=True).limit(limit)

    if symbol:
        query = query.eq("symbol", symbol)
    if tradeable_only:
        query = query.eq("signal_tradeable", True)

    response = query.execute()
    return pd.DataFrame(response.data) if response.data else pd.DataFrame()


# ---------------------------------------------------------------------------
# Fresh sweep (in-memory fallback)
# ---------------------------------------------------------------------------


def run_fresh_sweep(
    grid_config: Any,
    bt_config: Any,
    signal_config: Any,
    ohlcv_df: pd.DataFrame,
) -> Any:
    """Run a fresh grid sweep and optionally persist to Supabase.

    Returns a GridResult object.
    """
    from trade_analysis.analyzer.persistence import persist_grid_run
    from trade_analysis.grid import GridRunner

    runner = GridRunner(grid_config, bt_config, signal_config)
    result = runner.run(ohlcv_df)

    sb = get_supabase_client()
    if sb.enabled:
        persist_grid_run(sb, grid_config, result)

    return result
