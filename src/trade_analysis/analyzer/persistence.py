"""Supabase persistence for grid results and hypothesis evaluations.

Gracefully skips if SUPABASE_URL / SUPABASE_KEY env vars are not set.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from trade_analysis.analyzer.hypothesis import HypothesisResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SupabaseClient:
    """Thin wrapper around supabase-py for persisting analysis results.

    Initialises from SUPABASE_URL and SUPABASE_KEY environment variables.
    If credentials are missing, the client is created in *disabled* mode
    and all write operations become no-ops.
    """

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
    ) -> None:
        self._url = url or os.environ.get("SUPABASE_URL", "")
        self._key = key or os.environ.get("SUPABASE_KEY", "")
        self._client: Any = None
        self._enabled = False

        if self._url and self._key:
            try:
                from supabase import create_client

                self._client = create_client(self._url, self._key)
                self._enabled = True
                logger.info("Supabase client connected to %s", self._url)
            except ImportError:
                logger.warning(
                    "supabase-py not installed — persistence disabled. "
                    "Install with: pip install supabase"
                )
            except Exception as exc:
                logger.warning("Supabase connection failed: %s", exc)
        else:
            logger.info("Supabase credentials not configured — persistence disabled.")

    @property
    def enabled(self) -> bool:
        """Whether the client is connected and ready."""
        return self._enabled

    @property
    def client(self) -> Any:
        """Raw supabase client (None if disabled)."""
        return self._client


# ---------------------------------------------------------------------------
# Persist grid run + results
# ---------------------------------------------------------------------------


def persist_grid_run(
    sb: SupabaseClient,
    grid_config: Any,
    grid_result: Any,
) -> int | None:
    """Insert a grid_runs row and its grid_results rows.

    Args:
        sb: SupabaseClient instance.
        grid_config: GridConfig with symbol, asset_class, timeframe, parameters.
        grid_result: GridResult with rows and total_combos.

    Returns:
        The grid_runs.id if persisted, None if skipped.
    """
    if not sb.enabled:
        logger.info("Supabase disabled — skipping grid run persistence.")
        return None

    # Insert run metadata
    run_data = {
        "symbol": grid_config.symbol,
        "asset_class": grid_config.asset_class,
        "timeframe": grid_config.timeframe,
        "parameters": grid_config.parameters,
        "min_trades": grid_config.min_trades,
        "rank_by": grid_config.rank_by,
        "total_combos": grid_result.total_combos,
        "sufficient_combos": grid_result.sufficient_combos,
    }

    response = sb.client.table("grid_runs").insert(run_data).execute()
    run_id = response.data[0]["id"]

    # Insert individual results
    df = grid_result.to_dataframe()
    param_cols = [
        c for c in df.columns
        if c not in {
            "total_trades", "win_rate", "avg_r", "total_r",
            "profit_factor", "max_drawdown_r", "sufficient_trades",
        }
    ]

    result_rows = []
    for _, row in df.iterrows():
        params = {col: row[col] for col in param_cols if col in row}
        result_rows.append({
            "run_id": run_id,
            "params": params,
            "total_trades": int(row.get("total_trades", 0)),
            "win_rate": float(row["win_rate"]) if "win_rate" in row else None,
            "avg_r": float(row["avg_r"]) if "avg_r" in row else None,
            "total_r": float(row["total_r"]) if "total_r" in row else None,
            "profit_factor": (
                float(row["profit_factor"]) if "profit_factor" in row else None
            ),
            "max_drawdown_r": (
                float(row["max_drawdown_r"]) if "max_drawdown_r" in row else None
            ),
            "sufficient_trades": bool(row.get("sufficient_trades", False)),
        })

    if result_rows:
        sb.client.table("grid_results").insert(result_rows).execute()

    logger.info("Persisted grid run %d with %d results.", run_id, len(result_rows))
    return run_id


# ---------------------------------------------------------------------------
# Persist hypothesis results
# ---------------------------------------------------------------------------


def persist_hypothesis_results(
    sb: SupabaseClient,
    results: list[HypothesisResult],
    grid_run_id: int | None = None,
) -> list[int]:
    """Insert hypothesis evaluation results.

    Args:
        sb: SupabaseClient instance.
        results: List of HypothesisResult objects.
        grid_run_id: Optional grid_runs.id to link results to.

    Returns:
        List of inserted hypothesis_results.id values, empty if skipped.
    """
    if not sb.enabled:
        logger.info("Supabase disabled — skipping hypothesis persistence.")
        return []

    rows = []
    for r in results:
        row: dict[str, Any] = {
            "hypothesis_id": r.hypothesis_id,
            "question": r.question,
            "verdict": r.verdict,
            "evidence": r.evidence,
            "summary": r.summary,
        }
        if grid_run_id is not None:
            row["grid_run_id"] = grid_run_id
        rows.append(row)

    if not rows:
        return []

    response = sb.client.table("hypothesis_results").insert(rows).execute()
    ids = [row["id"] for row in response.data]
    logger.info("Persisted %d hypothesis results.", len(ids))
    return ids
