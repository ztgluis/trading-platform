"""Central display-label mapping for column names.

Import ``col_label`` anywhere in the dashboard to convert raw DataFrame
column names (e.g. ``"total_r"``) into human-readable labels
(e.g. ``"Total R"``).
"""

from __future__ import annotations

# Raw column name → human-readable label
COLUMN_LABELS: dict[str, str] = {
    # Swept parameters
    "rsi_period": "RSI Period",
    "trend_ma_period": "Trend MA Period",
    # Performance metrics
    "total_r": "Total R-Multiple",
    "avg_r": "Avg R-Multiple",
    "win_rate": "Win Rate",
    "profit_factor": "Profit Factor",
    "total_trades": "Total Trades",
    "max_drawdown_r": "Max Drawdown (R)",
    "max_consecutive_wins": "Max Consec. Wins",
    "max_consecutive_losses": "Max Consec. Losses",
    "avg_duration_bars": "Avg Duration (Bars)",
    "avg_duration_days": "Avg Duration (Days)",
    "longest_trade_bars": "Longest Trade (Bars)",
    "shortest_trade_bars": "Shortest Trade (Bars)",
    "sufficient_trades": "Sufficient Trades",
}


def col_label(raw: str) -> str:
    """Return the display label for *raw*, falling back to title-case."""
    return COLUMN_LABELS.get(raw, raw.replace("_", " ").title())
