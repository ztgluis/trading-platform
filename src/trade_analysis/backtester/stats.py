"""Backtest summary statistics: win rate, profit factor, drawdown, breakdowns.

Pure functions operating on BacktestResult / trade lists.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

from trade_analysis.backtester.models import BacktestResult, Trade


def compute_backtest_stats(result: BacktestResult) -> dict:
    """Compute comprehensive summary statistics from a backtest result.

    Args:
        result: BacktestResult containing trade list.

    Returns:
        Dict with keys:
            total_trades, win_rate, avg_r, total_r, profit_factor,
            max_consecutive_wins, max_consecutive_losses, max_drawdown_r,
            avg_duration_bars, avg_duration_days, longest_trade_bars,
            shortest_trade_bars,
            by_regime, by_direction, by_signal_score, by_exit_reason,
            sufficient_trades (bool — >= 30 trades).
    """
    trades = result.trades
    base = _compute_sub_stats(trades)

    # Breakdowns
    base["by_regime"] = _breakdown(trades, lambda t: t.entry_regime)
    base["by_direction"] = _breakdown(trades, lambda t: t.direction)
    base["by_signal_score"] = _breakdown(
        trades, lambda t: str(t.entry_signal_score)
    )
    base["by_exit_reason"] = _breakdown(trades, lambda t: t.exit_reason)
    base["sufficient_trades"] = len(trades) >= 30

    return base


def _compute_sub_stats(trades: list[Trade]) -> dict:
    """Compute stats for a subset of trades."""
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "max_drawdown_r": 0.0,
            "avg_duration_bars": 0.0,
            "avg_duration_days": 0.0,
            "longest_trade_bars": 0,
            "shortest_trade_bars": 0,
        }

    winners = [t for t in trades if t.is_winner]
    r_values = [t.pnl_r for t in trades]
    durations_bars = [t.duration_bars for t in trades]
    durations_days = [t.duration_calendar_days for t in trades]

    return {
        "total_trades": n,
        "win_rate": len(winners) / n,
        "avg_r": sum(r_values) / n,
        "total_r": sum(r_values),
        "profit_factor": _compute_profit_factor(trades),
        "max_consecutive_wins": _max_consecutive(trades, win=True),
        "max_consecutive_losses": _max_consecutive(trades, win=False),
        "max_drawdown_r": _compute_max_drawdown(trades),
        "avg_duration_bars": sum(durations_bars) / n,
        "avg_duration_days": sum(durations_days) / n,
        "longest_trade_bars": max(durations_bars),
        "shortest_trade_bars": min(durations_bars),
    }


def _compute_profit_factor(trades: list[Trade]) -> float:
    """Gross winning R / gross losing R.

    Returns inf if no losers, 0.0 if no winners.
    """
    gross_win = sum(t.pnl_r for t in trades if t.pnl_r > 0)
    gross_loss = abs(sum(t.pnl_r for t in trades if t.pnl_r < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _compute_max_drawdown(trades: list[Trade]) -> float:
    """Max peak-to-trough drawdown in cumulative R-multiples."""
    if not trades:
        return 0.0

    cum_r = list(itertools.accumulate(t.pnl_r for t in trades))
    peak = cum_r[0]
    max_dd = 0.0
    for r in cum_r:
        peak = max(peak, r)
        dd = peak - r
        max_dd = max(max_dd, dd)
    return max_dd


def _max_consecutive(trades: list[Trade], win: bool) -> int:
    """Max consecutive wins or losses."""
    max_streak = 0
    current = 0
    for t in trades:
        if t.is_winner == win:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def _breakdown(
    trades: list[Trade], key_fn: callable
) -> dict[str, dict]:
    """Group trades by a key function and compute sub-stats for each group."""
    groups: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)

    return {k: _compute_sub_stats(v) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def format_stats_report(stats: dict) -> str:
    """Format stats dict as a human-readable text report."""
    lines = [
        "=" * 50,
        "BACKTEST SUMMARY",
        "=" * 50,
        f"Total Trades:       {stats['total_trades']}",
        f"Win Rate:           {stats['win_rate']:.1%}",
        f"Avg R:              {stats['avg_r']:+.2f}",
        f"Total R:            {stats['total_r']:+.2f}",
        f"Profit Factor:      {stats['profit_factor']:.2f}",
        f"Max Drawdown (R):   {stats['max_drawdown_r']:.2f}",
        f"Max Consec Wins:    {stats['max_consecutive_wins']}",
        f"Max Consec Losses:  {stats['max_consecutive_losses']}",
        f"Avg Duration (bars):{stats['avg_duration_bars']:.1f}",
        f"Avg Duration (days):{stats['avg_duration_days']:.1f}",
        f"Sufficient Trades:  {'Yes' if stats.get('sufficient_trades') else 'No'}",
        "",
    ]

    # Breakdowns
    for section_name, section_key in [
        ("BY DIRECTION", "by_direction"),
        ("BY EXIT REASON", "by_exit_reason"),
        ("BY REGIME", "by_regime"),
    ]:
        if section_key in stats and stats[section_key]:
            lines.append(f"--- {section_name} ---")
            for key, sub in stats[section_key].items():
                lines.append(
                    f"  {key:12s}  n={sub['total_trades']:3d}  "
                    f"WR={sub['win_rate']:.0%}  "
                    f"avgR={sub['avg_r']:+.2f}"
                )
            lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)
