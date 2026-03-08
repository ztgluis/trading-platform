"""Backtester — historical replay with trade logging and walk-forward support."""

from trade_analysis.backtester.config import (
    BacktestConfig,
    WalkForwardConfig,
    load_backtest_config,
)
from trade_analysis.backtester.engine import Backtester
from trade_analysis.backtester.models import (
    BacktestResult,
    Position,
    Trade,
    WalkForwardResult,
    WalkForwardSplit,
)
from trade_analysis.backtester.stats import compute_backtest_stats, format_stats_report
from trade_analysis.backtester.walk_forward import (
    generate_walk_forward_splits,
    run_walk_forward,
)

__all__ = [
    # Config
    "BacktestConfig",
    "WalkForwardConfig",
    "load_backtest_config",
    # Engine
    "Backtester",
    # Models
    "BacktestResult",
    "Position",
    "Trade",
    "WalkForwardResult",
    "WalkForwardSplit",
    # Stats
    "compute_backtest_stats",
    "format_stats_report",
    # Walk-forward
    "generate_walk_forward_splits",
    "run_walk_forward",
]
