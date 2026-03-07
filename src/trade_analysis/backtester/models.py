"""Backtest data models: Position, Trade, BacktestResult.

Position represents an open trade (mutable state).
Trade represents a closed trade (frozen record).
BacktestResult wraps a complete backtest run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trade_analysis.backtester.config import BacktestConfig


# ---------------------------------------------------------------------------
# Position (mutable — open trade)
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """An open trade being tracked by the backtester."""

    symbol: str
    asset_class: str
    timeframe: str
    bucket: str  # "A" or "B"
    direction: str  # "long" or "short"
    entry_bar_index: int
    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_signal_score: int
    entry_regime: str

    # Exit levels (set at entry)
    stop_loss: float  # original stop level
    target: float
    trail_breakeven_price: float  # price that triggers trail to breakeven
    max_hold_bars: int | None  # None = no limit

    # Mutable state
    stop_trailed_to_breakeven: bool = False
    current_stop: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_stop = self.stop_loss


# ---------------------------------------------------------------------------
# Trade (frozen — closed trade)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trade:
    """A closed trade record."""

    symbol: str
    asset_class: str
    timeframe: str
    bucket: str
    direction: str

    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_signal_score: int
    entry_regime: str

    exit_timestamp: pd.Timestamp
    exit_price: float
    exit_reason: str  # "stop" | "trail_stop" | "target" | "max_hold" | "end_of_data"

    pnl_r: float  # realized R-multiple
    pnl_dollar: float  # raw price difference (signed)
    duration_bars: int
    duration_calendar_days: int

    @property
    def is_winner(self) -> bool:
        """Trade was profitable (positive R-multiple)."""
        return self.pnl_r > 0


# ---------------------------------------------------------------------------
# Walk-Forward Split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardSplit:
    """A single walk-forward fold with in-sample and out-of-sample periods."""

    fold: int
    is_start: date  # in-sample start
    is_end: date  # in-sample end
    oos_start: date  # out-of-sample start
    oos_end: date  # out-of-sample end


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Complete result of a single backtest run."""

    trades: list[Trade]
    config: BacktestConfig
    signal_config_hash: str  # SHA-256 of signals.yaml for reproducibility
    symbol: str
    asset_class: str
    timeframe: str
    start_date: date
    end_date: date

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade list to a DataFrame (trade log).

        Returns:
            DataFrame with one row per trade. Empty DataFrame with correct
            schema if no trades.
        """
        columns = [
            "symbol",
            "asset_class",
            "timeframe",
            "bucket",
            "direction",
            "entry_timestamp",
            "entry_price",
            "entry_signal_score",
            "entry_regime",
            "exit_timestamp",
            "exit_price",
            "exit_reason",
            "pnl_r",
            "pnl_dollar",
            "duration_bars",
            "duration_calendar_days",
            "is_winner",
        ]

        if not self.trades:
            return pd.DataFrame(columns=columns)

        rows = []
        for t in self.trades:
            rows.append(
                {
                    "symbol": t.symbol,
                    "asset_class": t.asset_class,
                    "timeframe": t.timeframe,
                    "bucket": t.bucket,
                    "direction": t.direction,
                    "entry_timestamp": t.entry_timestamp,
                    "entry_price": t.entry_price,
                    "entry_signal_score": t.entry_signal_score,
                    "entry_regime": t.entry_regime,
                    "exit_timestamp": t.exit_timestamp,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl_r": t.pnl_r,
                    "pnl_dollar": t.pnl_dollar,
                    "duration_bars": t.duration_bars,
                    "duration_calendar_days": t.duration_calendar_days,
                    "is_winner": t.is_winner,
                }
            )

        return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# WalkForwardResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardResult:
    """Combined result of a walk-forward validation run."""

    splits: list[WalkForwardSplit]
    in_sample_results: list[BacktestResult]
    out_of_sample_results: list[BacktestResult]
