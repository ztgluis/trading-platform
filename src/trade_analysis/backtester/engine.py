"""Backtester engine: bar-by-bar replay with position state machine.

Processes signal-enriched OHLCV data one bar at a time, opening positions
on tradeable signals and closing them via stop/target/trail/max-hold rules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.backtester.models import BacktestResult, Position, Trade
from trade_analysis.signals.engine import (
    BucketConfig,
    SignalEngineConfig,
    get_bucket_for_asset,
)


# Approximate bars per week by timeframe
_BARS_PER_WEEK: dict[str, float] = {
    "1H": 5 * 6.5,   # 6.5 trading hours × 5 days (equities)
    "4H": 5 * 2,     # ~2 per day × 5 days (rough)
    "Daily": 5,
    "Weekly": 1,
    "Monthly": 0.25,
}


def _timeframe_to_bars_per_week(timeframe: str) -> float:
    """Convert a timeframe string to approximate bars per trading week."""
    return _BARS_PER_WEEK.get(timeframe, 5.0)


def _compute_config_hash(config_path: Path | None = None) -> str:
    """SHA-256 hash of signals.yaml for reproducibility tracking."""
    path = config_path or Path("config/signals.yaml")
    if path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return "unknown"


class Backtester:
    """Bar-by-bar replay engine with position state machine.

    Usage:
        bt = Backtester(backtest_config, signal_config)
        result = bt.run(enriched_df, "AAPL", "stock", "Daily")
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        signal_config: SignalEngineConfig,
    ) -> None:
        self._config = backtest_config
        self._signal_config = signal_config

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        asset_class: str,
        timeframe: str,
    ) -> BacktestResult:
        """Run backtest on a signal-enriched DataFrame.

        The DataFrame must already have signal columns from generate_signals().
        The backtester processes bar-by-bar with no lookahead.

        Args:
            df: Signal-enriched OHLCV DataFrame.
            symbol: Symbol being tested.
            asset_class: Asset class (for bucket resolution).
            timeframe: Timeframe string.

        Returns:
            BacktestResult with trade log.
        """
        bucket = get_bucket_for_asset(asset_class, self._signal_config)
        max_hold_bars = self._compute_max_hold_bars(bucket, timeframe)

        position: Position | None = None
        trades: list[Trade] = []

        for i in range(len(df)):
            row = df.iloc[i]

            # If we have an open position, check exits first
            if position is not None:
                trade = self._check_exits(row, i, position)
                if trade is not None:
                    trades.append(trade)
                    position = None

            # If no position, check for entry
            if position is None and self._should_enter(row):
                position = self._open_position(
                    row, i, symbol, asset_class, timeframe, bucket, max_hold_bars
                )

        # Force-close any open position at end of data
        if position is not None:
            last_row = df.iloc[-1]
            trade = self._close_position(
                position,
                last_row["timestamp"],
                last_row["close"],
                "end_of_data",
                len(df) - 1,
            )
            trades.append(trade)

        return BacktestResult(
            trades=trades,
            config=self._config,
            signal_config_hash=_compute_config_hash(),
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
            start_date=df["timestamp"].iloc[0].date()
            if hasattr(df["timestamp"].iloc[0], "date")
            else df["timestamp"].iloc[0],
            end_date=df["timestamp"].iloc[-1].date()
            if hasattr(df["timestamp"].iloc[-1], "date")
            else df["timestamp"].iloc[-1],
        )

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _should_enter(self, row: pd.Series) -> bool:
        """Check if this bar triggers a new entry.

        Entry criteria:
        - signal_tradeable is True
        - exit_stop is not NaN (valid exit levels)
        """
        if not row.get("signal_tradeable", False):
            return False
        if pd.isna(row.get("exit_stop", np.nan)):
            return False
        if pd.isna(row.get("exit_target", np.nan)):
            return False
        return True

    def _open_position(
        self,
        row: pd.Series,
        bar_index: int,
        symbol: str,
        asset_class: str,
        timeframe: str,
        bucket: BucketConfig,
        max_hold_bars: int | None,
    ) -> Position:
        """Create a new Position from a signal bar."""
        return Position(
            symbol=symbol,
            asset_class=asset_class,
            timeframe=timeframe,
            bucket="A" if bucket.name == "Short Swing" else "B",
            direction=row["signal_direction"],
            entry_bar_index=bar_index,
            entry_timestamp=row["timestamp"],
            entry_price=row["close"],
            entry_signal_score=int(row["signal_score"]),
            entry_regime=row["regime"],
            stop_loss=row["exit_stop"],
            target=row["exit_target"],
            trail_breakeven_price=row["exit_trail_be"],
            max_hold_bars=max_hold_bars,
        )

    def _compute_max_hold_bars(
        self, bucket: BucketConfig, timeframe: str
    ) -> int | None:
        """Convert max_hold_weeks to max_hold_bars."""
        if bucket.max_hold_weeks is None:
            return None
        bars_per_week = _timeframe_to_bars_per_week(timeframe)
        return int(bucket.max_hold_weeks * bars_per_week)

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        row: pd.Series,
        bar_index: int,
        position: Position,
    ) -> Trade | None:
        """Check all exit conditions in priority order.

        Priority: stop → target → trail update → max hold.
        Returns a Trade if position was closed, None if still open.
        """
        # 1. Stop loss
        exit_price = self._check_stop_hit(row, position)
        if exit_price is not None:
            reason = (
                "trail_stop" if position.stop_trailed_to_breakeven else "stop"
            )
            return self._close_position(
                position, row["timestamp"], exit_price, reason, bar_index
            )

        # 2. Target
        exit_price = self._check_target_hit(row, position)
        if exit_price is not None:
            return self._close_position(
                position, row["timestamp"], exit_price, "target", bar_index
            )

        # 3. Trail-to-breakeven update (does not close — updates stop)
        self._update_trail_stop(row, position)

        # 4. Max hold
        exit_price = self._check_max_hold(row, bar_index, position)
        if exit_price is not None:
            return self._close_position(
                position, row["timestamp"], exit_price, "max_hold", bar_index
            )

        return None

    def _check_stop_hit(
        self, row: pd.Series, position: Position
    ) -> float | None:
        """Check if stop was hit. Returns exit price or None."""
        if position.direction == "long":
            if row["low"] <= position.current_stop:
                return position.current_stop
        else:  # short
            if row["high"] >= position.current_stop:
                return position.current_stop
        return None

    def _check_target_hit(
        self, row: pd.Series, position: Position
    ) -> float | None:
        """Check if target was hit. Returns exit price or None."""
        if position.direction == "long":
            if row["high"] >= position.target:
                return position.target
        else:  # short
            if row["low"] <= position.target:
                return position.target
        return None

    def _update_trail_stop(
        self, row: pd.Series, position: Position
    ) -> None:
        """Update stop to breakeven if trail level has been reached."""
        if position.stop_trailed_to_breakeven:
            return  # Already trailed

        if np.isnan(position.trail_breakeven_price):
            return

        if position.direction == "long":
            if row["high"] >= position.trail_breakeven_price:
                position.current_stop = position.entry_price
                position.stop_trailed_to_breakeven = True
        else:  # short
            if row["low"] <= position.trail_breakeven_price:
                position.current_stop = position.entry_price
                position.stop_trailed_to_breakeven = True

    def _check_max_hold(
        self, row: pd.Series, bar_index: int, position: Position
    ) -> float | None:
        """Check if max hold period exceeded. Returns close price or None."""
        if position.max_hold_bars is None:
            return None
        bars_held = bar_index - position.entry_bar_index
        if bars_held >= position.max_hold_bars:
            return row["close"]
        return None

    def _close_position(
        self,
        position: Position,
        exit_timestamp: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        bar_index: int,
    ) -> Trade:
        """Convert an open Position to a closed Trade."""
        # Risk is always based on original stop, not current_stop
        risk = abs(position.entry_price - position.stop_loss)
        if risk < 1e-10:
            risk = 1e-10  # Guard against zero division

        if position.direction == "long":
            pnl_dollar = exit_price - position.entry_price
        else:
            pnl_dollar = position.entry_price - exit_price

        pnl_r = pnl_dollar / risk

        duration_bars = bar_index - position.entry_bar_index

        # Calendar days
        if hasattr(exit_timestamp, "date") and hasattr(
            position.entry_timestamp, "date"
        ):
            duration_days = (exit_timestamp - position.entry_timestamp).days
        else:
            duration_days = 0

        return Trade(
            symbol=position.symbol,
            asset_class=position.asset_class,
            timeframe=position.timeframe,
            bucket=position.bucket,
            direction=position.direction,
            entry_timestamp=position.entry_timestamp,
            entry_price=position.entry_price,
            entry_signal_score=position.entry_signal_score,
            entry_regime=position.entry_regime,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_r=pnl_r,
            pnl_dollar=pnl_dollar,
            duration_bars=duration_bars,
            duration_calendar_days=duration_days,
        )
