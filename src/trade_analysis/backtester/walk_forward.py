"""Walk-forward validation: split generation and sequential runner.

Walk-forward prevents overfitting by splitting data into in-sample (IS) and
out-of-sample (OOS) windows. The same signal config is used for both IS and
OOS — no parameter re-optimization between folds (PRD requirement).
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from trade_analysis.backtester.config import BacktestConfig
from trade_analysis.backtester.engine import Backtester
from trade_analysis.backtester.models import (
    BacktestResult,
    WalkForwardResult,
    WalkForwardSplit,
)
from trade_analysis.exceptions import ConfigError
from trade_analysis.signals.engine import SignalEngineConfig


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------


def generate_walk_forward_splits(
    start_date: date,
    end_date: date,
    in_sample_years: int,
    out_of_sample_years: int,
    anchored: bool = True,
) -> list[WalkForwardSplit]:
    """Generate walk-forward in-sample / out-of-sample split windows.

    Args:
        start_date: Earliest date in the data.
        end_date: Latest date in the data.
        in_sample_years: Length of each IS window in years.
        out_of_sample_years: Length of each OOS window in years.
        anchored: If True, IS start is always ``start_date`` (expanding window).
            If False, IS window rolls forward (fixed size).

    Returns:
        List of WalkForwardSplit, ordered by fold number.

    Raises:
        ConfigError: If the date range is too short for even one split.
    """
    min_years = in_sample_years + out_of_sample_years
    total_days = (end_date - start_date).days
    min_days = int(min_years * 365.25)

    if total_days < min_days:
        raise ConfigError(
            f"Date range ({start_date} to {end_date}) is too short for "
            f"walk-forward splits requiring {min_years} years "
            f"({in_sample_years} IS + {out_of_sample_years} OOS)"
        )

    splits: list[WalkForwardSplit] = []
    fold = 0

    while True:
        if anchored:
            # Anchored: IS always starts at start_date, expands each fold
            # Fold k: IS = [start, start + IS + k*OOS)
            #          OOS = [start + IS + k*OOS, start + IS + (k+1)*OOS)
            fold_is_start = start_date
            is_end_date = _add_years(start_date, in_sample_years + fold * out_of_sample_years)
        else:
            # Rolling: IS window slides forward by OOS years each fold
            # Fold k: IS = [start + k*OOS, start + k*OOS + IS)
            #          OOS = [start + k*OOS + IS, start + (k+1)*OOS + IS)
            fold_is_start = _add_years(start_date, fold * out_of_sample_years)
            is_end_date = _add_years(fold_is_start, in_sample_years)

        oos_start_date = is_end_date
        oos_end_date = _add_years(oos_start_date, out_of_sample_years)

        # Stop if OOS extends beyond our data
        if oos_end_date > end_date:
            break

        splits.append(
            WalkForwardSplit(
                fold=fold,
                is_start=fold_is_start,
                is_end=is_end_date,
                oos_start=oos_start_date,
                oos_end=oos_end_date,
            )
        )

        fold += 1

    if not splits:
        raise ConfigError(
            f"Date range ({start_date} to {end_date}) is too short for "
            f"walk-forward splits requiring {min_years} years "
            f"({in_sample_years} IS + {out_of_sample_years} OOS)"
        )

    return splits


def _add_years(d: date, years: int) -> date:
    """Add approximately ``years`` to a date.

    Handles leap year edge case (Feb 29 → Feb 28).
    """
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        # Feb 29 in a leap year + non-leap target
        return d.replace(year=d.year + years, month=2, day=28)


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------


def run_walk_forward(
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    signal_config: SignalEngineConfig,
) -> WalkForwardResult:
    """Run walk-forward validation across all splits.

    Uses the same signal config for in-sample and out-of-sample.
    The DataFrame is sliced by each split's date range.

    Args:
        df: Signal-enriched OHLCV DataFrame (full date range).
        symbol: Symbol being tested.
        asset_class: Asset class for bucket resolution.
        timeframe: Timeframe string.
        backtest_config: Backtest configuration (contains WF params).
        signal_config: Signal engine configuration.

    Returns:
        WalkForwardResult with IS and OOS results for each fold.

    Raises:
        ConfigError: If walk_forward is not configured or date range is
            insufficient.
    """
    wf = backtest_config.walk_forward
    if wf is None:
        raise ConfigError("walk_forward config is required for walk-forward runs")

    splits = generate_walk_forward_splits(
        start_date=backtest_config.start_date,
        end_date=backtest_config.end_date,
        in_sample_years=wf.in_sample_years,
        out_of_sample_years=wf.out_of_sample_years,
        anchored=wf.anchored,
    )

    backtester = Backtester(backtest_config, signal_config)

    is_results: list[BacktestResult] = []
    oos_results: list[BacktestResult] = []

    for split in splits:
        # Slice dataframe by date range
        is_df = _slice_df_by_dates(df, split.is_start, split.is_end)
        oos_df = _slice_df_by_dates(df, split.oos_start, split.oos_end)

        # Run backtester on each slice
        is_result = backtester.run(is_df, symbol, asset_class, timeframe)
        oos_result = backtester.run(oos_df, symbol, asset_class, timeframe)

        is_results.append(is_result)
        oos_results.append(oos_result)

    return WalkForwardResult(
        splits=splits,
        in_sample_results=is_results,
        out_of_sample_results=oos_results,
    )


def _slice_df_by_dates(
    df: pd.DataFrame, start: date, end: date
) -> pd.DataFrame:
    """Slice a DataFrame to a date range using the timestamp column.

    Args:
        df: DataFrame with a 'timestamp' column (timezone-aware).
        start: Start date (inclusive).
        end: End date (exclusive).

    Returns:
        Filtered DataFrame.
    """
    timestamps = df["timestamp"]

    # Handle both timezone-aware and naive timestamps
    if hasattr(timestamps.iloc[0], "tz") and timestamps.iloc[0].tz is not None:
        start_ts = pd.Timestamp(start, tz=timestamps.iloc[0].tz)
        end_ts = pd.Timestamp(end, tz=timestamps.iloc[0].tz)
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

    mask = (timestamps >= start_ts) & (timestamps < end_ts)
    return df.loc[mask].reset_index(drop=True)
