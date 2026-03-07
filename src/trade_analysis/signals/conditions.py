"""Condition evaluators: trend, structure, momentum.

Each evaluator adds boolean columns to the DataFrame indicating whether
a specific trading condition is met on each bar. The signal scoring
module then combines these into a composite score.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from trade_analysis.indicators.levels import (
    detect_pivot_levels,
    detect_round_numbers,
    find_nearest_level,
)
from trade_analysis.indicators.structure import (
    detect_higher_lows,
    detect_lower_highs,
)


# ---------------------------------------------------------------------------
# Condition 1: Trend
# ---------------------------------------------------------------------------


def evaluate_trend_condition(
    df: pd.DataFrame,
    ma_column: str,
) -> pd.DataFrame:
    """Evaluate the trend condition: is price above or below the trend MA?

    A simple but critical filter — price must be on the correct side of the
    bucket's trend-defining MA (e.g. EMA 21 for Bucket A, SMA 50 for Bucket B).

    Args:
        df: DataFrame that already contains the MA column.
        ma_column: Name of the moving average column to compare against.

    Returns:
        DataFrame with columns:
            - trend_bull: bool — close > MA (bullish trend)
            - trend_bear: bool — close < MA (bearish trend)
    """
    result = df.copy()
    close = df["close"]
    ma = df[ma_column]

    result["trend_bull"] = close > ma
    result["trend_bear"] = close < ma

    # Fill NaN (during MA warmup) with False
    result["trend_bull"] = result["trend_bull"].fillna(False)
    result["trend_bear"] = result["trend_bear"].fillna(False)

    return result


# ---------------------------------------------------------------------------
# Condition 2: Structure
# ---------------------------------------------------------------------------


def evaluate_structure_condition(
    df: pd.DataFrame,
    swing_lookback: int = 3,
    level_proximity_pct: float = 3.0,
    pivot_lookback: int = 5,
    pivot_merge_distance_pct: float = 0.5,
) -> pd.DataFrame:
    """Evaluate the structure condition: higher lows / lower highs near key levels.

    Bullish structure = higher low detected near a support level (pivot or round number).
    Bearish structure = lower high detected near a resistance level.
    Multi-method = the level is confirmed by both pivot S/R AND round numbers.

    Args:
        df: Canonical OHLCV DataFrame.
        swing_lookback: Bars on each side for swing detection.
        level_proximity_pct: Max distance % to consider "near" a level.
        pivot_lookback: Lookback for pivot level detection.
        pivot_merge_distance_pct: Cluster distance for merging pivot levels.

    Returns:
        DataFrame with columns:
            - structure_bull: bool — bullish structure confirmed
            - structure_bear: bool — bearish structure confirmed
            - structure_near_pivot: bool — near a pivot S/R level
            - structure_near_round: bool — near a round number
            - structure_multi_method: bool — near BOTH pivot and round number
    """
    # Detect higher lows and lower highs
    with_hl = detect_higher_lows(df, lookback=swing_lookback)
    with_lh = detect_lower_highs(df, lookback=swing_lookback)

    result = df.copy()
    result["higher_low"] = with_hl["higher_low"]
    result["lower_high"] = with_lh["lower_high"]

    # Detect pivot levels from the data
    pivot_levels = detect_pivot_levels(
        df, lookback=pivot_lookback, merge_distance_pct=pivot_merge_distance_pct
    )

    # For each bar, check proximity to levels
    n = len(df)
    near_pivot = np.zeros(n, dtype=bool)
    near_round = np.zeros(n, dtype=bool)

    for i in range(n):
        price = df["close"].iloc[i]

        # Check pivot level proximity
        pivot_match = find_nearest_level(
            price, pivot_levels, max_distance_pct=level_proximity_pct
        )
        if pivot_match is not None:
            near_pivot[i] = True

        # Check round number proximity
        round_levels = detect_round_numbers(price)
        round_match = find_nearest_level(
            price, round_levels, max_distance_pct=level_proximity_pct
        )
        if round_match is not None:
            near_round[i] = True

    result["structure_near_pivot"] = near_pivot
    result["structure_near_round"] = near_round
    result["structure_multi_method"] = near_pivot & near_round

    # Structure conditions: swing pattern + near any level
    near_any_level = near_pivot | near_round
    result["structure_bull"] = result["higher_low"] & near_any_level
    result["structure_bear"] = result["lower_high"] & near_any_level

    return result


# ---------------------------------------------------------------------------
# Condition 3: Momentum
# ---------------------------------------------------------------------------


def evaluate_momentum_condition(
    df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_bull_threshold: float = 50.0,
    rsi_bear_threshold: float = 50.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Evaluate the momentum condition: RSI and MACD alignment.

    Bullish momentum = RSI > bull_threshold AND rising, OR MACD histogram > 0.
    Bearish momentum = RSI < bear_threshold AND falling, OR MACD histogram < 0.

    This evaluator is idempotent — it checks for existing RSI/MACD columns
    before computing them.

    Args:
        df: Canonical OHLCV DataFrame.
        rsi_period: RSI lookback period.
        rsi_bull_threshold: RSI level above which momentum is bullish.
        rsi_bear_threshold: RSI level below which momentum is bearish.
        macd_fast: MACD fast EMA period.
        macd_slow: MACD slow EMA period.
        macd_signal: MACD signal line period.

    Returns:
        DataFrame with columns:
            - momentum_rsi_bull: bool — RSI confirms bullish
            - momentum_rsi_bear: bool — RSI confirms bearish
            - momentum_macd_bull: bool — MACD confirms bullish
            - momentum_macd_bear: bool — MACD confirms bearish
            - momentum_bull: bool — overall bullish momentum (RSI OR MACD)
            - momentum_bear: bool — overall bearish momentum (RSI OR MACD)
    """
    result = df.copy()
    close = df["close"]

    # Compute RSI if not present
    rsi_col = f"rsi_{rsi_period}"
    if rsi_col in df.columns:
        rsi = df[rsi_col]
    else:
        rsi = ta.rsi(close, length=rsi_period)
        result[rsi_col] = rsi

    rsi_rising = rsi.diff() > 0

    # Compute MACD if not present
    if "macd_histogram" in df.columns:
        macd_hist = df["macd_histogram"]
    else:
        macd_df = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        hist_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
        macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        signal_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
        macd_hist = macd_df[hist_col]
        result["macd"] = macd_df[macd_col]
        result["macd_signal"] = macd_df[signal_col]
        result["macd_histogram"] = macd_hist

    # RSI conditions
    momentum_rsi_bull = (rsi > rsi_bull_threshold) & rsi_rising
    momentum_rsi_bear = (rsi < rsi_bear_threshold) & ~rsi_rising

    # MACD conditions
    momentum_macd_bull = macd_hist > 0
    momentum_macd_bear = macd_hist < 0

    # Fill NaNs with False
    momentum_rsi_bull = momentum_rsi_bull.fillna(False)
    momentum_rsi_bear = momentum_rsi_bear.fillna(False)
    momentum_macd_bull = momentum_macd_bull.fillna(False)
    momentum_macd_bear = momentum_macd_bear.fillna(False)

    # Overall: RSI OR MACD
    result["momentum_rsi_bull"] = momentum_rsi_bull
    result["momentum_rsi_bear"] = momentum_rsi_bear
    result["momentum_macd_bull"] = momentum_macd_bull
    result["momentum_macd_bear"] = momentum_macd_bear
    result["momentum_bull"] = momentum_rsi_bull | momentum_macd_bull
    result["momentum_bear"] = momentum_rsi_bear | momentum_macd_bear

    return result
