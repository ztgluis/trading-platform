"""Structure indicators: swing highs/lows, higher lows, lower highs."""

import pandas as pd
import numpy as np


def detect_swing_highs(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect swing highs (local maxima in 'high' column).

    A swing high occurs when the high at bar i is greater than the highs
    of the preceding and following `lookback` bars.

    Args:
        df: Canonical OHLCV DataFrame.
        lookback: Number of bars on each side to compare (default: 3).

    Returns:
        DataFrame with 'swing_high' (bool) and 'swing_high_price' columns.
    """
    result = df.copy()
    highs = df["high"].values
    n = len(highs)
    is_swing_high = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        left = highs[i - lookback:i]
        right = highs[i + 1:i + lookback + 1]
        if highs[i] > left.max() and highs[i] > right.max():
            is_swing_high[i] = True

    result["swing_high"] = is_swing_high
    result["swing_high_price"] = np.where(is_swing_high, highs, np.nan)
    return result


def detect_swing_lows(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect swing lows (local minima in 'low' column).

    A swing low occurs when the low at bar i is less than the lows
    of the preceding and following `lookback` bars.

    Args:
        df: Canonical OHLCV DataFrame.
        lookback: Number of bars on each side to compare.

    Returns:
        DataFrame with 'swing_low' (bool) and 'swing_low_price' columns.
    """
    result = df.copy()
    lows = df["low"].values
    n = len(lows)
    is_swing_low = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        left = lows[i - lookback:i]
        right = lows[i + 1:i + lookback + 1]
        if lows[i] < left.min() and lows[i] < right.min():
            is_swing_low[i] = True

    result["swing_low"] = is_swing_low
    result["swing_low_price"] = np.where(is_swing_low, lows, np.nan)
    return result


def detect_higher_lows(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect bars where the most recent swing low is higher than the previous one.

    Requires swing lows to be detected first (calls detect_swing_lows internally).

    Args:
        df: Canonical OHLCV DataFrame.
        lookback: Swing detection lookback.

    Returns:
        DataFrame with swing low columns plus 'higher_low' (bool).
    """
    result = detect_swing_lows(df, lookback)

    # Extract swing low prices in order
    swing_prices = result.loc[result["swing_low"], "swing_low_price"].values
    higher_low = pd.Series(False, index=result.index)

    if len(swing_prices) >= 2:
        swing_indices = result.index[result["swing_low"]].tolist()
        for i in range(1, len(swing_prices)):
            if swing_prices[i] > swing_prices[i - 1]:
                # Mark all bars from this swing low until the next swing low (or end)
                idx = swing_indices[i]
                higher_low.loc[idx] = True

    result["higher_low"] = higher_low
    return result


def detect_lower_highs(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Detect bars where the most recent swing high is lower than the previous one.

    Args:
        df: Canonical OHLCV DataFrame.
        lookback: Swing detection lookback.

    Returns:
        DataFrame with swing high columns plus 'lower_high' (bool).
    """
    result = detect_swing_highs(df, lookback)

    swing_prices = result.loc[result["swing_high"], "swing_high_price"].values
    lower_high = pd.Series(False, index=result.index)

    if len(swing_prices) >= 2:
        swing_indices = result.index[result["swing_high"]].tolist()
        for i in range(1, len(swing_prices)):
            if swing_prices[i] < swing_prices[i - 1]:
                idx = swing_indices[i]
                lower_high.loc[idx] = True

    result["lower_high"] = lower_high
    return result
