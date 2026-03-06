"""Momentum indicators: RSI, MACD."""

import pandas as pd
import pandas_ta as ta


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index column.

    Args:
        df: Canonical OHLCV DataFrame.
        period: RSI lookback period (default: 14).

    Returns:
        DataFrame with 'rsi_{period}' column appended.
    """
    result = df.copy()
    result[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return result


def add_rsi_direction(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI value and direction (rising/falling) columns.

    Args:
        df: Canonical OHLCV DataFrame.
        period: RSI lookback period.

    Returns:
        DataFrame with 'rsi_{period}' and 'rsi_rising' columns.
    """
    result = df.copy()
    rsi = ta.rsi(df["close"], length=period)
    result[f"rsi_{period}"] = rsi
    result["rsi_rising"] = rsi.diff() > 0
    return result


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD, signal line, and histogram columns.

    Args:
        df: Canonical OHLCV DataFrame.
        fast: Fast EMA period (default: 12).
        slow: Slow EMA period (default: 26).
        signal: Signal line EMA period (default: 9).

    Returns:
        DataFrame with 'macd', 'macd_signal', 'macd_histogram' columns.
    """
    result = df.copy()
    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)

    # pandas-ta returns columns like MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd_col = f"MACD_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    signal_col = f"MACDs_{fast}_{slow}_{signal}"

    result["macd"] = macd_df[macd_col]
    result["macd_signal"] = macd_df[signal_col]
    result["macd_histogram"] = macd_df[hist_col]

    return result
