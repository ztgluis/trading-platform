"""Trend indicators: moving averages (SMA, EMA)."""

import pandas as pd
import pandas_ta as ta


def add_sma(df: pd.DataFrame, period: int = 50, column: str = "close") -> pd.DataFrame:
    """Add Simple Moving Average column.

    Args:
        df: Canonical OHLCV DataFrame.
        period: Lookback period.
        column: Source column (default: "close").

    Returns:
        DataFrame with 'sma_{period}' column appended.
    """
    result = df.copy()
    result[f"sma_{period}"] = ta.sma(df[column], length=period)
    return result


def add_ema(df: pd.DataFrame, period: int = 21, column: str = "close") -> pd.DataFrame:
    """Add Exponential Moving Average column.

    Args:
        df: Canonical OHLCV DataFrame.
        period: Lookback period.
        column: Source column (default: "close").

    Returns:
        DataFrame with 'ema_{period}' column appended.
    """
    result = df.copy()
    result[f"ema_{period}"] = ta.ema(df[column], length=period)
    return result


def add_ma(
    df: pd.DataFrame,
    period: int = 50,
    ma_type: str = "sma",
    column: str = "close",
) -> pd.DataFrame:
    """Add a moving average column (unified interface).

    Args:
        df: Canonical OHLCV DataFrame.
        period: Lookback period.
        ma_type: "sma" or "ema".
        column: Source column.

    Returns:
        DataFrame with '{ma_type}_{period}' column appended.
    """
    if ma_type == "sma":
        return add_sma(df, period, column)
    elif ma_type == "ema":
        return add_ema(df, period, column)
    else:
        raise ValueError(f"Unknown MA type: {ma_type}. Use 'sma' or 'ema'.")
