"""Trend indicators: moving averages and volatility measures."""

import numpy as np
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
    dispatchers = {
        "sma": add_sma,
        "ema": add_ema,
        "hma": add_hma,
        "zlema": add_zlema,
    }
    if ma_type not in dispatchers:
        raise ValueError(
            f"Unknown MA type: {ma_type}. Use one of: {', '.join(dispatchers)}."
        )
    return dispatchers[ma_type](df, period, column)


def add_hma(df: pd.DataFrame, period: int = 21, column: str = "close") -> pd.DataFrame:
    """Add Hull Moving Average column.

    HMA = WMA(2 * WMA(src, n/2) - WMA(src, n), sqrt(n))
    Provides smoothing with minimal lag.

    Args:
        df: Canonical OHLCV DataFrame.
        period: Lookback period.
        column: Source column.

    Returns:
        DataFrame with 'hma_{period}' column appended.
    """
    result = df.copy()
    result[f"hma_{period}"] = ta.hma(df[column], length=period)
    return result


def add_zlema(df: pd.DataFrame, period: int = 70, column: str = "close") -> pd.DataFrame:
    """Add Zero-Lag EMA column.

    ZLEMA compensates for EMA lag by pre-adjusting the input:
        lag = floor((period - 1) / 2)
        zlema = EMA(src + (src - src[lag]), period)

    Args:
        df: Canonical OHLCV DataFrame.
        period: Lookback period.
        column: Source column.

    Returns:
        DataFrame with 'zlema_{period}' column appended.
    """
    result = df.copy()
    src = df[column]
    lag = int(np.floor((period - 1) / 2))
    adjusted = src + (src - src.shift(lag))
    result[f"zlema_{period}"] = ta.ema(adjusted, length=period)
    return result


def add_vidya(
    df: pd.DataFrame,
    length: int = 10,
    momentum_period: int = 20,
    smoothing: int = 15,
    column: str = "close",
) -> pd.DataFrame:
    """Add Variable Index Dynamic Average (VIDYA) column.

    Tushar Chande's adaptive MA that speeds up in trends and slows in chop.
    Uses the absolute Chande Momentum Oscillator (CMO) as the adaptive factor.

    Args:
        df: Canonical OHLCV DataFrame.
        length: Base EMA smoothing length (alpha = 2 / (length + 1)).
        momentum_period: CMO lookback window.
        smoothing: Final SMA smoothing period applied to raw VIDYA.
        column: Source column.

    Returns:
        DataFrame with 'vidya_{length}' column appended.
    """
    result = df.copy()
    src = df[column].values
    n = len(src)

    # Step 1: Compute bar-by-bar momentum
    mom = np.diff(src, prepend=np.nan)

    # Step 2: Rolling sums of positive and negative momentum
    pos_mom = np.where(np.isnan(mom), 0.0, np.where(mom >= 0, mom, 0.0))
    neg_mom = np.where(np.isnan(mom), 0.0, np.where(mom < 0, -mom, 0.0))

    sum_pos = pd.Series(pos_mom).rolling(momentum_period).sum().values
    sum_neg = pd.Series(neg_mom).rolling(momentum_period).sum().values

    # Step 3: Absolute CMO
    total = sum_pos + sum_neg
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_cmo = np.abs(100.0 * (sum_pos - sum_neg) / total)
    abs_cmo = np.where(np.isfinite(abs_cmo), abs_cmo, 0.0)

    # Step 4: Adaptive EMA (VIDYA)
    alpha = 2.0 / (length + 1)
    vidya = np.full(n, np.nan)
    vidya[0] = src[0]
    for i in range(1, n):
        if np.isnan(vidya[i - 1]):
            vidya[i] = src[i]
        else:
            factor = alpha * abs_cmo[i] / 100.0
            vidya[i] = factor * src[i] + (1.0 - factor) * vidya[i - 1]

    # Step 5: Final SMA smoothing
    vidya_series = pd.Series(vidya)
    result[f"vidya_{length}"] = vidya_series.rolling(smoothing).mean().values

    return result


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range column.

    Args:
        df: Canonical OHLCV DataFrame (must have high, low, close).
        period: ATR lookback period.

    Returns:
        DataFrame with 'atr_{period}' column appended.
    """
    result = df.copy()
    result[f"atr_{period}"] = ta.atr(
        high=df["high"], low=df["low"], close=df["close"], length=period
    )
    return result
