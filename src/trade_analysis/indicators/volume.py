"""Volume indicators: volume SMA, spike detection."""

import pandas as pd
import pandas_ta as ta


def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add volume Simple Moving Average column.

    Args:
        df: Canonical OHLCV DataFrame.
        period: Lookback period.

    Returns:
        DataFrame with 'volume_sma_{period}' column appended.
    """
    result = df.copy()
    result[f"volume_sma_{period}"] = ta.sma(df["volume"], length=period)
    return result


def detect_volume_spike(
    df: pd.DataFrame,
    period: int = 20,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect volume spikes (volume exceeding threshold * SMA).

    Args:
        df: Canonical OHLCV DataFrame.
        period: SMA lookback period for volume baseline.
        threshold: Multiplier above SMA to qualify as a spike (default: 1.5x).

    Returns:
        DataFrame with 'volume_sma_{period}' and 'volume_spike' (bool) columns.
    """
    result = df.copy()
    vol_sma = ta.sma(df["volume"], length=period)
    result[f"volume_sma_{period}"] = vol_sma
    result["volume_spike"] = df["volume"] > (vol_sma * threshold)
    return result
