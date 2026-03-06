"""Normalize raw provider data to canonical OHLCV format."""

import pandas as pd

from trade_analysis.models.ohlcv import (
    AssetClass,
    OHLCVMeta,
    Timeframe,
    attach_metadata,
    validate_ohlcv,
)


def normalize_yfinance(
    raw_df: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    asset_class: AssetClass,
) -> pd.DataFrame:
    """Normalize yfinance output to canonical schema.

    Handles:
    - DatetimeIndex → 'timestamp' column
    - Column names capitalized → lowercase
    - Drops 'Adj Close' if present
    - Timezone-naive timestamps → UTC
    - NaN rows → dropped
    """
    df = raw_df.copy()

    # Reset index if timestamp is the index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Rename columns to lowercase
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        if lower == "date" or lower == "datetime":
            lower = "timestamp"
        col_map[col] = lower
    df = df.rename(columns=col_map)

    # Drop adj_close if present
    for col in ["adj_close", "adj close"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Ensure we have the timestamp column
    if "timestamp" not in df.columns:
        raise ValueError(f"Cannot find timestamp column in: {list(df.columns)}")

    # Convert timestamp to UTC-aware datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Ensure float64 dtypes
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")

    # Drop rows with NaN prices
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Keep only canonical columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Sort and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Attach metadata
    meta = OHLCVMeta(
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        provider="yfinance",
    )
    attach_metadata(df, meta)

    validate_ohlcv(df)
    return df


def normalize_ccxt(
    raw_data: list[list] | pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    asset_class: AssetClass,
) -> pd.DataFrame:
    """Normalize CCXT output to canonical schema.

    CCXT returns list of [timestamp_ms, open, high, low, close, volume].
    """
    if isinstance(raw_data, list):
        df = pd.DataFrame(
            raw_data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
    else:
        df = raw_data.copy()

    # Convert millisecond timestamps to datetime UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Ensure float64 dtypes
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")

    # Drop NaN prices
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Keep only canonical columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # Sort and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Attach metadata
    meta = OHLCVMeta(
        symbol=symbol,
        asset_class=asset_class,
        timeframe=timeframe,
        provider="ccxt",
    )
    attach_metadata(df, meta)

    validate_ohlcv(df)
    return df
