"""Shared fixtures for all test modules."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_daily_ohlcv() -> pd.DataFrame:
    """10 days of realistic AAPL daily data in canonical format."""
    timestamps = pd.date_range("2024-01-02", periods=10, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    base = 185.0
    closes = base + rng.standard_normal(10).cumsum()
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes - rng.uniform(0, 1, 10),
        "high": closes + rng.uniform(1, 3, 10),
        "low": closes - rng.uniform(1, 3, 10),
        "close": closes,
        "volume": rng.uniform(5e7, 1e8, 10),
    })
    df.attrs = {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "Daily",
        "provider": "yfinance",
        "is_inverse": False,
        "inverse_of": None,
    }
    return df


@pytest.fixture
def sample_1h_ohlcv() -> pd.DataFrame:
    """40 hours of 1H data (enough for 10 x 4H bars)."""
    timestamps = pd.date_range("2024-01-02 09:30", periods=40, freq="h", tz="UTC")
    rng = np.random.default_rng(123)
    base = 185.0
    closes = base + rng.standard_normal(40).cumsum() * 0.5
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes - rng.uniform(0, 0.5, 40),
        "high": closes + rng.uniform(0.2, 1.5, 40),
        "low": closes - rng.uniform(0.2, 1.5, 40),
        "close": closes,
        "volume": rng.uniform(1e6, 5e6, 40),
    })
    df.attrs = {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "1H",
        "provider": "yfinance",
        "is_inverse": False,
        "inverse_of": None,
    }
    return df


@pytest.fixture
def sample_yfinance_raw() -> pd.DataFrame:
    """Raw yfinance-style DataFrame (capitalized columns, DatetimeIndex)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rng = np.random.default_rng(42)
    base = 185.0
    closes = base + rng.standard_normal(5).cumsum()
    df = pd.DataFrame(
        {
            "Open": closes - rng.uniform(0, 1, 5),
            "High": closes + rng.uniform(1, 3, 5),
            "Low": closes - rng.uniform(1, 3, 5),
            "Close": closes,
            "Adj Close": closes * 0.99,
            "Volume": rng.integers(5e7, 1e8, 5).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )
    return df


@pytest.fixture
def sample_ccxt_raw() -> list[list]:
    """Raw CCXT-style data: list of [timestamp_ms, O, H, L, C, V]."""
    import time

    base_ts = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    rng = np.random.default_rng(42)
    base_price = 42000.0
    data = []
    for i in range(10):
        ts = base_ts + i * 3600_000  # 1 hour intervals
        close = base_price + rng.standard_normal() * 200
        data.append([
            ts,
            close - rng.uniform(0, 50),
            close + rng.uniform(50, 200),
            close - rng.uniform(50, 200),
            close,
            rng.uniform(100, 1000),
        ])
    return data


@pytest.fixture
def sample_200bar_ohlcv() -> pd.DataFrame:
    """200 business days of AAPL-like data for indicator warmup testing."""
    timestamps = pd.date_range("2023-01-02", periods=200, freq="B", tz="UTC")
    rng = np.random.default_rng(42)
    base = 150.0
    closes = base + rng.standard_normal(200).cumsum() * 0.8
    # Ensure all prices stay positive
    closes = np.maximum(closes, 10.0)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes - rng.uniform(0, 1.5, 200),
        "high": closes + rng.uniform(0.5, 3, 200),
        "low": closes - rng.uniform(0.5, 3, 200),
        "close": closes,
        "volume": rng.uniform(3e7, 1.2e8, 200),
    })
    df.attrs = {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "Daily",
        "provider": "yfinance",
        "is_inverse": False,
        "inverse_of": None,
    }
    return df
