"""Tests for data normalization transforms."""

import pandas as pd
import pytest

from trade_analysis.models.ohlcv import AssetClass, Timeframe, validate_ohlcv
from trade_analysis.transforms.normalize import normalize_ccxt, normalize_yfinance


class TestNormalizeYFinance:
    def test_basic_normalization(self, sample_yfinance_raw):
        df = normalize_yfinance(
            sample_yfinance_raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK
        )
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 5

    def test_timestamp_is_utc(self, sample_yfinance_raw):
        df = normalize_yfinance(
            sample_yfinance_raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK
        )
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_adj_close_dropped(self, sample_yfinance_raw):
        df = normalize_yfinance(
            sample_yfinance_raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK
        )
        assert "adj_close" not in df.columns
        assert "Adj Close" not in df.columns

    def test_passes_validation(self, sample_yfinance_raw):
        df = normalize_yfinance(
            sample_yfinance_raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK
        )
        assert validate_ohlcv(df) is True

    def test_metadata_attached(self, sample_yfinance_raw):
        df = normalize_yfinance(
            sample_yfinance_raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK
        )
        assert df.attrs["symbol"] == "AAPL"
        assert df.attrs["provider"] == "yfinance"
        assert df.attrs["timeframe"] == "Daily"

    def test_nan_rows_dropped(self, sample_yfinance_raw):
        raw = sample_yfinance_raw.copy()
        raw.loc[raw.index[2], "Close"] = float("nan")
        df = normalize_yfinance(raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK)
        assert len(df) == 4

    def test_sorted_ascending(self, sample_yfinance_raw):
        raw = sample_yfinance_raw.iloc[::-1]  # Reverse
        df = normalize_yfinance(raw, "AAPL", Timeframe.DAILY, AssetClass.STOCK)
        assert df["timestamp"].is_monotonic_increasing


class TestNormalizeCCXT:
    def test_basic_normalization(self, sample_ccxt_raw):
        df = normalize_ccxt(
            sample_ccxt_raw, "BTC/USDT", Timeframe.H1, AssetClass.CRYPTO
        )
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(df) == 10

    def test_timestamp_from_milliseconds(self, sample_ccxt_raw):
        df = normalize_ccxt(
            sample_ccxt_raw, "BTC/USDT", Timeframe.H1, AssetClass.CRYPTO
        )
        assert df["timestamp"].dt.tz is not None
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_passes_validation(self, sample_ccxt_raw):
        df = normalize_ccxt(
            sample_ccxt_raw, "BTC/USDT", Timeframe.H1, AssetClass.CRYPTO
        )
        assert validate_ohlcv(df) is True

    def test_metadata_attached(self, sample_ccxt_raw):
        df = normalize_ccxt(
            sample_ccxt_raw, "BTC/USDT", Timeframe.H1, AssetClass.CRYPTO
        )
        assert df.attrs["symbol"] == "BTC/USDT"
        assert df.attrs["provider"] == "ccxt"
        assert df.attrs["asset_class"] == "crypto"

    def test_accepts_dataframe_input(self, sample_ccxt_raw):
        raw_df = pd.DataFrame(
            sample_ccxt_raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df = normalize_ccxt(raw_df, "ETH/USDT", Timeframe.H4, AssetClass.CRYPTO)
        assert validate_ohlcv(df) is True
