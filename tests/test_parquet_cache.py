"""Tests for parquet-based OHLCV cache."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_analysis.cache.parquet_cache import ParquetCache
from trade_analysis.models.ohlcv import Timeframe


def make_ohlcv(rows: int = 5, start: str = "2024-01-01") -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=rows, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    closes = 150.0 + rng.standard_normal(rows).cumsum()
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes - rng.uniform(0, 2, rows),
        "high": closes + rng.uniform(1, 5, rows),
        "low": closes - rng.uniform(1, 5, rows),
        "close": closes,
        "volume": rng.uniform(1e6, 1e7, rows),
    })


@pytest.fixture
def cache(tmp_path) -> ParquetCache:
    return ParquetCache(
        storage_path=tmp_path / "cache",
        ttl_seconds={"Daily": 3600, "4H": 1800},
        max_age_days=30,
    )


class TestParquetCache:
    def test_put_and_get(self, cache):
        df = make_ohlcv()
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")
        result = cache.get("AAPL", Timeframe.DAILY, "yfinance")
        assert result is not None
        assert len(result) == 5
        pd.testing.assert_frame_equal(result, df)

    def test_cache_miss(self, cache):
        result = cache.get("AAPL", Timeframe.DAILY, "yfinance")
        assert result is None

    def test_ttl_expired(self, cache, tmp_path):
        df = make_ohlcv()
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")

        # Manually set last_fetch to 2 hours ago (TTL is 3600s)
        meta_path = (
            tmp_path / "cache" / "yfinance" / "AAPL" / "Daily.meta.json"
        )
        meta = json.loads(meta_path.read_text())
        meta["last_fetch"] = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).isoformat()
        meta_path.write_text(json.dumps(meta))

        result = cache.get("AAPL", Timeframe.DAILY, "yfinance")
        assert result is None

    def test_date_range_filter(self, cache):
        df = make_ohlcv(10, "2024-01-01")
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")

        result = cache.get(
            "AAPL", Timeframe.DAILY, "yfinance",
            start=datetime(2024, 1, 3, tzinfo=timezone.utc),
            end=datetime(2024, 1, 7, tzinfo=timezone.utc),
        )
        assert result is not None
        assert len(result) == 5

    def test_merge_with_existing(self, cache):
        df1 = make_ohlcv(5, "2024-01-01")
        cache.put(df1, "AAPL", Timeframe.DAILY, "yfinance")

        df2 = make_ohlcv(5, "2024-01-04")  # Overlaps days 4-5
        cache.put(df2, "AAPL", Timeframe.DAILY, "yfinance")

        result = cache.get("AAPL", Timeframe.DAILY, "yfinance")
        assert result is not None
        # Should have merged: days 1-3 from df1, days 4-8 from df2
        assert len(result) == 8  # 3 unique from df1 + 5 from df2
        assert result["timestamp"].is_monotonic_increasing

    def test_symbol_sanitization(self, cache):
        df = make_ohlcv()
        cache.put(df, "BTC/USDT", Timeframe.DAILY, "ccxt")
        result = cache.get("BTC/USDT", Timeframe.DAILY, "ccxt")
        assert result is not None

    def test_futures_symbol_sanitization(self, cache):
        df = make_ohlcv()
        cache.put(df, "GC=F", Timeframe.DAILY, "yfinance")
        result = cache.get("GC=F", Timeframe.DAILY, "yfinance")
        assert result is not None

    def test_invalidate_specific(self, cache):
        df = make_ohlcv()
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")
        cache.put(df, "MSFT", Timeframe.DAILY, "yfinance")

        count = cache.invalidate(symbol="AAPL")
        assert count == 1
        assert cache.get("AAPL", Timeframe.DAILY, "yfinance") is None
        assert cache.get("MSFT", Timeframe.DAILY, "yfinance") is not None

    def test_invalidate_all(self, cache):
        df = make_ohlcv()
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")
        cache.put(df, "MSFT", Timeframe.DAILY, "yfinance")

        count = cache.invalidate()
        assert count == 2

    def test_cleanup_expired(self, cache, tmp_path):
        df = make_ohlcv()
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")

        # Set last_fetch to 60 days ago (max_age is 30)
        meta_path = (
            tmp_path / "cache" / "yfinance" / "AAPL" / "Daily.meta.json"
        )
        meta = json.loads(meta_path.read_text())
        meta["last_fetch"] = (
            datetime.now(timezone.utc) - timedelta(days=60)
        ).isoformat()
        meta_path.write_text(json.dumps(meta))

        count = cache.cleanup_expired()
        assert count == 1

    def test_list_cached(self, cache):
        df = make_ohlcv()
        cache.put(df, "AAPL", Timeframe.DAILY, "yfinance")
        cache.put(df, "BTC/USDT", Timeframe.H4, "ccxt")

        entries = cache.list_cached()
        assert len(entries) == 2
        symbols = {e["symbol"] for e in entries}
        assert symbols == {"AAPL", "BTC/USDT"}

    def test_corrupt_parquet_returns_none(self, cache, tmp_path):
        # Write a corrupt file
        path = tmp_path / "cache" / "yfinance" / "AAPL"
        path.mkdir(parents=True)
        (path / "Daily.parquet").write_text("not a parquet file")
        meta = {"last_fetch": datetime.now(timezone.utc).isoformat()}
        (path / "Daily.meta.json").write_text(json.dumps(meta))

        result = cache.get("AAPL", Timeframe.DAILY, "yfinance")
        assert result is None
        # Corrupt file should be cleaned up
        assert not (path / "Daily.parquet").exists()
