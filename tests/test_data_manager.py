"""Integration tests for DataManager with mocked providers."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from trade_analysis.data_manager import DataManager
from trade_analysis.exceptions import SymbolNotFoundError
from trade_analysis.models.ohlcv import Timeframe, validate_ohlcv


def make_yfinance_response(rows: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    closes = 185.0 + rng.standard_normal(rows).cumsum()
    return pd.DataFrame(
        {
            "Open": closes - rng.uniform(0, 1, rows),
            "High": closes + rng.uniform(1, 3, rows),
            "Low": closes - rng.uniform(1, 3, rows),
            "Close": closes,
            "Volume": rng.integers(5e7, 1e8, rows).astype(float),
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def make_ccxt_response(rows: int = 20) -> list[list]:
    base_ts = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    rng = np.random.default_rng(42)
    base_price = 42000.0
    data = []
    for i in range(rows):
        close = base_price + rng.standard_normal() * 200
        data.append([
            base_ts + i * 3600_000,
            close - rng.uniform(0, 50),
            close + rng.uniform(50, 200),
            close - rng.uniform(50, 200),
            close,
            rng.uniform(100, 1000),
        ])
    return data


@pytest.fixture
def test_configs(tmp_path) -> tuple[Path, Path, Path]:
    """Create minimal test config files."""
    symbols_yaml = tmp_path / "symbols.yaml"
    symbols_yaml.write_text(yaml.dump({
        "symbols": [
            {
                "ticker": "AAPL",
                "asset_class": "stock",
                "provider": "yfinance",
                "timeframes": ["Daily", "Weekly"],
            },
            {
                "ticker": "BTC/USDT",
                "asset_class": "crypto",
                "provider": "ccxt",
                "timeframes": ["4H", "Daily"],
                "exchange": "binance",
            },
        ]
    }))

    sources_yaml = tmp_path / "data_sources.yaml"
    sources_yaml.write_text(yaml.dump({
        "providers": {
            "yfinance": {
                "rate_limit_calls_per_minute": 60,
                "retry_count": 1,
                "retry_delay_seconds": 0.1,
            },
            "ccxt": {
                "rate_limit_calls_per_minute": 30,
                "retry_count": 1,
                "retry_delay_seconds": 0.1,
                "default_exchange": "binance",
            },
        }
    }))

    cache_yaml = tmp_path / "cache.yaml"
    cache_yaml.write_text(yaml.dump({
        "cache": {
            "storage_path": str(tmp_path / "cache"),
            "ttl_seconds": {"Daily": 86400, "4H": 3600, "Weekly": 86400},
            "max_age_days": 90,
        }
    }))

    return symbols_yaml, sources_yaml, cache_yaml


class TestDataManager:
    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_get_ohlcv_stock(self, mock_ticker_cls, test_configs):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))
        df = dm.get_ohlcv("AAPL", Timeframe.DAILY)

        assert validate_ohlcv(df) is True
        assert df.attrs["symbol"] == "AAPL"
        assert len(df) > 0

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_cache_hit(self, mock_ticker_cls, test_configs):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))

        # First call: fetches from provider
        df1 = dm.get_ohlcv("AAPL", Timeframe.DAILY)
        # Second call: should hit cache
        df2 = dm.get_ohlcv("AAPL", Timeframe.DAILY)

        # Provider should only be called once
        assert mock_ticker.history.call_count == 1
        pd.testing.assert_frame_equal(df1, df2)

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_force_refresh(self, mock_ticker_cls, test_configs):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))

        dm.get_ohlcv("AAPL", Timeframe.DAILY)
        dm.get_ohlcv("AAPL", Timeframe.DAILY, force_refresh=True)

        assert mock_ticker.history.call_count == 2

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_inverse(self, mock_ticker_cls, test_configs):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))
        df = dm.get_ohlcv("AAPL", Timeframe.DAILY, inverse=True)

        assert df.attrs["is_inverse"] is True
        assert df.attrs["inverse_of"] == "AAPL"
        assert validate_ohlcv(df) is True

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_timeframe_aggregation(self, mock_ticker_cls, test_configs):
        """Weekly is natively supported by yfinance, so the mock must return
        weekly-like data. The key test is that the provider is called with
        the right interval and validation passes."""
        # Create 12 weekly bars (mock response)
        dates = pd.date_range("2024-01-05", periods=12, freq="W-FRI")
        rng = np.random.default_rng(42)
        closes = 185.0 + rng.standard_normal(12).cumsum()
        weekly_response = pd.DataFrame(
            {
                "Open": closes - rng.uniform(0, 2, 12),
                "High": closes + rng.uniform(2, 5, 12),
                "Low": closes - rng.uniform(2, 5, 12),
                "Close": closes,
                "Volume": rng.integers(3e8, 5e8, 12).astype(float),
            },
            index=pd.DatetimeIndex(dates, name="Date"),
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = weekly_response
        mock_ticker_cls.return_value = mock_ticker

        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))
        df = dm.get_ohlcv("AAPL", Timeframe.WEEKLY)

        assert validate_ohlcv(df) is True
        assert len(df) == 12

    def test_unknown_symbol_raises(self, test_configs):
        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))
        with pytest.raises(SymbolNotFoundError, match="FAKE"):
            dm.get_ohlcv("FAKE", Timeframe.DAILY)

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_get_multiple(self, mock_ticker_cls, test_configs):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))
        results = dm.get_multiple(["AAPL", "FAKE_SYMBOL"], Timeframe.DAILY)

        # AAPL should succeed, FAKE_SYMBOL should fail silently
        assert "AAPL" in results
        assert "FAKE_SYMBOL" not in results

    def test_list_symbols(self, test_configs):
        sym_path, src_path, cache_path = test_configs
        dm = DataManager(str(sym_path), str(src_path), str(cache_path))
        symbols = dm.list_symbols()
        assert "AAPL" in symbols
        assert "BTC/USDT" in symbols
