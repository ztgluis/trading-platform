"""Tests for CCXT provider with mocked API calls."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trade_analysis.exceptions import ProviderConnectionError, SymbolNotFoundError
from trade_analysis.models.ohlcv import Timeframe
from trade_analysis.providers.ccxt_provider import CCXTProvider


def make_ccxt_response(rows: int = 10) -> list[list]:
    """Create mock CCXT fetch_ohlcv response."""
    import pandas as pd

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


class TestCCXTProvider:
    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_name(self, mock_ccxt):
        mock_ccxt.binance = MagicMock
        provider = CCXTProvider(exchange_id="binance")
        assert provider.name == "ccxt"

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_supported_asset_classes(self, mock_ccxt):
        mock_ccxt.binance = MagicMock
        provider = CCXTProvider(exchange_id="binance")
        assert provider.supported_asset_classes == ["crypto"]

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_supported_timeframes(self, mock_ccxt):
        mock_ccxt.binance = MagicMock
        provider = CCXTProvider(exchange_id="binance")
        tfs = provider.get_supported_timeframes()
        assert Timeframe.H4 in tfs
        assert Timeframe.DAILY in tfs

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_fetch_returns_data(self, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = make_ccxt_response(10)
        mock_ccxt.binance.return_value = mock_exchange

        provider = CCXTProvider(exchange_id="binance")
        result = provider.fetch_ohlcv("BTC/USDT", Timeframe.H1)

        assert isinstance(result, list)
        assert len(result) == 10
        mock_exchange.fetch_ohlcv.assert_called_once()

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_pagination(self, mock_ccxt):
        """Test that pagination fetches multiple pages."""
        page1 = make_ccxt_response(5)
        page2 = make_ccxt_response(3)
        # Make page2 timestamps after page1
        for i, row in enumerate(page2):
            row[0] = page1[-1][0] + (i + 1) * 3600_000

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = [page1, page2]
        mock_ccxt.binance.return_value = mock_exchange

        provider = CCXTProvider(exchange_id="binance")
        provider._max_candles = 5  # Force pagination
        result = provider.fetch_ohlcv("BTC/USDT", Timeframe.H1)

        assert len(result) == 8
        assert mock_exchange.fetch_ohlcv.call_count == 2

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_empty_response_raises(self, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []
        mock_ccxt.binance.return_value = mock_exchange

        provider = CCXTProvider(exchange_id="binance")
        with pytest.raises(SymbolNotFoundError):
            provider.fetch_ohlcv("FAKE/USDT", Timeframe.DAILY)

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_bad_symbol_raises(self, mock_ccxt):
        import ccxt as real_ccxt

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = real_ccxt.BadSymbol("bad")
        mock_ccxt.binance.return_value = mock_exchange
        mock_ccxt.BadSymbol = real_ccxt.BadSymbol
        mock_ccxt.NetworkError = real_ccxt.NetworkError
        mock_ccxt.ExchangeNotAvailable = real_ccxt.ExchangeNotAvailable
        mock_ccxt.ExchangeError = real_ccxt.ExchangeError

        provider = CCXTProvider(exchange_id="binance")
        with pytest.raises(SymbolNotFoundError):
            provider.fetch_ohlcv("INVALID/PAIR", Timeframe.DAILY)

    @patch("trade_analysis.providers.ccxt_provider.ccxt")
    def test_network_error_raises(self, mock_ccxt):
        import ccxt as real_ccxt

        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = real_ccxt.NetworkError("timeout")
        mock_ccxt.binance.return_value = mock_exchange
        mock_ccxt.BadSymbol = real_ccxt.BadSymbol
        mock_ccxt.NetworkError = real_ccxt.NetworkError
        mock_ccxt.ExchangeNotAvailable = real_ccxt.ExchangeNotAvailable
        mock_ccxt.ExchangeError = real_ccxt.ExchangeError

        provider = CCXTProvider(exchange_id="binance")
        with pytest.raises(ProviderConnectionError):
            provider.fetch_ohlcv("BTC/USDT", Timeframe.DAILY)
