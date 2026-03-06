"""Tests for yfinance provider with mocked API calls."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from trade_analysis.exceptions import ProviderConnectionError, SymbolNotFoundError
from trade_analysis.models.ohlcv import Timeframe
from trade_analysis.providers.yfinance_provider import YFinanceProvider


def make_yfinance_response(rows: int = 5) -> pd.DataFrame:
    """Create a mock yfinance history() response."""
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


class TestYFinanceProvider:
    def test_name(self):
        provider = YFinanceProvider()
        assert provider.name == "yfinance"

    def test_supported_asset_classes(self):
        provider = YFinanceProvider()
        assert "stock" in provider.supported_asset_classes
        assert "crypto" not in provider.supported_asset_classes

    def test_supported_timeframes(self):
        provider = YFinanceProvider()
        tfs = provider.get_supported_timeframes()
        assert Timeframe.DAILY in tfs
        assert Timeframe.H1 in tfs
        # 4H is not natively supported — needs aggregation
        assert Timeframe.H4 not in tfs

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_fetch_daily(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        provider = YFinanceProvider()
        result = provider.fetch_ohlcv("AAPL", Timeframe.DAILY)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        mock_ticker.history.assert_called_once()
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs["interval"] == "1d"

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_fetch_4h_uses_1h_interval(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response(40)
        mock_ticker_cls.return_value = mock_ticker

        provider = YFinanceProvider()
        result = provider.fetch_ohlcv("AAPL", Timeframe.H4)

        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs["interval"] == "1h"

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_empty_response_raises(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        provider = YFinanceProvider()
        with pytest.raises(SymbolNotFoundError):
            provider.fetch_ohlcv("FAKE_SYMBOL", Timeframe.DAILY)

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_network_error_raises(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = ConnectionError("timeout")
        mock_ticker_cls.return_value = mock_ticker

        provider = YFinanceProvider()
        with pytest.raises(ProviderConnectionError):
            provider.fetch_ohlcv("AAPL", Timeframe.DAILY)

    @patch("trade_analysis.providers.yfinance_provider.yf.Ticker")
    def test_fetch_with_date_range(self, mock_ticker_cls):
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = make_yfinance_response()
        mock_ticker_cls.return_value = mock_ticker

        provider = YFinanceProvider()
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        provider.fetch_ohlcv("AAPL", Timeframe.DAILY, start=start, end=end)

        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs["start"] == "2024-01-01"
        assert call_kwargs["end"] == "2024-06-01"
