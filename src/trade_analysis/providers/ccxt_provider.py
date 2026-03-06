"""CCXT data provider for cryptocurrency markets."""

import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd

from trade_analysis.config.loader import DataSourceConfig
from trade_analysis.exceptions import ProviderConnectionError, SymbolNotFoundError
from trade_analysis.models.ohlcv import Timeframe
from trade_analysis.providers.base import DataProvider, with_retry

logger = logging.getLogger(__name__)

# CCXT timeframe mapping
TIMEFRAME_MAP: dict[Timeframe, str] = {
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.DAILY: "1d",
    Timeframe.WEEKLY: "1w",
    Timeframe.MONTHLY: "1M",
}


class CCXTProvider(DataProvider):

    def __init__(self, config: DataSourceConfig | None = None,
                 exchange_id: str = "binance"):
        self._config = config
        self._exchange_id = exchange_id
        self._max_candles = 1000
        if config and config.extra.get("max_candles_per_request"):
            self._max_candles = config.extra["max_candles_per_request"]

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange_id}")

        exchange_params = {"enableRateLimit": True}
        if config and config.api_key:
            exchange_params["apiKey"] = config.api_key
        if config and config.api_secret:
            exchange_params["secret"] = config.api_secret

        self._exchange = exchange_class(exchange_params)

    @property
    def name(self) -> str:
        return "ccxt"

    @property
    def supported_asset_classes(self) -> list[str]:
        return ["crypto"]

    @with_retry(max_retries=3, base_delay=5.0)
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[list]:
        """Fetch from CCXT with pagination for large date ranges.

        Returns list of [timestamp_ms, O, H, L, C, V].
        """
        tf_str = TIMEFRAME_MAP.get(timeframe)
        if tf_str is None:
            raise ValueError(f"Unsupported timeframe for CCXT: {timeframe}")

        since = None
        if start:
            since = int(start.timestamp() * 1000)

        end_ms = None
        if end:
            end_ms = int(end.timestamp() * 1000)

        all_data = []
        try:
            while True:
                data = self._exchange.fetch_ohlcv(
                    symbol, tf_str, since=since, limit=self._max_candles
                )
                if not data:
                    break

                # Filter out data beyond end date
                if end_ms:
                    data = [d for d in data if d[0] <= end_ms]

                all_data.extend(data)

                if len(data) < self._max_candles:
                    break

                # Move since to after the last candle
                since = data[-1][0] + 1

                if end_ms and since > end_ms:
                    break

        except ccxt.BadSymbol as e:
            raise SymbolNotFoundError(f"Symbol {symbol} not found: {e}") from e
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            raise ProviderConnectionError(
                f"Failed to fetch {symbol} from {self._exchange_id}: {e}"
            ) from e
        except ccxt.ExchangeError as e:
            raise ProviderConnectionError(
                f"Exchange error fetching {symbol}: {e}"
            ) from e

        if not all_data:
            raise SymbolNotFoundError(
                f"No data returned for {symbol} on {self._exchange_id}"
            )

        # Deduplicate by timestamp (pagination overlap)
        seen = set()
        unique_data = []
        for row in all_data:
            if row[0] not in seen:
                seen.add(row[0])
                unique_data.append(row)

        logger.info(
            f"Fetched {len(unique_data)} bars for {symbol} ({tf_str})"
        )
        return unique_data

    def get_supported_timeframes(self) -> list[Timeframe]:
        return [
            Timeframe.H1, Timeframe.H4, Timeframe.DAILY,
            Timeframe.WEEKLY, Timeframe.MONTHLY,
        ]

    def health_check(self) -> bool:
        try:
            self._exchange.fetch_ticker("BTC/USDT")
            return True
        except Exception:
            return False
