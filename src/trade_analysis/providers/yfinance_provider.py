"""yfinance data provider for stocks, ETFs, indices, and metals."""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from trade_analysis.config.loader import DataSourceConfig
from trade_analysis.exceptions import ProviderConnectionError, SymbolNotFoundError
from trade_analysis.models.ohlcv import Timeframe
from trade_analysis.providers.base import DataProvider, with_retry

logger = logging.getLogger(__name__)

# yfinance interval mapping
TIMEFRAME_MAP: dict[Timeframe, str] = {
    Timeframe.H1: "1h",
    Timeframe.H4: "1h",  # Fetch 1h, aggregate to 4h in transform layer
    Timeframe.DAILY: "1d",
    Timeframe.WEEKLY: "1wk",
    Timeframe.MONTHLY: "1mo",
}


class YFinanceProvider(DataProvider):

    def __init__(self, config: DataSourceConfig | None = None):
        self._config = config
        self._max_intraday_days = 729
        if config and config.extra.get("max_intraday_history_days"):
            self._max_intraday_days = config.extra["max_intraday_history_days"]

    @property
    def name(self) -> str:
        return "yfinance"

    @property
    def supported_asset_classes(self) -> list[str]:
        return ["stock", "etf", "index", "metal"]

    @with_retry(max_retries=3, base_delay=2.0)
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch from yfinance.

        Important constraints:
        - Intraday data (1h) limited to last ~730 days
        - 4H not natively available; fetches 1H for later aggregation
        """
        interval = TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")

        # For intraday, enforce max history limit
        if interval in ("1h", "2m", "5m", "15m", "30m", "60m", "90m"):
            max_start = datetime.now(timezone.utc) - timedelta(
                days=self._max_intraday_days
            )
            if start is None or start < max_start:
                start = max_start
                logger.info(
                    f"Clamping intraday start to {start.date()} "
                    f"(max {self._max_intraday_days} days)"
                )

        try:
            ticker = yf.Ticker(symbol)
            kwargs = {"interval": interval}
            if start:
                kwargs["start"] = start.strftime("%Y-%m-%d")
            if end:
                kwargs["end"] = end.strftime("%Y-%m-%d")
            if not start and not end:
                kwargs["period"] = "max"

            df = ticker.history(**kwargs)
        except Exception as e:
            raise ProviderConnectionError(
                f"Failed to fetch {symbol} from yfinance: {e}"
            ) from e

        if df is None or df.empty:
            raise SymbolNotFoundError(
                f"No data returned for {symbol} with interval={interval}"
            )

        logger.info(
            f"Fetched {len(df)} bars for {symbol} "
            f"({interval}, {df.index[0]} to {df.index[-1]})"
        )
        return df

    def get_supported_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1, Timeframe.DAILY, Timeframe.WEEKLY, Timeframe.MONTHLY]

    def health_check(self) -> bool:
        """Attempt to fetch 1 day of SPY as a connectivity check."""
        try:
            df = yf.Ticker("SPY").history(period="1d")
            return df is not None and not df.empty
        except Exception:
            return False
