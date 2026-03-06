"""Schwab API data provider — stub for future implementation."""

from trade_analysis.config.loader import DataSourceConfig
from trade_analysis.models.ohlcv import Timeframe
from trade_analysis.providers.base import DataProvider


class SchwabProvider(DataProvider):
    """Placeholder for Schwab Trader API integration.

    Per PRD: "Migrate to Schwab API only when the live runner needs quotes
    from the same source it executes on."

    Implementation deferred to M8/M9.
    """

    def __init__(self, config: DataSourceConfig | None = None):
        self._config = config

    @property
    def name(self) -> str:
        return "schwab"

    @property
    def supported_asset_classes(self) -> list[str]:
        return ["stock", "etf", "index"]

    def fetch_ohlcv(self, symbol, timeframe, start=None, end=None):
        raise NotImplementedError(
            "Schwab provider not yet implemented. "
            "Use yfinance for stocks/ETFs/indices in M1-M7."
        )

    def get_supported_timeframes(self):
        return [Timeframe.DAILY, Timeframe.WEEKLY, Timeframe.MONTHLY]
