"""Exception hierarchy for the trade analysis platform."""


class TradeAnalysisError(Exception):
    """Base exception for the entire platform."""


class ConfigError(TradeAnalysisError):
    """YAML parsing failure, missing required field, unresolved env var."""


class ProviderError(TradeAnalysisError):
    """Base for all data provider errors."""


class ProviderConnectionError(ProviderError):
    """Network failure, timeout, API unreachable."""


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class SymbolNotFoundError(ProviderError):
    """Symbol does not exist on this provider."""


class OHLCVValidationError(TradeAnalysisError):
    """DataFrame fails canonical schema validation."""


class CacheError(TradeAnalysisError):
    """Cache read/write failure."""


class BacktestError(TradeAnalysisError):
    """Backtester runtime error: invalid data, position logic failure."""


class ExecutionError(TradeAnalysisError):
    """Base for all order execution errors."""


class OrderRejectedError(ExecutionError):
    """Broker rejected the order (insufficient funds, invalid symbol, etc.)."""


class KillSwitchEngagedError(ExecutionError):
    """Action blocked because the kill switch is engaged."""


class PositionLimitError(ExecutionError):
    """At max open positions or per-symbol limit."""


class SchwabConnectionError(ExecutionError):
    """Cannot connect to Schwab API."""


class SchwabTokenExpiredError(SchwabConnectionError):
    """Schwab OAuth refresh token has expired (7-day limit)."""
