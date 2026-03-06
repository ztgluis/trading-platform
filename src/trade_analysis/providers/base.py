"""Abstract base class for all data providers."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps

import pandas as pd

from trade_analysis.exceptions import ProviderConnectionError
from trade_analysis.models.ohlcv import Timeframe

logger = logging.getLogger(__name__)


def with_retry(max_retries: int = 3, base_delay: float = 2.0):
    """Decorator that retries on ProviderConnectionError with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ProviderConnectionError as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
            raise last_error

        return wrapper

    return decorator


class DataProvider(ABC):
    """Base class that all data providers must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string."""
        ...

    @property
    @abstractmethod
    def supported_asset_classes(self) -> list[str]:
        """List of asset classes this provider can serve."""
        ...

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch raw OHLCV data from the provider.

        Returns data in the provider's native format.
        Normalization happens in the transform layer.
        """
        ...

    @abstractmethod
    def get_supported_timeframes(self) -> list[Timeframe]:
        """Return timeframes this provider natively supports."""
        ...

    def health_check(self) -> bool:
        """Verify provider connectivity. Default returns True."""
        return True
