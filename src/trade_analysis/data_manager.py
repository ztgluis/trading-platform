"""High-level data manager: the single entry point for all OHLCV data requests."""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from trade_analysis.cache.parquet_cache import ParquetCache
from trade_analysis.config.loader import (
    DataSourceConfig,
    SymbolConfig,
    load_cache_config,
    load_data_sources,
    load_symbols,
)
from trade_analysis.exceptions import ProviderError, SymbolNotFoundError
from trade_analysis.models.ohlcv import AssetClass, Timeframe, validate_ohlcv
from trade_analysis.providers.base import DataProvider
from trade_analysis.providers.ccxt_provider import CCXTProvider
from trade_analysis.providers.schwab_provider import SchwabProvider
from trade_analysis.providers.yfinance_provider import YFinanceProvider
from trade_analysis.transforms.inverse import compute_inverse
from trade_analysis.transforms.normalize import normalize_ccxt, normalize_yfinance
from trade_analysis.transforms.timeframe import (
    aggregate_timeframe,
    get_source_timeframe,
    needs_aggregation,
)

logger = logging.getLogger(__name__)

# Provider registry: maps provider name → (class, normalizer)
PROVIDER_REGISTRY: dict[str, tuple[type[DataProvider], callable]] = {
    "yfinance": (YFinanceProvider, normalize_yfinance),
    "ccxt": (CCXTProvider, normalize_ccxt),
    "schwab": (SchwabProvider, None),
}


class DataManager:
    """Orchestrates data fetching, caching, normalization, and transformation.

    Usage:
        dm = DataManager()
        df = dm.get_ohlcv("AAPL", Timeframe.DAILY)
        df_inv = dm.get_ohlcv("AAPL", Timeframe.DAILY, inverse=True)
    """

    def __init__(
        self,
        symbols_path: str | None = None,
        sources_path: str | None = None,
        cache_path: str | None = None,
    ):
        # Load configs
        sym_path = Path(symbols_path) if symbols_path else Path("config/symbols.yaml")
        src_path = Path(sources_path) if sources_path else Path("config/data_sources.yaml")
        cch_path = Path(cache_path) if cache_path else Path("config/cache.yaml")

        self._symbols = {s.ticker: s for s in load_symbols(sym_path)}
        self._source_configs = load_data_sources(src_path)
        cache_config = load_cache_config(cch_path)

        # Initialize cache
        self._cache = ParquetCache(
            storage_path=cache_config.storage_path,
            ttl_seconds=cache_config.ttl_seconds,
            max_age_days=cache_config.max_age_days,
        )

        # Initialize providers (lazily, one per provider name)
        self._providers: dict[str, DataProvider] = {}

    def _get_provider(self, provider_name: str, symbol_config: SymbolConfig) -> DataProvider:
        """Get or create a provider instance."""
        cache_key = provider_name
        if provider_name == "ccxt" and symbol_config.exchange:
            cache_key = f"ccxt_{symbol_config.exchange}"

        if cache_key not in self._providers:
            config = self._source_configs.get(provider_name)
            provider_cls, _ = PROVIDER_REGISTRY[provider_name]

            if provider_name == "ccxt":
                exchange = symbol_config.exchange or config.extra.get("default_exchange", "binance")
                self._providers[cache_key] = provider_cls(config=config, exchange_id=exchange)
            else:
                self._providers[cache_key] = provider_cls(config=config)

        return self._providers[cache_key]

    def _get_normalizer(self, provider_name: str) -> callable:
        """Get the normalizer function for a provider."""
        _, normalizer = PROVIDER_REGISTRY[provider_name]
        if normalizer is None:
            raise NotImplementedError(f"No normalizer for provider: {provider_name}")
        return normalizer

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
        inverse: bool = False,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Main entry point. Returns a validated, canonical OHLCV DataFrame.

        Pipeline:
        1. Check cache (skip if force_refresh)
        2. Determine provider from symbol config
        3. If timeframe needs aggregation, fetch the source timeframe
        4. Fetch from provider
        5. Normalize to canonical schema
        6. Aggregate timeframe if needed
        7. Validate
        8. Write to cache
        9. If inverse=True, compute inverse series
        10. Return
        """
        symbol_config = self._symbols.get(symbol)
        if symbol_config is None:
            raise SymbolNotFoundError(
                f"Symbol '{symbol}' not found in config. "
                f"Available: {list(self._symbols.keys())}"
            )

        provider_name = symbol_config.provider
        asset_class = AssetClass(symbol_config.asset_class)

        # Determine if we need timeframe aggregation
        provider = self._get_provider(provider_name, symbol_config)
        do_aggregate = needs_aggregation(timeframe, provider.get_supported_timeframes())
        fetch_timeframe = get_source_timeframe(timeframe) if do_aggregate else timeframe

        # Step 1: Check cache
        if not force_refresh:
            cached = self._cache.get(symbol, fetch_timeframe, provider_name, start, end)
            if cached is not None:
                logger.info(f"Cache hit for {symbol}/{fetch_timeframe.value}")
                if do_aggregate:
                    cached = aggregate_timeframe(
                        cached, timeframe,
                        week_end_day="SUN" if asset_class == AssetClass.CRYPTO else "FRI",
                    )
                if inverse:
                    cached = compute_inverse(cached)
                return cached

        # Steps 2-4: Fetch from provider
        logger.info(f"Fetching {symbol}/{fetch_timeframe.value} from {provider_name}")
        raw = provider.fetch_ohlcv(symbol, fetch_timeframe, start, end)

        # Step 5: Normalize
        normalizer = self._get_normalizer(provider_name)
        df = normalizer(raw, symbol, fetch_timeframe, asset_class)

        # Step 7: Validate (normalizer already validates, but double-check)
        validate_ohlcv(df)

        # Step 8: Write to cache
        self._cache.put(df, symbol, fetch_timeframe, provider_name)

        # Step 6: Aggregate if needed
        if do_aggregate:
            df = aggregate_timeframe(
                df, timeframe,
                week_end_day="SUN" if asset_class == AssetClass.CRYPTO else "FRI",
            )

        # Step 9: Inverse
        if inverse:
            df = compute_inverse(df)

        return df

    def get_multiple(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple symbols. Continues on individual failures."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_ohlcv(symbol, timeframe, start, end)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        return results

    def refresh_cache(self, symbol: str | None = None) -> None:
        """Force re-fetch and re-cache."""
        if symbol:
            sym_config = self._symbols.get(symbol)
            if sym_config:
                for tf_str in sym_config.timeframes:
                    tf = Timeframe(tf_str)
                    self.get_ohlcv(symbol, tf, force_refresh=True)
        else:
            for sym, config in self._symbols.items():
                for tf_str in config.timeframes:
                    try:
                        tf = Timeframe(tf_str)
                        self.get_ohlcv(sym, tf, force_refresh=True)
                    except Exception as e:
                        logger.warning(f"Failed to refresh {sym}/{tf_str}: {e}")

    def list_cached(self) -> list[dict]:
        """Return summary of all cached data."""
        return self._cache.list_cached()

    def list_symbols(self) -> list[str]:
        """Return all configured symbols."""
        return list(self._symbols.keys())
