"""Parquet-based OHLCV cache with TTL management."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trade_analysis.exceptions import CacheError
from trade_analysis.models.ohlcv import Timeframe

logger = logging.getLogger(__name__)


class ParquetCache:
    """File-based cache using Parquet format.

    File layout:
        {base_path}/{provider}/{symbol}/{timeframe}.parquet
        {base_path}/{provider}/{symbol}/{timeframe}.meta.json
    """

    def __init__(self, storage_path: Path, ttl_seconds: dict[str, int] | None = None,
                 max_age_days: int = 90):
        self._base_path = Path(storage_path)
        self._ttl = ttl_seconds or {}
        self._max_age_days = max_age_days

    def get(
        self,
        symbol: str,
        timeframe: Timeframe,
        provider: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Return cached data if it exists and is not expired.

        Returns None if no cache exists or TTL is expired.
        """
        cache_path = self._cache_path(provider, symbol, timeframe)
        meta_path = self._meta_path(provider, symbol, timeframe)

        if not cache_path.exists():
            return None

        # Check TTL
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                last_fetch = datetime.fromisoformat(meta["last_fetch"])
                ttl = self._ttl.get(timeframe.value, 86400)
                age = (datetime.now(timezone.utc) - last_fetch).total_seconds()
                if age > ttl:
                    logger.debug(
                        f"Cache expired for {symbol}/{timeframe.value} "
                        f"(age={age:.0f}s, ttl={ttl}s)"
                    )
                    return None
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupt meta file {meta_path}: {e}")
                return None

        # Read parquet
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to read cache {cache_path}: {e}")
            self._safe_delete(cache_path)
            self._safe_delete(meta_path)
            return None

        # Filter date range if requested
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            df = df[df["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
            df = df[df["timestamp"] <= end_ts]

        df = df.reset_index(drop=True)
        logger.debug(f"Cache hit for {symbol}/{timeframe.value}: {len(df)} rows")
        return df

    def put(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        provider: str,
    ) -> None:
        """Write DataFrame to cache. Merges with existing data if present."""
        cache_path = self._cache_path(provider, symbol, timeframe)
        meta_path = self._meta_path(provider, symbol, timeframe)

        # Merge with existing cache if present
        if cache_path.exists():
            try:
                existing = pd.read_parquet(cache_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"], keep="last")
                df = df.sort_values("timestamp").reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Failed to merge with existing cache: {e}")

        # Ensure directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet
        try:
            df.to_parquet(cache_path, index=False)
        except Exception as e:
            raise CacheError(f"Failed to write cache {cache_path}: {e}") from e

        # Write metadata
        meta = {
            "last_fetch": datetime.now(timezone.utc).isoformat(),
            "rows": len(df),
            "symbol": symbol,
            "timeframe": timeframe.value,
            "provider": provider,
        }
        try:
            meta_path.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.warning(f"Failed to write meta {meta_path}: {e}")

    def invalidate(
        self,
        symbol: str | None = None,
        timeframe: Timeframe | None = None,
        provider: str | None = None,
    ) -> int:
        """Remove cache entries matching filters. Returns count of files removed."""
        count = 0
        for cache_file in self._base_path.rglob("*.parquet"):
            parts = cache_file.relative_to(self._base_path).parts
            if len(parts) != 3:
                continue
            file_provider, file_symbol, file_tf = parts
            file_tf = file_tf.replace(".parquet", "")

            if provider and file_provider != provider:
                continue
            if symbol and file_symbol != self._sanitize_symbol(symbol):
                continue
            if timeframe and file_tf != timeframe.value:
                continue

            self._safe_delete(cache_file)
            self._safe_delete(cache_file.with_suffix(".meta.json"))
            count += 1

        return count

    def cleanup_expired(self) -> int:
        """Remove all cache entries older than max_age_days."""
        count = 0
        cutoff = datetime.now(timezone.utc)

        for meta_file in self._base_path.rglob("*.meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
                last_fetch = datetime.fromisoformat(meta["last_fetch"])
                age_days = (cutoff - last_fetch).days
                if age_days > self._max_age_days:
                    parquet_file = meta_file.with_suffix(".parquet")
                    self._safe_delete(parquet_file)
                    self._safe_delete(meta_file)
                    count += 1
            except (json.JSONDecodeError, KeyError):
                continue

        return count

    def list_cached(self) -> list[dict]:
        """Return summary of all cached data."""
        entries = []
        for meta_file in self._base_path.rglob("*.meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
                entries.append(meta)
            except (json.JSONDecodeError, KeyError):
                continue
        return entries

    def _cache_path(self, provider: str, symbol: str, timeframe: Timeframe) -> Path:
        safe_symbol = self._sanitize_symbol(symbol)
        return self._base_path / provider / safe_symbol / f"{timeframe.value}.parquet"

    def _meta_path(self, provider: str, symbol: str, timeframe: Timeframe) -> Path:
        return self._cache_path(provider, symbol, timeframe).with_suffix(".meta.json")

    @staticmethod
    def _sanitize_symbol(symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_").replace("=", "_")

    @staticmethod
    def _safe_delete(path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete {path}: {e}")
