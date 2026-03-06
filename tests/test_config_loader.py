"""Tests for config loading and env var resolution."""

import os
from pathlib import Path

import pytest
import yaml

from trade_analysis.config.loader import (
    CacheConfig,
    DataSourceConfig,
    SymbolConfig,
    load_cache_config,
    load_data_sources,
    load_symbols,
    resolve_env_vars,
)
from trade_analysis.exceptions import ConfigError


class TestResolveEnvVars:
    def test_resolves_existing_var(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "my_secret")
        assert resolve_env_vars("${TEST_KEY}") == "my_secret"

    def test_resolves_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        assert resolve_env_vars("${HOST}:${PORT}") == "localhost:5432"

    def test_raises_on_missing_var(self):
        # Ensure the var definitely doesn't exist
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        with pytest.raises(ConfigError, match="NONEXISTENT_VAR_XYZ"):
            resolve_env_vars("${NONEXISTENT_VAR_XYZ}")

    def test_returns_string_without_pattern_unchanged(self):
        assert resolve_env_vars("plain_string") == "plain_string"

    def test_returns_non_string_unchanged(self):
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(None) is None


class TestLoadSymbols:
    def test_loads_project_symbols(self):
        symbols = load_symbols(Path("config/symbols.yaml"))
        assert len(symbols) > 0
        assert all(isinstance(s, SymbolConfig) for s in symbols)

    def test_aapl_config(self):
        symbols = load_symbols(Path("config/symbols.yaml"))
        aapl = next(s for s in symbols if s.ticker == "AAPL")
        assert aapl.asset_class == "stock"
        assert aapl.provider == "yfinance"
        assert "Daily" in aapl.timeframes

    def test_crypto_has_exchange(self):
        symbols = load_symbols(Path("config/symbols.yaml"))
        btc = next(s for s in symbols if s.ticker == "BTC/USDT")
        assert btc.exchange == "binance"
        assert btc.provider == "ccxt"

    def test_etf_has_inverse_ticker(self):
        symbols = load_symbols(Path("config/symbols.yaml"))
        qqq = next(s for s in symbols if s.ticker == "QQQ")
        assert qqq.inverse_ticker == "SQQQ"

    def test_missing_file_raises(self):
        with pytest.raises(ConfigError, match="not found"):
            load_symbols(Path("nonexistent.yaml"))

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(ConfigError, match="empty"):
            load_symbols(empty)

    def test_missing_required_field_raises(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(yaml.dump({
            "symbols": [{"ticker": "AAPL"}]  # missing asset_class, provider, timeframes
        }))
        with pytest.raises(ConfigError, match="Missing required field"):
            load_symbols(bad_yaml)


class TestLoadDataSources:
    def test_loads_project_config(self):
        configs = load_data_sources(
            Path("config/data_sources.yaml"),
            resolve_secrets=False,
        )
        assert "yfinance" in configs
        assert "ccxt" in configs
        assert "schwab" in configs

    def test_yfinance_no_auth(self):
        configs = load_data_sources(
            Path("config/data_sources.yaml"),
            resolve_secrets=False,
        )
        yf = configs["yfinance"]
        assert yf.api_key is None
        assert yf.retry_count == 3

    def test_extra_settings_captured(self):
        configs = load_data_sources(
            Path("config/data_sources.yaml"),
            resolve_secrets=False,
        )
        ccxt_config = configs["ccxt"]
        assert ccxt_config.extra.get("default_exchange") == "binance"
        assert ccxt_config.extra.get("max_candles_per_request") == 1000

    def test_resolves_env_vars_when_set(self, monkeypatch):
        monkeypatch.setenv("CCXT_API_KEY", "test_key")
        monkeypatch.setenv("CCXT_API_SECRET", "test_secret")
        configs = load_data_sources(Path("config/data_sources.yaml"))
        assert configs["ccxt"].api_key == "test_key"
        assert configs["ccxt"].api_secret == "test_secret"

    def test_missing_env_var_returns_none(self):
        # When env vars aren't set, api_key/secret should be None (graceful)
        os.environ.pop("CCXT_API_KEY", None)
        os.environ.pop("CCXT_API_SECRET", None)
        configs = load_data_sources(Path("config/data_sources.yaml"))
        assert configs["ccxt"].api_key is None


class TestLoadCacheConfig:
    def test_loads_project_config(self):
        config = load_cache_config(Path("config/cache.yaml"))
        assert isinstance(config, CacheConfig)
        assert config.storage_path == Path("data/cache")
        assert config.max_age_days == 90

    def test_ttl_values(self):
        config = load_cache_config(Path("config/cache.yaml"))
        assert config.ttl_seconds["Daily"] == 43200
        assert config.ttl_seconds["4H"] == 3600
