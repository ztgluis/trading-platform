"""Load YAML config files with environment variable resolution."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from trade_analysis.exceptions import ConfigError

# Load .env file if present
load_dotenv()

ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


@dataclass
class SymbolConfig:
    ticker: str
    asset_class: str
    provider: str
    timeframes: list[str]
    inverse_ticker: str | None = None
    exchange: str | None = None


@dataclass
class DataSourceConfig:
    provider_name: str
    rate_limit_calls_per_minute: int
    retry_count: int
    retry_delay_seconds: float
    api_key: str | None = None
    api_secret: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class CacheConfig:
    storage_path: Path
    ttl_seconds: dict[str, int]
    max_age_days: int


def resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with actual environment variable values.

    Raises ConfigError if a referenced env var is not set.
    Returns the original value if no pattern is found.
    """
    if not isinstance(value, str):
        return value

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ConfigError(f"Environment variable '{var_name}' is not set")
        return env_value

    return ENV_VAR_PATTERN.sub(replacer, value)


def _resolve_recursive(data):
    """Recursively resolve env vars in a nested data structure."""
    if isinstance(data, str):
        return resolve_env_vars(data)
    if isinstance(data, dict):
        return {k: _resolve_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_recursive(item) for item in data]
    return data


def _load_yaml(config_path: Path) -> dict:
    """Load and parse a YAML file."""
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML file {config_path}: {e}") from e
    if data is None:
        raise ConfigError(f"Config file is empty: {config_path}")
    return data


def load_symbols(config_path: Path = Path("config/symbols.yaml")) -> list[SymbolConfig]:
    """Load symbol universe from YAML."""
    data = _load_yaml(config_path)
    symbols_data = data.get("symbols")
    if not symbols_data:
        raise ConfigError("No 'symbols' key found in symbols config")

    symbols = []
    for item in symbols_data:
        try:
            symbols.append(SymbolConfig(
                ticker=item["ticker"],
                asset_class=item["asset_class"],
                provider=item["provider"],
                timeframes=item["timeframes"],
                inverse_ticker=item.get("inverse_ticker"),
                exchange=item.get("exchange"),
            ))
        except KeyError as e:
            raise ConfigError(f"Missing required field {e} in symbol config: {item}") from e

    return symbols


def load_data_sources(
    config_path: Path = Path("config/data_sources.yaml"),
    resolve_secrets: bool = True,
) -> dict[str, DataSourceConfig]:
    """Load provider configurations. Resolves env var references for API keys."""
    data = _load_yaml(config_path)
    providers_data = data.get("providers")
    if not providers_data:
        raise ConfigError("No 'providers' key found in data sources config")

    configs = {}
    for name, props in providers_data.items():
        api_key = None
        api_secret = None

        api_key_ref = props.get("api_key_env_var")
        api_secret_ref = props.get("api_secret_env_var")

        if resolve_secrets and api_key_ref:
            try:
                api_key = resolve_env_vars(api_key_ref)
            except ConfigError:
                api_key = None

        if resolve_secrets and api_secret_ref:
            try:
                api_secret = resolve_env_vars(api_secret_ref)
            except ConfigError:
                api_secret = None

        # Collect provider-specific extra settings
        known_keys = {
            "rate_limit_calls_per_minute", "retry_count", "retry_delay_seconds",
            "api_key_env_var", "api_secret_env_var",
        }
        extra = {k: v for k, v in props.items() if k not in known_keys}

        configs[name] = DataSourceConfig(
            provider_name=name,
            rate_limit_calls_per_minute=props.get("rate_limit_calls_per_minute", 60),
            retry_count=props.get("retry_count", 3),
            retry_delay_seconds=props.get("retry_delay_seconds", 2.0),
            api_key=api_key,
            api_secret=api_secret,
            extra=extra,
        )

    return configs


def load_cache_config(config_path: Path = Path("config/cache.yaml")) -> CacheConfig:
    """Load cache configuration."""
    data = _load_yaml(config_path)
    cache_data = data.get("cache")
    if not cache_data:
        raise ConfigError("No 'cache' key found in cache config")

    return CacheConfig(
        storage_path=Path(cache_data.get("storage_path", "data/cache")),
        ttl_seconds=cache_data.get("ttl_seconds", {}),
        max_age_days=cache_data.get("max_age_days", 90),
    )
