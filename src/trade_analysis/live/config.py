"""Live runner configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for order execution (M9)."""

    dry_run: bool = True
    max_positions: int = 5
    max_per_symbol: int = 1
    proposal_expiry_hours: int = 24
    default_order_type: str = "market"


@dataclass(frozen=True)
class LiveConfig:
    """Configuration for the live webhook runner."""

    webhook_secret: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    persist_all_signals: bool = False
    force_refresh_ohlcv: bool = True
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


_DEFAULT_PATH = Path(__file__).parents[3] / "config" / "live.yaml"


def load_live_config(path: Path | None = None) -> LiveConfig:
    """Load live config from YAML, resolving env var references."""
    config_path = path or _DEFAULT_PATH

    if not config_path.exists():
        return LiveConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    live = raw.get("live", {})

    # Resolve webhook secret from env var
    secret_ref = live.get("webhook_secret_env_var", "")
    webhook_secret = ""
    if secret_ref.startswith("${") and secret_ref.endswith("}"):
        env_name = secret_ref[2:-1]
        webhook_secret = os.environ.get(env_name, "")
    else:
        webhook_secret = secret_ref

    # Port: prefer PORT env var (Railway sets this)
    port = int(os.environ.get("PORT", live.get("port", 8000)))

    # Parse execution config
    exec_raw = live.get("execution", {})
    execution = ExecutionConfig(
        dry_run=exec_raw.get("dry_run", True),
        max_positions=exec_raw.get("max_positions", 5),
        max_per_symbol=exec_raw.get("max_per_symbol", 1),
        proposal_expiry_hours=exec_raw.get("proposal_expiry_hours", 24),
        default_order_type=exec_raw.get("default_order_type", "market"),
    )

    return LiveConfig(
        webhook_secret=webhook_secret,
        host=live.get("host", "0.0.0.0"),
        port=port,
        log_level=live.get("log_level", "info"),
        persist_all_signals=live.get("persist_all_signals", False),
        force_refresh_ohlcv=live.get("force_refresh_ohlcv", True),
        execution=execution,
    )
