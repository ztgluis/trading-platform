"""Signal handler — fetch OHLCV, generate signals, extract latest bar.

Core logic for processing a webhook alert:
  1. Fetch fresh OHLCV data via DataManager
  2. Run the signal engine
  3. Extract the latest bar's signal data
  4. Map to the signals table schema
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trade_analysis.config.loader import SymbolConfig, load_symbols
from trade_analysis.models.ohlcv import Timeframe
from trade_analysis.signals import (
    SignalEngineConfig,
    generate_signals,
    get_bucket_for_asset,
    load_signal_config,
)

logger = logging.getLogger(__name__)

_SIGNALS_YAML = Path(__file__).parents[3] / "config" / "signals.yaml"


# ---------------------------------------------------------------------------
# Symbol lookup
# ---------------------------------------------------------------------------


def lookup_symbol(ticker: str, symbols: list[SymbolConfig] | None = None) -> SymbolConfig | None:
    """Find a symbol config by ticker. Returns None if not found."""
    if symbols is None:
        symbols = load_symbols()
    for sym in symbols:
        if sym.ticker.upper() == ticker.upper():
            return sym
    return None


# ---------------------------------------------------------------------------
# Config hash
# ---------------------------------------------------------------------------


def compute_config_hash(path: Path | None = None) -> str:
    """SHA-256 hash of the signals.yaml file for reproducibility tracking."""
    config_path = path or _SIGNALS_YAML
    if not config_path.exists():
        return ""
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def extract_latest_signal(
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    timeframe: str,
    signal_config: SignalEngineConfig | None = None,
) -> dict[str, Any] | None:
    """Extract the latest bar's signal data and map to signals table schema.

    Returns None if the latest bar has no signal direction (no entry).
    """
    if df.empty:
        return None

    latest = df.iloc[-1]

    # Must have a signal direction
    direction = latest.get("signal_direction")
    if direction is None or (isinstance(direction, float) and pd.isna(direction)):
        return None

    # Resolve bucket
    config = signal_config or load_signal_config()
    bucket_config = get_bucket_for_asset(asset_class, config)
    bucket = "A" if asset_class.lower() in [c.lower() for c in config.bucket_a.asset_classes] else "B"

    # Determine which condition flags to read based on direction
    is_long = direction == "long"
    trend_confirmed = bool(latest.get("trend_bull" if is_long else "trend_bear", False))
    structure_confirmed = bool(latest.get("structure_bull" if is_long else "structure_bear", False))
    momentum_confirmed = bool(latest.get("momentum_bull" if is_long else "momentum_bear", False))

    # Build the signal row matching the signals table schema
    bar_ts = latest.get("timestamp") or latest.name
    if hasattr(bar_ts, "isoformat"):
        bar_ts = bar_ts.isoformat()

    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "bucket": bucket,
        "bar_timestamp": bar_ts,
        "regime": str(latest.get("regime", "transition")),
        "regime_distance_pct": _safe_float(latest.get("regime_distance_pct")),
        "regime_strongly_aligned": bool(latest.get("regime_strongly_aligned", False)),
        "signal_direction": str(direction),
        "conditions_met": int(latest.get("signal_conditions_met", 0)),
        "trend_confirmed": trend_confirmed,
        "structure_confirmed": structure_confirmed,
        "structure_multi_method": bool(latest.get("structure_multi_method", False)),
        "momentum_confirmed": momentum_confirmed,
        "volume_spike": bool(latest.get("volume_spike", False)),
        "signal_score": int(latest.get("signal_score", 0)),
        "signal_tradeable": bool(latest.get("signal_tradeable", False)),
        "entry_price": _safe_float(latest.get("entry_price") or latest.get("close")),
        "exit_stop": _safe_float(latest.get("exit_stop")),
        "exit_target": _safe_float(latest.get("exit_target")),
        "exit_trail_be": _safe_float(latest.get("exit_trail_be")),
        "exit_risk": _safe_float(latest.get("exit_risk")),
        "exit_reward": _safe_float(latest.get("exit_reward")),
        "exit_rr_ratio": _safe_float(latest.get("exit_rr_ratio")),
        "config_hash": compute_config_hash(),
    }


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None for NaN or None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Full scan handler
# ---------------------------------------------------------------------------


def handle_scan(
    symbol: str,
    timeframe: str,
    asset_class: str,
    dm: Any,
    signal_config: SignalEngineConfig | None = None,
    force_refresh: bool = True,
) -> dict[str, Any] | None:
    """Run the full scan pipeline: fetch OHLCV → generate signals → extract.

    Returns the signal row dict for persistence, or None if no signal.
    """
    logger.info("Scanning %s %s (%s)", symbol, timeframe, asset_class)

    # Fetch OHLCV
    df = dm.get_ohlcv(symbol, Timeframe(timeframe), force_refresh=force_refresh)
    logger.info("Fetched %d bars for %s %s", len(df), symbol, timeframe)

    # Generate signals
    config = signal_config or load_signal_config()
    enriched = generate_signals(df, asset_class, config=config)
    logger.info("Generated signals for %s %s", symbol, timeframe)

    # Extract latest bar
    signal = extract_latest_signal(enriched, symbol, asset_class, timeframe, config)

    if signal:
        logger.info(
            "Signal: %s %s score=%d tradeable=%s",
            symbol,
            signal["signal_direction"],
            signal["signal_score"],
            signal["signal_tradeable"],
        )
    else:
        logger.info("No signal for %s %s", symbol, timeframe)

    return signal
