"""Signal Engine: config loader and main orchestrator.

Loads signal parameters from config/signals.yaml and provides the
generate_signals() pipeline that ties conditions, scoring, and exits together.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from trade_analysis.exceptions import ConfigError
from trade_analysis.indicators.trend import add_ma
from trade_analysis.indicators.volume import detect_volume_spike
from trade_analysis.signals.conditions import (
    evaluate_momentum_condition,
    evaluate_structure_condition,
    evaluate_trend_condition,
)
from trade_analysis.signals.exits import compute_exit_levels
from trade_analysis.signals.regime import detect_regime
from trade_analysis.signals.scoring import compute_signal_score, determine_signal_direction


# ---------------------------------------------------------------------------
# Config Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketConfig:
    """Configuration for a trading bucket (A or B)."""

    name: str
    asset_classes: list[str]
    primary_timeframe: str
    confirmation_timeframe: str
    trend_ma_type: str
    trend_ma_period: int
    max_hold_weeks: int | None
    target_r_multiple: float
    trail_breakeven_r: float


@dataclass(frozen=True)
class SignalEngineConfig:
    """Complete signal engine configuration loaded from YAML."""

    # Buckets
    bucket_a: BucketConfig
    bucket_b: BucketConfig

    # Regime
    regime_ma_type: str
    regime_ma_period: int
    regime_transition_closes: int
    regime_strong_alignment_pct: float

    # Structure condition
    swing_lookback: int
    level_proximity_pct: float
    pivot_lookback: int
    pivot_merge_distance_pct: float

    # Momentum condition
    rsi_period: int
    rsi_bull_threshold: float
    rsi_bear_threshold: float
    macd_fast: int
    macd_slow: int
    macd_signal: int

    # Volume
    volume_sma_period: int
    volume_spike_threshold: float

    # Scoring
    scoring_weights: dict[str, int] = field(default_factory=dict)
    tradeable_threshold: int = 3

    # Exits
    atr_period: int = 14
    stop_method: str = "swing"
    atr_stop_multiplier: float = 1.5


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------


def _parse_bucket(raw: dict, key: str) -> BucketConfig:
    """Parse a bucket config from raw YAML dict."""
    required = [
        "name",
        "asset_classes",
        "primary_timeframe",
        "confirmation_timeframe",
        "trend_ma_type",
        "trend_ma_period",
        "target_r_multiple",
        "trail_breakeven_r",
    ]
    for field_name in required:
        if field_name not in raw:
            raise ConfigError(
                f"Missing required field '{field_name}' in bucket '{key}'"
            )

    return BucketConfig(
        name=raw["name"],
        asset_classes=raw["asset_classes"],
        primary_timeframe=raw["primary_timeframe"],
        confirmation_timeframe=raw["confirmation_timeframe"],
        trend_ma_type=raw["trend_ma_type"],
        trend_ma_period=int(raw["trend_ma_period"]),
        max_hold_weeks=raw.get("max_hold_weeks"),
        target_r_multiple=float(raw["target_r_multiple"]),
        trail_breakeven_r=float(raw["trail_breakeven_r"]),
    )


def load_signal_config(
    config_path: Path | None = None,
) -> SignalEngineConfig:
    """Load signal engine configuration from YAML.

    Args:
        config_path: Path to signals.yaml. If None, uses default location.

    Returns:
        SignalEngineConfig dataclass.

    Raises:
        ConfigError: If config is missing, malformed, or missing required fields.
    """
    if config_path is None:
        config_path = Path("config/signals.yaml")

    if not config_path.exists():
        raise ConfigError(f"Signal config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw or "signals" not in raw:
        raise ConfigError("Signal config must have a top-level 'signals' key")

    signals = raw["signals"]

    # Parse buckets
    buckets = signals.get("buckets", {})
    if "A" not in buckets or "B" not in buckets:
        raise ConfigError("Signal config must define buckets 'A' and 'B'")

    bucket_a = _parse_bucket(buckets["A"], "A")
    bucket_b = _parse_bucket(buckets["B"], "B")

    # Parse regime
    regime = signals.get("regime", {})

    # Parse conditions
    conditions = signals.get("conditions", {})
    structure = conditions.get("structure", {})
    momentum = conditions.get("momentum", {})

    # Parse scoring
    scoring = signals.get("scoring", {})
    scoring_weights = {
        "trend_confirmed": scoring.get("trend_confirmed", 1),
        "structure_single_method": scoring.get("structure_single_method", 1),
        "structure_multi_method": scoring.get("structure_multi_method", 2),
        "momentum_confirmed": scoring.get("momentum_confirmed", 1),
        "regime_strongly_aligned": scoring.get("regime_strongly_aligned", 1),
        "volume_spike_on_entry": scoring.get("volume_spike_on_entry", 1),
    }

    # Parse volume
    volume = signals.get("volume", {})

    # Parse exits
    exits = signals.get("exits", {})

    return SignalEngineConfig(
        bucket_a=bucket_a,
        bucket_b=bucket_b,
        # Regime
        regime_ma_type=regime.get("ma_type", "sma"),
        regime_ma_period=int(regime.get("ma_period", 200)),
        regime_transition_closes=int(
            regime.get("transition_consecutive_closes", 3)
        ),
        regime_strong_alignment_pct=float(
            regime.get("strong_alignment_pct", 5.0)
        ),
        # Structure condition
        swing_lookback=int(structure.get("swing_lookback", 3)),
        level_proximity_pct=float(structure.get("level_proximity_pct", 3.0)),
        pivot_lookback=int(structure.get("pivot_lookback", 5)),
        pivot_merge_distance_pct=float(
            structure.get("pivot_merge_distance_pct", 0.5)
        ),
        # Momentum condition
        rsi_period=int(momentum.get("rsi_period", 14)),
        rsi_bull_threshold=float(momentum.get("rsi_bull_threshold", 50)),
        rsi_bear_threshold=float(momentum.get("rsi_bear_threshold", 50)),
        macd_fast=int(momentum.get("macd_fast", 12)),
        macd_slow=int(momentum.get("macd_slow", 26)),
        macd_signal=int(momentum.get("macd_signal", 9)),
        # Volume
        volume_sma_period=int(volume.get("sma_period", 20)),
        volume_spike_threshold=float(volume.get("spike_threshold", 1.5)),
        # Scoring
        scoring_weights=scoring_weights,
        tradeable_threshold=int(scoring.get("tradeable_threshold", 3)),
        # Exits
        atr_period=int(exits.get("atr_period", 14)),
        stop_method=exits.get("stop_method", "swing"),
        atr_stop_multiplier=float(exits.get("atr_stop_multiplier", 1.5)),
    )


def get_bucket_for_asset(
    asset_class: str,
    config: SignalEngineConfig,
) -> BucketConfig:
    """Determine which bucket (A or B) an asset class belongs to.

    Args:
        asset_class: The asset's class (stock, etf, index, crypto, metal).
        config: Signal engine configuration.

    Returns:
        BucketConfig for the appropriate bucket.

    Raises:
        ConfigError: If asset class doesn't map to any bucket.
    """
    ac = asset_class.lower()
    if ac in [c.lower() for c in config.bucket_a.asset_classes]:
        return config.bucket_a
    if ac in [c.lower() for c in config.bucket_b.asset_classes]:
        return config.bucket_b
    raise ConfigError(
        f"Asset class '{asset_class}' not mapped to any bucket. "
        f"Bucket A: {config.bucket_a.asset_classes}, "
        f"Bucket B: {config.bucket_b.asset_classes}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_signals(
    df: pd.DataFrame,
    asset_class: str,
    config: SignalEngineConfig | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Run the full signal engine pipeline on an OHLCV DataFrame.

    Pipeline steps:
        1. Load config and resolve bucket for asset class
        2. Add trend MA (bucket-specific)
        3. Detect volume spikes
        4. Detect market regime (SMA/EMA 200)
        5. Evaluate conditions: trend, structure, momentum
        6. Determine signal direction (2-of-3 gate + regime filter)
        7. Compute signal score (0-6)
        8. Compute exit levels (stop, target, trail-to-breakeven)

    Args:
        df: Canonical OHLCV DataFrame.
        asset_class: Asset class string (stock, etf, crypto, index, metal).
        config: Pre-loaded SignalEngineConfig. If None, loaded from config_path.
        config_path: Path to signals.yaml (used if config is None).

    Returns:
        DataFrame with all signal columns appended:
            - Trend MA, volume spike
            - Regime: regime_ma, regime, regime_allow_long/short, etc.
            - Conditions: trend_bull/bear, structure_bull/bear, momentum_bull/bear
            - Direction: signal_direction, signal_conditions_met
            - Score: signal_score, signal_tradeable
            - Exits: exit_stop, exit_target, exit_trail_be, exit_risk, etc.
    """
    if config is None:
        config = load_signal_config(config_path)

    bucket = get_bucket_for_asset(asset_class, config)

    # Step 1: Add trend MA for this bucket
    ma_col = f"{bucket.trend_ma_type}_{bucket.trend_ma_period}"
    result = add_ma(
        df,
        period=bucket.trend_ma_period,
        ma_type=bucket.trend_ma_type,
    )

    # Step 2: Volume spike detection
    result = detect_volume_spike(
        result,
        period=config.volume_sma_period,
        threshold=config.volume_spike_threshold,
    )

    # Step 3: Regime detection
    result = detect_regime(
        result,
        ma_type=config.regime_ma_type,
        ma_period=config.regime_ma_period,
        transition_closes=config.regime_transition_closes,
        strong_alignment_pct=config.regime_strong_alignment_pct,
    )

    # Step 4: Trend condition
    result = evaluate_trend_condition(result, ma_column=ma_col)

    # Step 5: Structure condition
    result = evaluate_structure_condition(
        result,
        swing_lookback=config.swing_lookback,
        level_proximity_pct=config.level_proximity_pct,
        pivot_lookback=config.pivot_lookback,
        pivot_merge_distance_pct=config.pivot_merge_distance_pct,
    )

    # Step 6: Momentum condition
    result = evaluate_momentum_condition(
        result,
        rsi_period=config.rsi_period,
        rsi_bull_threshold=config.rsi_bull_threshold,
        rsi_bear_threshold=config.rsi_bear_threshold,
        macd_fast=config.macd_fast,
        macd_slow=config.macd_slow,
        macd_signal=config.macd_signal,
    )

    # Step 7: Signal direction (2-of-3 gate)
    result = determine_signal_direction(result)

    # Step 8: Signal scoring
    result = compute_signal_score(
        result,
        weights=config.scoring_weights,
        tradeable_threshold=config.tradeable_threshold,
    )

    # Step 9: Exit levels
    result = compute_exit_levels(
        result,
        atr_period=config.atr_period,
        stop_method=config.stop_method,
        atr_stop_multiplier=config.atr_stop_multiplier,
        swing_lookback=config.swing_lookback,
        target_r_multiple=bucket.target_r_multiple,
        trail_breakeven_r=bucket.trail_breakeven_r,
    )

    return result
