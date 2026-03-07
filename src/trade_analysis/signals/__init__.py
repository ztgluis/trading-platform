"""Signal engine — 3-condition entry system with scoring and exit levels.

Main entry point:
    generate_signals(df, asset_class) → DataFrame with all signal columns

Components:
    - engine: config loader, bucket routing, generate_signals() orchestrator
    - regime: bull/bear/transition market regime classification
    - conditions: trend, structure, momentum condition evaluators
    - scoring: 2-of-3 gate, composite quality score (0-6)
    - exits: stop loss, target, trail-to-breakeven computation
"""

from trade_analysis.signals.conditions import (
    evaluate_momentum_condition,
    evaluate_structure_condition,
    evaluate_trend_condition,
)
from trade_analysis.signals.engine import (
    BucketConfig,
    SignalEngineConfig,
    generate_signals,
    get_bucket_for_asset,
    load_signal_config,
)
from trade_analysis.signals.exits import compute_exit_levels
from trade_analysis.signals.regime import detect_regime
from trade_analysis.signals.scoring import (
    compute_signal_score,
    determine_signal_direction,
)

__all__ = [
    # Engine
    "generate_signals",
    "load_signal_config",
    "get_bucket_for_asset",
    "BucketConfig",
    "SignalEngineConfig",
    # Regime
    "detect_regime",
    # Conditions
    "evaluate_trend_condition",
    "evaluate_structure_condition",
    "evaluate_momentum_condition",
    # Scoring
    "determine_signal_direction",
    "compute_signal_score",
    # Exits
    "compute_exit_levels",
]
