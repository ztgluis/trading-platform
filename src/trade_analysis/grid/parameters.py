"""Parameter grid generation and signal config modification.

Generates all parameter combinations from ranges and applies them to
frozen SignalEngineConfig via dataclasses.replace().
"""

from __future__ import annotations

import itertools
from dataclasses import replace

from trade_analysis.exceptions import ConfigError
from trade_analysis.signals.engine import (
    BucketConfig,
    SignalEngineConfig,
    get_bucket_for_asset,
)


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

# Parameters that live directly on SignalEngineConfig
_DIRECT_PARAMS: set[str] = {
    "regime_ma_type",
    "regime_ma_period",
    "regime_transition_closes",
    "regime_strong_alignment_pct",
    "swing_lookback",
    "level_proximity_pct",
    "pivot_lookback",
    "pivot_merge_distance_pct",
    "rsi_period",
    "rsi_bull_threshold",
    "rsi_bear_threshold",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "volume_sma_period",
    "volume_spike_threshold",
    "tradeable_threshold",
    "atr_period",
    "stop_method",
    "atr_stop_multiplier",
}

# Parameters that live on BucketConfig (applied to the active bucket)
_BUCKET_PARAMS: set[str] = {
    "trend_ma_type",
    "trend_ma_period",
    "max_hold_weeks",
    "target_r_multiple",
    "trail_breakeven_r",
}

ALL_KNOWN_PARAMS: set[str] = _DIRECT_PARAMS | _BUCKET_PARAMS


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------


def generate_parameter_grid(parameters: dict[str, list]) -> list[dict]:
    """Generate all combinations of parameter values.

    Args:
        parameters: Dict mapping parameter names to lists of values.

    Returns:
        List of dicts, each representing one parameter combination.

    Example:
        >>> generate_parameter_grid({"a": [1, 2], "b": [10, 20]})
        [{"a": 1, "b": 10}, {"a": 1, "b": 20}, {"a": 2, "b": 10}, {"a": 2, "b": 20}]
    """
    if not parameters:
        return [{}]

    names = list(parameters.keys())
    value_lists = [parameters[name] for name in names]

    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


# ---------------------------------------------------------------------------
# Config modification
# ---------------------------------------------------------------------------


def apply_params_to_config(
    base_config: SignalEngineConfig,
    params: dict,
    asset_class: str,
) -> SignalEngineConfig:
    """Create a modified SignalEngineConfig with the given parameter values.

    Resolves bucket-specific parameters (e.g., trend_ma_period) to the
    correct bucket based on asset_class.

    Args:
        base_config: Base frozen config to modify.
        params: Dict of param_name → value to set.
        asset_class: Asset class for bucket resolution.

    Returns:
        New SignalEngineConfig with the parameters applied.

    Raises:
        ConfigError: If an unknown parameter name is encountered.
    """
    if not params:
        return base_config

    # Validate all param names
    unknown = set(params.keys()) - ALL_KNOWN_PARAMS
    if unknown:
        raise ConfigError(f"Unknown grid parameters: {sorted(unknown)}")

    # Split params into direct and bucket
    direct_params = {k: v for k, v in params.items() if k in _DIRECT_PARAMS}
    bucket_params = {k: v for k, v in params.items() if k in _BUCKET_PARAMS}

    # Apply direct params
    config = replace(base_config, **direct_params) if direct_params else base_config

    # Apply bucket params
    if bucket_params:
        config = _apply_bucket_params(config, bucket_params, asset_class)

    return config


def _apply_bucket_params(
    config: SignalEngineConfig,
    params: dict,
    asset_class: str,
) -> SignalEngineConfig:
    """Apply bucket-specific parameters to the correct bucket.

    Determines which bucket (A or B) to modify based on asset_class.
    """
    # Determine which bucket this asset class belongs to
    bucket = get_bucket_for_asset(asset_class, config)
    is_bucket_a = bucket is config.bucket_a

    # Create modified bucket
    new_bucket = replace(bucket, **params)

    # Replace the correct bucket on the config
    if is_bucket_a:
        return replace(config, bucket_a=new_bucket)
    else:
        return replace(config, bucket_b=new_bucket)
