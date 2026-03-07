"""Signal scoring: direction determination and composite score.

Combines condition outputs with regime filtering to determine signal
direction (long/short/None) and compute a 0-6 quality score.
"""

import numpy as np
import pandas as pd


# Default scoring weights
DEFAULT_WEIGHTS = {
    "trend_confirmed": 1,
    "structure_single_method": 1,
    "structure_multi_method": 2,
    "momentum_confirmed": 1,
    "regime_strongly_aligned": 1,
    "volume_spike_on_entry": 1,
}


def determine_signal_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Determine signal direction based on regime + 2-of-3 condition gate.

    A long signal requires:
        - regime_allow_long is True
        - At least 2 of: trend_bull, structure_bull, momentum_bull

    A short signal requires:
        - regime_allow_short is True
        - At least 2 of: trend_bear, structure_bear, momentum_bear

    If both long and short conditions are met (rare), long takes priority.

    Args:
        df: DataFrame with regime and condition columns.

    Returns:
        DataFrame with columns:
            - signal_direction: "long" / "short" / None
            - signal_conditions_met: int (0-3) — number of conditions met
    """
    result = df.copy()

    # Count bullish and bearish conditions
    bull_count = (
        df["trend_bull"].astype(int)
        + df["structure_bull"].astype(int)
        + df["momentum_bull"].astype(int)
    )
    bear_count = (
        df["trend_bear"].astype(int)
        + df["structure_bear"].astype(int)
        + df["momentum_bear"].astype(int)
    )

    # 2-of-3 gate + regime filter
    long_signal = (bull_count >= 2) & df["regime_allow_long"]
    short_signal = (bear_count >= 2) & df["regime_allow_short"]

    # Resolve conflicts: long takes priority
    direction = pd.Series(None, index=df.index, dtype=object)
    direction[long_signal] = "long"
    direction[short_signal & ~long_signal] = "short"

    # Conditions met = max of bull/bear count for the chosen direction
    conditions_met = pd.Series(0, index=df.index, dtype=int)
    conditions_met[direction == "long"] = bull_count[direction == "long"]
    conditions_met[direction == "short"] = bear_count[direction == "short"]

    result["signal_direction"] = direction
    result["signal_conditions_met"] = conditions_met

    return result


def compute_signal_score(
    df: pd.DataFrame,
    weights: dict[str, int] | None = None,
    tradeable_threshold: int = 3,
    volume_spike_col: str = "volume_spike",
) -> pd.DataFrame:
    """Compute the composite signal quality score (0-6).

    Scoring breakdown:
        - trend_confirmed (+1): trend condition matches signal direction
        - structure_single_method (+1): structure condition met via single level type
        - structure_multi_method (+2): structure confirmed by BOTH pivot and round levels
        - momentum_confirmed (+1): momentum condition matches signal direction
        - regime_strongly_aligned (+1): price is far from regime MA in signal direction
        - volume_spike_on_entry (+1): volume spike on the signal bar

    The structure score is exclusive: either +1 (single) or +2 (multi), not both.

    Args:
        df: DataFrame with signal_direction and all condition/regime columns.
        weights: Custom scoring weights (default: DEFAULT_WEIGHTS).
        tradeable_threshold: Minimum score to be considered tradeable (default: 3).
        volume_spike_col: Column name for volume spike indicator.

    Returns:
        DataFrame with columns:
            - signal_score: int (0-6)
            - signal_tradeable: bool — score >= tradeable_threshold
    """
    result = df.copy()
    w = weights or DEFAULT_WEIGHTS

    direction = df["signal_direction"]
    is_long = direction == "long"
    is_short = direction == "short"
    has_signal = is_long | is_short

    # Initialize score
    score = pd.Series(0, index=df.index, dtype=int)

    # 1. Trend confirmed
    trend_match = (is_long & df["trend_bull"]) | (is_short & df["trend_bear"])
    score += trend_match.astype(int) * w.get("trend_confirmed", 1)

    # 2. Structure (single vs multi-method — exclusive)
    structure_match = (is_long & df["structure_bull"]) | (
        is_short & df["structure_bear"]
    )
    multi_method = structure_match & df["structure_multi_method"]
    single_method = structure_match & ~df["structure_multi_method"]

    score += multi_method.astype(int) * w.get("structure_multi_method", 2)
    score += single_method.astype(int) * w.get("structure_single_method", 1)

    # 3. Momentum confirmed
    momentum_match = (is_long & df["momentum_bull"]) | (
        is_short & df["momentum_bear"]
    )
    score += momentum_match.astype(int) * w.get("momentum_confirmed", 1)

    # 4. Regime strongly aligned
    score += df["regime_strongly_aligned"].astype(int) * w.get(
        "regime_strongly_aligned", 1
    )

    # 5. Volume spike on entry
    if volume_spike_col in df.columns:
        score += df[volume_spike_col].astype(int) * w.get("volume_spike_on_entry", 1)

    # Only bars with a signal get a score; everything else is 0
    score = score.where(has_signal, 0)

    result["signal_score"] = score
    result["signal_tradeable"] = score >= tradeable_threshold

    return result
