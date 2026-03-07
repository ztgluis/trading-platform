"""Exit level computation: stop loss, target, and trailing breakeven.

For each signal bar, computes:
    - Stop loss: recent swing extreme or ATR-based fallback
    - Target: entry + R × (entry - stop)
    - Trail-to-breakeven level: entry + trail_breakeven_R × (entry - stop)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta

from trade_analysis.indicators.structure import detect_swing_highs, detect_swing_lows


def compute_exit_levels(
    df: pd.DataFrame,
    signal_direction_col: str = "signal_direction",
    atr_period: int = 14,
    stop_method: str = "swing",
    atr_stop_multiplier: float = 1.5,
    swing_lookback: int = 3,
    target_r_multiple: float = 2.0,
    trail_breakeven_r: float = 1.0,
) -> pd.DataFrame:
    """Compute stop, target, and trail-to-breakeven for signal bars.

    Stop methods:
        - "swing": Use the most recent swing low (for longs) or swing high
          (for shorts) as the stop level. Falls back to ATR if no recent
          swing is found.
        - "atr": Use entry ± ATR × multiplier as the stop.

    Args:
        df: DataFrame with signal_direction column and OHLCV data.
        signal_direction_col: Column name for signal direction.
        atr_period: ATR lookback period.
        stop_method: "swing" or "atr".
        atr_stop_multiplier: Multiplier for ATR-based stops.
        swing_lookback: Lookback for swing detection (swing method).
        target_r_multiple: R-multiple for target level.
        trail_breakeven_r: R-multiple at which to trail stop to breakeven.

    Returns:
        DataFrame with columns:
            - exit_stop: stop loss price
            - exit_target: target price
            - exit_trail_be: trail-to-breakeven price (when to move stop to entry)
            - exit_risk: absolute risk per share (|entry - stop|)
            - exit_reward: absolute reward per share (|target - entry|)
            - exit_rr_ratio: reward/risk ratio
    """
    result = df.copy()
    direction = df[signal_direction_col]
    close = df["close"].values
    n = len(df)

    # Compute ATR
    atr_col = f"atr_{atr_period}"
    if atr_col in df.columns:
        atr = df[atr_col].values
    else:
        atr_series = ta.atr(
            high=df["high"], low=df["low"], close=df["close"], length=atr_period
        )
        atr = atr_series.values if atr_series is not None else np.full(n, np.nan)
        result[atr_col] = atr

    # Detect swings for swing-based stops
    if stop_method == "swing":
        with_lows = detect_swing_lows(df, lookback=swing_lookback)
        with_highs = detect_swing_highs(df, lookback=swing_lookback)
        swing_low_price = with_lows["swing_low_price"].values
        swing_high_price = with_highs["swing_high_price"].values

    # Initialize exit columns
    exit_stop = np.full(n, np.nan)
    exit_target = np.full(n, np.nan)
    exit_trail_be = np.full(n, np.nan)
    exit_risk = np.full(n, np.nan)
    exit_reward = np.full(n, np.nan)
    exit_rr_ratio = np.full(n, np.nan)

    for i in range(n):
        dir_val = direction.iloc[i]
        if dir_val is None or (isinstance(dir_val, float) and np.isnan(dir_val)):
            continue
        if dir_val not in ("long", "short"):
            continue

        entry = close[i]

        # --- Determine stop ---
        if stop_method == "swing":
            stop = _find_swing_stop(
                i, dir_val, swing_low_price, swing_high_price, entry, atr[i],
                atr_stop_multiplier,
            )
        else:
            # ATR-based stop
            if np.isnan(atr[i]):
                continue
            if dir_val == "long":
                stop = entry - atr[i] * atr_stop_multiplier
            else:
                stop = entry + atr[i] * atr_stop_multiplier

        if np.isnan(stop) or stop <= 0:
            continue

        # --- Compute risk ---
        risk = abs(entry - stop)
        if risk < 1e-10:
            continue

        # --- Target ---
        if dir_val == "long":
            target = entry + risk * target_r_multiple
        else:
            target = entry - risk * target_r_multiple

        # --- Trail to breakeven ---
        if dir_val == "long":
            trail_be = entry + risk * trail_breakeven_r
        else:
            trail_be = entry - risk * trail_breakeven_r

        # --- Reward ---
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else np.nan

        exit_stop[i] = stop
        exit_target[i] = target
        exit_trail_be[i] = trail_be
        exit_risk[i] = risk
        exit_reward[i] = reward
        exit_rr_ratio[i] = rr

    result["exit_stop"] = exit_stop
    result["exit_target"] = exit_target
    result["exit_trail_be"] = exit_trail_be
    result["exit_risk"] = exit_risk
    result["exit_reward"] = exit_reward
    result["exit_rr_ratio"] = exit_rr_ratio

    return result


def _find_swing_stop(
    bar_idx: int,
    direction: str,
    swing_low_prices: np.ndarray,
    swing_high_prices: np.ndarray,
    entry: float,
    atr_val: float,
    atr_multiplier: float,
) -> float:
    """Find the most recent swing extreme for stop placement.

    For longs: most recent swing low below entry.
    For shorts: most recent swing high above entry.
    Falls back to ATR-based stop if no valid swing found.
    """
    if direction == "long":
        # Look backwards for the most recent swing low
        for j in range(bar_idx - 1, -1, -1):
            if not np.isnan(swing_low_prices[j]) and swing_low_prices[j] < entry:
                return float(swing_low_prices[j])
        # Fallback: ATR-based
        if not np.isnan(atr_val):
            return entry - atr_val * atr_multiplier
    else:
        # Look backwards for the most recent swing high
        for j in range(bar_idx - 1, -1, -1):
            if not np.isnan(swing_high_prices[j]) and swing_high_prices[j] > entry:
                return float(swing_high_prices[j])
        # Fallback: ATR-based
        if not np.isnan(atr_val):
            return entry + atr_val * atr_multiplier

    return np.nan
