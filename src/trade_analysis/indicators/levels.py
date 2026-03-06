"""Key level detection: pivot S/R, round numbers, proximity checks."""

import numpy as np
import pandas as pd

from trade_analysis.indicators.structure import detect_swing_highs, detect_swing_lows


def detect_pivot_levels(
    df: pd.DataFrame,
    lookback: int = 5,
    merge_distance_pct: float = 0.5,
) -> pd.DataFrame:
    """Detect support and resistance levels from swing highs/lows.

    Clusters nearby levels within merge_distance_pct to avoid duplicates.

    Args:
        df: Canonical OHLCV DataFrame.
        lookback: Swing detection lookback bars.
        merge_distance_pct: Merge levels within this percentage distance.

    Returns:
        DataFrame with columns: price, type ("support" | "resistance"), count.
        Sorted by price. Count indicates how many pivots clustered into this level.
    """
    # Detect swings
    with_highs = detect_swing_highs(df, lookback)
    with_lows = detect_swing_lows(df, lookback)

    # Collect all pivot prices
    resistance_prices = with_highs.loc[
        with_highs["swing_high"], "swing_high_price"
    ].dropna().values
    support_prices = with_lows.loc[
        with_lows["swing_low"], "swing_low_price"
    ].dropna().values

    # Build raw levels list
    raw_levels = []
    for p in resistance_prices:
        raw_levels.append({"price": float(p), "type": "resistance"})
    for p in support_prices:
        raw_levels.append({"price": float(p), "type": "support"})

    if not raw_levels:
        return pd.DataFrame(columns=["price", "type", "count"])

    # Sort by price
    raw_levels.sort(key=lambda x: x["price"])

    # Cluster nearby levels
    clustered = []
    current_cluster = [raw_levels[0]]

    for level in raw_levels[1:]:
        cluster_avg = np.mean([l["price"] for l in current_cluster])
        distance_pct = abs(level["price"] - cluster_avg) / cluster_avg * 100

        if distance_pct <= merge_distance_pct:
            current_cluster.append(level)
        else:
            clustered.append(_merge_cluster(current_cluster))
            current_cluster = [level]

    clustered.append(_merge_cluster(current_cluster))

    return pd.DataFrame(clustered)


def _merge_cluster(cluster: list[dict]) -> dict:
    """Merge a cluster of nearby levels into one."""
    prices = [l["price"] for l in cluster]
    types = [l["type"] for l in cluster]
    # Type is determined by majority
    resistance_count = types.count("resistance")
    support_count = types.count("support")
    level_type = "resistance" if resistance_count >= support_count else "support"
    return {
        "price": float(np.mean(prices)),
        "type": level_type,
        "count": len(cluster),
    }


def detect_round_numbers(
    price: float,
    step: float | None = None,
    count: int = 3,
) -> list[float]:
    """Return the nearest round numbers above and below a price.

    Args:
        price: Current price.
        step: Round number step size. If None, auto-detected based on price magnitude.
        count: Number of round numbers to return on each side.

    Returns:
        Sorted list of round numbers (below and above the price).
    """
    if step is None:
        if price >= 1000:
            step = 100.0
        elif price >= 100:
            step = 10.0
        elif price >= 10:
            step = 5.0
        else:
            step = 1.0

    base = (price // step) * step
    levels = []
    for i in range(-count, count + 1):
        level = base + i * step
        if level > 0:
            levels.append(float(level))

    return sorted(set(levels))


def find_nearest_level(
    price: float,
    levels: pd.DataFrame | list[float],
    max_distance_pct: float = 3.0,
) -> dict | None:
    """Find the closest level to a price within a maximum distance.

    Args:
        price: Current price to check.
        levels: Either a DataFrame with 'price' column (from detect_pivot_levels)
            or a list of price floats.
        max_distance_pct: Maximum distance in percent to qualify as "near".

    Returns:
        Dict with 'price', 'distance_pct', and optionally 'type'/'count',
        or None if no level is within range.
    """
    if isinstance(levels, pd.DataFrame):
        if levels.empty:
            return None
        prices = levels["price"].values
        level_data = levels.to_dict("records")
    else:
        if not levels:
            return None
        prices = np.array(levels)
        level_data = [{"price": p} for p in levels]

    distances = np.abs(prices - price) / price * 100
    min_idx = int(np.argmin(distances))
    min_distance = distances[min_idx]

    if min_distance > max_distance_pct:
        return None

    result = level_data[min_idx].copy()
    result["distance_pct"] = float(min_distance)
    return result
