"""Robustness analysis: detect stable parameter zones vs isolated peaks.

A robust parameter value performs similarly to its neighbors.
An isolated peak spikes while neighbors underperform — a curve-fit artifact.
"""

from __future__ import annotations

import pandas as pd


def analyze_robustness(
    results_df: pd.DataFrame,
    param_name: str,
    metric: str = "total_r",
    tolerance: float = 0.2,
) -> pd.DataFrame:
    """Analyze robustness for a single parameter.

    For each unique value of ``param_name``, computes the average metric
    across all combos with that value, then compares to neighbor values.

    A value is "robust" if its metric is within ``tolerance`` (fraction)
    of its neighbors' average. A value is an "isolated peak" if it is
    more than 2x ``tolerance`` better than neighbors.

    Args:
        results_df: Full grid results DataFrame (from GridResult.to_dataframe()).
        param_name: Parameter to analyze.
        metric: Stat to compare (e.g., "total_r", "avg_r", "win_rate").
        tolerance: Fraction difference threshold for robustness (default 0.2 = 20%).

    Returns:
        DataFrame with columns: param_value, metric_avg, neighbor_avg,
        is_robust, is_isolated_peak — one row per unique param value,
        sorted by param_value.
    """
    if param_name not in results_df.columns or metric not in results_df.columns:
        return pd.DataFrame()

    # Average metric for each param value (averaging across other params)
    grouped = (
        results_df.groupby(param_name)[metric]
        .mean()
        .sort_index()
        .reset_index()
    )
    grouped.columns = ["param_value", "metric_avg"]

    # Compute neighbor averages
    n = len(grouped)
    neighbor_avgs = []
    for i in range(n):
        neighbors = []
        if i > 0:
            neighbors.append(grouped.iloc[i - 1]["metric_avg"])
        if i < n - 1:
            neighbors.append(grouped.iloc[i + 1]["metric_avg"])
        if neighbors:
            neighbor_avgs.append(sum(neighbors) / len(neighbors))
        else:
            # Single value — no neighbors, mark as robust by default
            neighbor_avgs.append(grouped.iloc[i]["metric_avg"])

    grouped["neighbor_avg"] = neighbor_avgs

    # Classify
    grouped["is_robust"] = grouped.apply(
        lambda row: _is_robust(row["metric_avg"], row["neighbor_avg"], tolerance),
        axis=1,
    )
    grouped["is_isolated_peak"] = grouped.apply(
        lambda row: _is_isolated_peak(
            row["metric_avg"], row["neighbor_avg"], tolerance
        ),
        axis=1,
    )

    return grouped


def _is_robust(value: float, neighbor_avg: float, tolerance: float) -> bool:
    """Value is within tolerance of its neighbor average."""
    if neighbor_avg == 0:
        return abs(value) < tolerance
    ratio = abs(value - neighbor_avg) / abs(neighbor_avg)
    return ratio <= tolerance


def _is_isolated_peak(
    value: float, neighbor_avg: float, tolerance: float
) -> bool:
    """Value is significantly better than neighbors (potential curve-fit)."""
    if neighbor_avg == 0:
        return value > tolerance * 2
    if neighbor_avg < 0 and value > 0:
        return True  # Positive when neighbors are negative
    if neighbor_avg > 0:
        ratio = (value - neighbor_avg) / abs(neighbor_avg)
        return ratio > tolerance * 2
    return False


def find_robust_zones(
    results_df: pd.DataFrame,
    metric: str = "total_r",
    tolerance: float = 0.2,
) -> dict[str, list]:
    """Find robust parameter zones across all swept parameters.

    For each parameter, identifies contiguous ranges of values where
    performance is stable (all values are robust).

    Args:
        results_df: Full grid results DataFrame.
        metric: Stat to evaluate robustness on.
        tolerance: Fraction tolerance for robustness.

    Returns:
        Dict mapping param_name → list of robust zones.
        Each zone is a dict with "values" (list) and "avg_metric" (float).
    """
    # Detect which columns are parameters (non-stat columns)
    stat_columns = {
        "total_trades", "win_rate", "avg_r", "total_r", "profit_factor",
        "max_consecutive_wins", "max_consecutive_losses", "max_drawdown_r",
        "avg_duration_bars", "avg_duration_days", "longest_trade_bars",
        "shortest_trade_bars", "sufficient_trades",
        "by_regime", "by_direction", "by_signal_score", "by_exit_reason",
        "rank",
    }
    param_columns = [
        col for col in results_df.columns if col not in stat_columns
    ]

    zones: dict[str, list] = {}
    for param in param_columns:
        robustness = analyze_robustness(results_df, param, metric, tolerance)
        if robustness.empty:
            continue

        # Find contiguous robust zones
        param_zones = _find_contiguous_zones(robustness)
        if param_zones:
            zones[param] = param_zones

    return zones


def _find_contiguous_zones(robustness_df: pd.DataFrame) -> list[dict]:
    """Find contiguous zones of robust values."""
    zones: list[dict] = []
    current_zone_values: list = []
    current_zone_metrics: list[float] = []

    for _, row in robustness_df.iterrows():
        if row["is_robust"]:
            current_zone_values.append(row["param_value"])
            current_zone_metrics.append(row["metric_avg"])
        else:
            if len(current_zone_values) >= 2:
                zones.append({
                    "values": list(current_zone_values),
                    "avg_metric": sum(current_zone_metrics) / len(current_zone_metrics),
                })
            current_zone_values = []
            current_zone_metrics = []

    # Flush last zone
    if len(current_zone_values) >= 2:
        zones.append({
            "values": list(current_zone_values),
            "avg_metric": sum(current_zone_metrics) / len(current_zone_metrics),
        })

    return zones
