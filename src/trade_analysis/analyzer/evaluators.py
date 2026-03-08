"""Hypothesis evaluators: H1-H5 trend and momentum hypotheses.

Each evaluator takes a grid results DataFrame and returns a HypothesisResult.
The DataFrame should come from GridResult.to_dataframe() and contain
parameter columns + stat columns.
"""

from __future__ import annotations

import pandas as pd

from trade_analysis.analyzer.hypothesis import (
    HypothesisResult,
    compare_groups,
)


# ---------------------------------------------------------------------------
# Significance threshold
# ---------------------------------------------------------------------------

# Minimum absolute difference in avg_r to consider a result significant
_MIN_SIGNIFICANT_DIFF = 0.05


# ---------------------------------------------------------------------------
# H1: Does applying any trend filter improve results vs. no filter?
# ---------------------------------------------------------------------------


def evaluate_h1(grid_df: pd.DataFrame) -> HypothesisResult:
    """H1: Does applying any trend filter improve results vs. no filter?

    Evaluates by comparing results across trend_ma_period values.
    Very small periods (< 5) approximate "no filter" since the MA tracks
    price closely. Larger periods provide more meaningful filtering.

    Requires grid_df to contain 'trend_ma_period' column.
    """
    question = (
        "Does applying any trend filter improve results vs. no filter?"
    )

    if "trend_ma_period" not in grid_df.columns:
        return HypothesisResult(
            hypothesis_id="H1",
            question=question,
            verdict="inconclusive",
            summary="trend_ma_period not swept — cannot evaluate.",
        )

    comparison = compare_groups(grid_df, "trend_ma_period", "avg_r")
    if not comparison:
        return HypothesisResult(
            hypothesis_id="H1",
            question=question,
            verdict="inconclusive",
            summary="Insufficient data to compare trend filter periods.",
        )

    groups = comparison["groups"]
    periods = sorted(groups.keys())

    if len(periods) < 2:
        return HypothesisResult(
            hypothesis_id="H1",
            question=question,
            verdict="inconclusive",
            summary="Only one trend_ma_period value — cannot compare.",
        )

    # Compare smallest period (≈ no filter) vs the rest
    smallest = periods[0]
    rest_avg = sum(
        groups[p]["mean"] for p in periods[1:]
    ) / len(periods[1:])

    no_filter_avg_r = groups[smallest]["mean"]
    filter_avg_r = rest_avg
    diff = filter_avg_r - no_filter_avg_r

    evidence = {
        "smallest_period": smallest,
        "smallest_period_avg_r": round(no_filter_avg_r, 4),
        "other_periods_avg_r": round(filter_avg_r, 4),
        "difference": round(diff, 4),
        "best_period": comparison["best_group"],
        "best_period_avg_r": round(groups[comparison["best_group"]]["mean"], 4),
    }

    if diff > _MIN_SIGNIFICANT_DIFF:
        return HypothesisResult(
            hypothesis_id="H1",
            question=question,
            verdict="supported",
            evidence=evidence,
            summary=(
                f"Trend filter improves avg R by {diff:+.2f}. "
                f"Best period: {comparison['best_group']}."
            ),
        )
    elif diff < -_MIN_SIGNIFICANT_DIFF:
        return HypothesisResult(
            hypothesis_id="H1",
            question=question,
            verdict="refuted",
            evidence=evidence,
            summary=(
                f"Trend filter reduces avg R by {diff:+.2f}. "
                f"Shorter periods perform better."
            ),
        )
    else:
        return HypothesisResult(
            hypothesis_id="H1",
            question=question,
            verdict="inconclusive",
            evidence=evidence,
            summary=(
                f"No significant difference (diff={diff:+.2f}). "
                f"Trend filter has minimal impact."
            ),
        )


# ---------------------------------------------------------------------------
# H2: Does EMA vs. SMA type matter?
# ---------------------------------------------------------------------------


def evaluate_h2(grid_df: pd.DataFrame) -> HypothesisResult:
    """H2: Does EMA vs. SMA type matter, or is the period the dominant variable?

    Compares avg_r across trend_ma_type groups. Also checks if
    trend_ma_period variance is larger than type variance.

    Requires grid_df to contain 'trend_ma_type' column.
    """
    question = (
        "Does EMA vs. SMA type matter, or is the period the dominant variable?"
    )

    if "trend_ma_type" not in grid_df.columns:
        return HypothesisResult(
            hypothesis_id="H2",
            question=question,
            verdict="inconclusive",
            summary="trend_ma_type not swept — cannot evaluate.",
        )

    type_comparison = compare_groups(grid_df, "trend_ma_type", "avg_r")
    if not type_comparison:
        return HypothesisResult(
            hypothesis_id="H2",
            question=question,
            verdict="inconclusive",
            summary="Insufficient data to compare MA types.",
        )

    type_spread = abs(type_comparison["spread"])

    # Check if period variance is larger
    period_spread = 0.0
    if "trend_ma_period" in grid_df.columns:
        period_comparison = compare_groups(grid_df, "trend_ma_period", "avg_r")
        if period_comparison:
            period_spread = abs(period_comparison["spread"])

    evidence = {
        "type_spread": round(type_spread, 4),
        "period_spread": round(period_spread, 4),
        "best_type": type_comparison["best_group"],
        "type_groups": type_comparison["groups"],
    }

    if type_spread < _MIN_SIGNIFICANT_DIFF:
        verdict = "refuted"
        summary = (
            f"MA type has negligible impact (spread={type_spread:.3f}). "
            f"Period is the dominant variable (spread={period_spread:.3f})."
        )
    elif period_spread > type_spread * 2:
        verdict = "refuted"
        summary = (
            f"Period matters more than type. "
            f"Period spread ({period_spread:.3f}) > type spread ({type_spread:.3f})."
        )
    else:
        verdict = "supported"
        summary = (
            f"MA type matters: {type_comparison['best_group']} is better "
            f"by {type_spread:.3f} avg R."
        )

    return HypothesisResult(
        hypothesis_id="H2",
        question=question,
        verdict=verdict,
        evidence=evidence,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# H3: What period produces the best risk-adjusted returns?
# ---------------------------------------------------------------------------


def evaluate_h3(grid_df: pd.DataFrame) -> HypothesisResult:
    """H3: What period produces the best risk-adjusted returns per asset class?

    Ranks trend_ma_period by avg_r (risk-adjusted) and reports the best
    period and whether it's part of a robust zone.

    Requires grid_df to contain 'trend_ma_period' column.
    """
    question = (
        "What period produces the best risk-adjusted returns per asset class?"
    )

    if "trend_ma_period" not in grid_df.columns:
        return HypothesisResult(
            hypothesis_id="H3",
            question=question,
            verdict="inconclusive",
            summary="trend_ma_period not swept — cannot evaluate.",
        )

    comparison = compare_groups(grid_df, "trend_ma_period", "avg_r")
    if not comparison:
        return HypothesisResult(
            hypothesis_id="H3",
            question=question,
            verdict="inconclusive",
            summary="Insufficient data.",
        )

    groups = comparison["groups"]
    best_period = comparison["best_group"]
    best_avg_r = groups[best_period]["mean"]

    # Check for robustness: are neighbors close?
    sorted_periods = sorted(groups.keys())
    best_idx = sorted_periods.index(best_period)
    neighbors = []
    if best_idx > 0:
        neighbors.append(groups[sorted_periods[best_idx - 1]]["mean"])
    if best_idx < len(sorted_periods) - 1:
        neighbors.append(groups[sorted_periods[best_idx + 1]]["mean"])

    is_robust = True
    if neighbors and best_avg_r != 0:
        neighbor_avg = sum(neighbors) / len(neighbors)
        if abs(best_avg_r - neighbor_avg) / abs(best_avg_r) > 0.4:
            is_robust = False

    # Ranking
    ranking = sorted(groups.items(), key=lambda x: x[1]["mean"], reverse=True)

    evidence = {
        "best_period": best_period,
        "best_avg_r": round(best_avg_r, 4),
        "is_robust": is_robust,
        "ranking": [
            {"period": p, "avg_r": round(g["mean"], 4)} for p, g in ranking
        ],
    }

    robust_str = "robust" if is_robust else "isolated (potential overfit)"
    return HypothesisResult(
        hypothesis_id="H3",
        question=question,
        verdict="supported",
        evidence=evidence,
        summary=(
            f"Best period: {best_period} (avg R={best_avg_r:+.2f}, {robust_str}). "
            f"Spread across periods: {comparison['spread']:.3f}."
        ),
    )


# ---------------------------------------------------------------------------
# H4: Single MA vs fast/slow crossover
# ---------------------------------------------------------------------------


def evaluate_h4() -> HypothesisResult:
    """H4: Is a single MA better or worse than a fast/slow crossover?

    Not testable with the current signal engine — crossover logic
    is not implemented.
    """
    return HypothesisResult(
        hypothesis_id="H4",
        question=(
            "Is a single MA better or worse than a fast/slow crossover?"
        ),
        verdict="not_testable",
        evidence={"reason": "Crossover logic not implemented in signal engine."},
        summary=(
            "Cannot test — signal engine uses single MA only. "
            "Crossover support needed for evaluation."
        ),
    )


# ---------------------------------------------------------------------------
# H5: RSI threshold impact
# ---------------------------------------------------------------------------


def evaluate_h5(grid_df: pd.DataFrame) -> HypothesisResult:
    """H5: Does requiring RSI > 50 improve win rate, or only reduce frequency?

    Compares win_rate and total_trades across rsi_bull_threshold values
    (or rsi_period as a proxy). Reports the trade-off.

    Requires grid_df to contain 'rsi_bull_threshold' or 'rsi_period' column.
    """
    question = (
        "Does requiring RSI > 50 improve win rate, "
        "or only reduce trade frequency?"
    )

    # Try rsi_bull_threshold first, fall back to rsi_period
    param_col = None
    if "rsi_bull_threshold" in grid_df.columns:
        param_col = "rsi_bull_threshold"
    elif "rsi_period" in grid_df.columns:
        param_col = "rsi_period"

    if param_col is None:
        return HypothesisResult(
            hypothesis_id="H5",
            question=question,
            verdict="inconclusive",
            summary="No RSI parameter swept — cannot evaluate.",
        )

    wr_comparison = compare_groups(grid_df, param_col, "win_rate")
    trades_comparison = compare_groups(grid_df, param_col, "total_trades")

    if not wr_comparison or not trades_comparison:
        return HypothesisResult(
            hypothesis_id="H5",
            question=question,
            verdict="inconclusive",
            summary="Insufficient data.",
        )

    wr_groups = wr_comparison["groups"]
    trades_groups = trades_comparison["groups"]

    # Check if higher thresholds improve win rate
    sorted_params = sorted(wr_groups.keys())
    if len(sorted_params) < 2:
        return HypothesisResult(
            hypothesis_id="H5",
            question=question,
            verdict="inconclusive",
            summary=f"Only one {param_col} value — cannot compare.",
        )

    lowest = sorted_params[0]
    highest = sorted_params[-1]

    wr_diff = wr_groups[highest]["mean"] - wr_groups[lowest]["mean"]
    trades_diff = trades_groups[highest]["mean"] - trades_groups[lowest]["mean"]

    evidence = {
        "parameter": param_col,
        "lowest_value": lowest,
        "highest_value": highest,
        "win_rate_diff": round(wr_diff, 4),
        "trade_count_diff": round(trades_diff, 1),
        "win_rate_by_param": {
            k: round(v["mean"], 4) for k, v in wr_groups.items()
        },
        "trades_by_param": {
            k: round(v["mean"], 1) for k, v in trades_groups.items()
        },
    }

    improves_wr = wr_diff > 0.02  # 2% improvement threshold
    reduces_trades = trades_diff < 0

    if improves_wr and reduces_trades:
        verdict = "supported"
        summary = (
            f"Higher {param_col} improves win rate by {wr_diff:+.1%} "
            f"but reduces trades by {abs(trades_diff):.0f}."
        )
    elif improves_wr and not reduces_trades:
        verdict = "supported"
        summary = (
            f"Higher {param_col} improves win rate by {wr_diff:+.1%} "
            f"without reducing trade count."
        )
    elif not improves_wr and reduces_trades:
        verdict = "refuted"
        summary = (
            f"Higher {param_col} only reduces trades by {abs(trades_diff):.0f} "
            f"without improving win rate (diff={wr_diff:+.1%})."
        )
    else:
        verdict = "inconclusive"
        summary = (
            f"No clear pattern: WR diff={wr_diff:+.1%}, "
            f"trade diff={trades_diff:+.0f}."
        )

    return HypothesisResult(
        hypothesis_id="H5",
        question=question,
        verdict=verdict,
        evidence=evidence,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------


def evaluate_all(grid_df: pd.DataFrame) -> list[HypothesisResult]:
    """Evaluate all H1-H5 hypotheses on a grid results DataFrame.

    Args:
        grid_df: Grid results DataFrame with parameter + stat columns.

    Returns:
        List of 5 HypothesisResults (H1-H5).
    """
    return [
        evaluate_h1(grid_df),
        evaluate_h2(grid_df),
        evaluate_h3(grid_df),
        evaluate_h4(),
        evaluate_h5(grid_df),
    ]
