"""Hypothesis result model and comparison utilities.

HypothesisResult captures the outcome of testing a trading hypothesis
against grid sweep data. Comparison utilities help evaluators group
and compare parameter configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Valid verdicts
# ---------------------------------------------------------------------------

VERDICTS = {"supported", "refuted", "inconclusive", "not_testable"}


# ---------------------------------------------------------------------------
# HypothesisResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisResult:
    """Outcome of evaluating a single trading hypothesis.

    Attributes:
        hypothesis_id: Short identifier (e.g., "H1").
        question: Full hypothesis text.
        verdict: One of "supported", "refuted", "inconclusive", "not_testable".
        evidence: Dict of metrics and data supporting the verdict.
        summary: Human-readable one-line conclusion.
    """

    hypothesis_id: str
    question: str
    verdict: str
    evidence: dict = field(default_factory=dict)
    summary: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"Invalid verdict '{self.verdict}', "
                f"must be one of: {sorted(VERDICTS)}"
            )


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "total_r",
) -> dict:
    """Compare a metric across groups defined by a column.

    Groups the DataFrame by ``group_col``, computes mean and count of
    ``metric`` per group, and identifies the best/worst groups.

    Args:
        df: Grid results DataFrame.
        group_col: Column to group by (e.g., "trend_ma_type").
        metric: Stat to compare (e.g., "total_r", "win_rate").

    Returns:
        Dict with keys: groups (dict of group→{mean, count}),
        best_group, worst_group, spread (best - worst).
        Returns empty dict if group_col or metric not in DataFrame.
    """
    if group_col not in df.columns or metric not in df.columns:
        return {}

    grouped = df.groupby(group_col)[metric].agg(["mean", "count"])

    groups = {}
    for name, row in grouped.iterrows():
        groups[name] = {
            "mean": float(row["mean"]),
            "count": int(row["count"]),
        }

    if not groups:
        return {}

    best_group = max(groups, key=lambda k: groups[k]["mean"])
    worst_group = min(groups, key=lambda k: groups[k]["mean"])
    spread = groups[best_group]["mean"] - groups[worst_group]["mean"]

    return {
        "groups": groups,
        "best_group": best_group,
        "worst_group": worst_group,
        "spread": spread,
        "metric": metric,
    }


def compare_metrics_by_group(
    df: pd.DataFrame,
    group_col: str,
    metrics: list[str] | None = None,
) -> dict:
    """Compare multiple metrics across groups.

    Args:
        df: Grid results DataFrame.
        group_col: Column to group by.
        metrics: List of metrics to compare. Defaults to common stats.

    Returns:
        Dict mapping metric name → compare_groups result.
    """
    if metrics is None:
        metrics = ["total_r", "avg_r", "win_rate", "profit_factor", "total_trades"]

    results = {}
    for metric in metrics:
        comparison = compare_groups(df, group_col, metric)
        if comparison:
            results[metric] = comparison

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_hypothesis_report(results: list[HypothesisResult]) -> str:
    """Format a list of hypothesis results as a readable report.

    Args:
        results: List of HypothesisResult from evaluators.

    Returns:
        Multi-line string report.
    """
    lines = [
        "=" * 60,
        "HYPOTHESIS EVALUATION REPORT",
        "=" * 60,
        "",
    ]

    for r in results:
        verdict_icon = {
            "supported": "[+]",
            "refuted": "[-]",
            "inconclusive": "[?]",
            "not_testable": "[~]",
        }.get(r.verdict, "[?]")

        lines.append(f"{verdict_icon} {r.hypothesis_id}: {r.verdict.upper()}")
        lines.append(f"    Q: {r.question}")
        if r.summary:
            lines.append(f"    A: {r.summary}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)
