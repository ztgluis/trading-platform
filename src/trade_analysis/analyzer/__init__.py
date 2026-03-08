"""Results Analyzer — hypothesis testing and Supabase persistence."""

from trade_analysis.analyzer.evaluators import (
    evaluate_all,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3,
    evaluate_h4,
    evaluate_h5,
)
from trade_analysis.analyzer.hypothesis import (
    VERDICTS,
    HypothesisResult,
    compare_groups,
    compare_metrics_by_group,
    format_hypothesis_report,
)
from trade_analysis.analyzer.persistence import (
    SupabaseClient,
    persist_grid_run,
    persist_hypothesis_results,
)

__all__ = [
    # Hypothesis model
    "HypothesisResult",
    "VERDICTS",
    "compare_groups",
    "compare_metrics_by_group",
    "format_hypothesis_report",
    # Evaluators
    "evaluate_h1",
    "evaluate_h2",
    "evaluate_h3",
    "evaluate_h4",
    "evaluate_h5",
    "evaluate_all",
    # Persistence
    "SupabaseClient",
    "persist_grid_run",
    "persist_hypothesis_results",
]
