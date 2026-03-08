"""Reusable metric display helpers."""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Verdict badge
# ---------------------------------------------------------------------------

VERDICT_COLORS = {
    "supported": "#4CAF50",
    "refuted": "#F44336",
    "inconclusive": "#FF9800",
    "not_testable": "#9E9E9E",
}

VERDICT_ICONS = {
    "supported": "+",
    "refuted": "-",
    "inconclusive": "~",
    "not_testable": "?",
}


def render_verdict_badge(verdict: str) -> str:
    """Return HTML for a colored verdict badge."""
    color = VERDICT_COLORS.get(verdict, "#9E9E9E")
    icon = VERDICT_ICONS.get(verdict, "?")
    return (
        f'<span style="background-color:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;font-size:0.85em;">'
        f"[{icon}] {verdict.upper()}</span>"
    )


# ---------------------------------------------------------------------------
# Metric cards row
# ---------------------------------------------------------------------------


def render_metric_row(metrics: dict[str, tuple[float | int, str | None]]) -> None:
    """Render a row of st.metric cards.

    *metrics* maps label → (value, optional_delta_text).
    """
    cols = st.columns(len(metrics))
    for col, (label, (value, delta)) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, float):
                st.metric(label, f"{value:.3f}", delta=delta)
            else:
                st.metric(label, value, delta=delta)
