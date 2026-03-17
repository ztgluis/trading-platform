"""Hypotheses page — H1-H5 verdict cards + evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from trade_analysis.dashboard.components.metrics import render_verdict_badge
from trade_analysis.dashboard.data_loader import load_hypothesis_results


def _render_evidence(hypothesis_id: str, evidence: dict[str, Any]) -> None:
    """Render evidence details specific to each hypothesis type."""
    if hypothesis_id == "H1":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Period", evidence.get("best_period", "N/A"))
            st.metric("Best Period Avg R-Multiple", f"{evidence.get('best_period_avg_r', 0):.3f}")
        with col2:
            st.metric("Difference vs No Filter", f"{evidence.get('difference', 0):+.3f}")
            st.metric(
                "Smallest Period Avg R-Multiple",
                f"{evidence.get('smallest_period_avg_r', 0):.3f}",
            )

    elif hypothesis_id == "H2":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Type Spread", f"{evidence.get('type_spread', 0):.4f}")
            st.metric("Best Type", evidence.get("best_type", "N/A"))
        with col2:
            st.metric("Period Spread", f"{evidence.get('period_spread', 0):.4f}")
        type_groups = evidence.get("type_groups", {})
        if type_groups:
            st.markdown("**By MA Type:**")
            for t, stats in type_groups.items():
                st.markdown(f"- {t}: Avg R = {stats.get('mean', 0):.3f}, n = {stats.get('count', 0)}")

    elif hypothesis_id == "H3":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Period", evidence.get("best_period", "N/A"))
            st.metric("Best Avg R-Multiple", f"{evidence.get('best_avg_r', 0):.3f}")
        with col2:
            robust = evidence.get("is_robust", False)
            st.metric("Robust?", "Yes" if robust else "No")
        ranking = evidence.get("ranking", [])
        if ranking:
            st.markdown("**Ranking:**")
            for entry in ranking[:10]:
                st.markdown(
                    f"- Period {entry.get('period', '?')}: Avg R = {entry.get('avg_r', 0):.3f}"
                )

    elif hypothesis_id == "H5":
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Win Rate Diff", f"{evidence.get('win_rate_diff', 0):+.1%}")
        with col2:
            st.metric("Trade Count Diff", f"{evidence.get('trade_count_diff', 0):+.0f}")
        wr_by_param = evidence.get("win_rate_by_param", {})
        if wr_by_param:
            st.markdown("**Win Rate by Parameter Value:**")
            for val, wr in wr_by_param.items():
                st.markdown(f"- {val}: {wr:.1%}")

    else:
        # Generic fallback (H4 or unknown)
        if evidence:
            st.json(evidence)


def render(grid_df: pd.DataFrame, run_meta: pd.Series | None) -> None:
    """Render the hypotheses page."""
    st.header("Hypothesis Results")

    if grid_df.empty:
        st.info("No grid results loaded.")
        return

    # ---- Load or evaluate hypotheses ----
    hypothesis_data: list[dict[str, Any]] = []

    # Try loading from Supabase
    if run_meta is not None and "id" in run_meta.index:
        hypothesis_data = load_hypothesis_results(int(run_meta["id"]))

    # Re-evaluate button or fallback
    if not hypothesis_data or st.button("Re-evaluate Hypotheses"):
        with st.spinner("Evaluating H1-H5..."):
            from trade_analysis.analyzer import evaluate_all

            results = evaluate_all(grid_df)
            hypothesis_data = [
                {
                    "hypothesis_id": r.hypothesis_id,
                    "question": r.question,
                    "verdict": r.verdict,
                    "evidence": r.evidence,
                    "summary": r.summary,
                }
                for r in results
            ]

    if not hypothesis_data:
        st.warning("No hypothesis results available.")
        return

    # ---- Verdict summary bar ----
    badge_cols = st.columns(len(hypothesis_data))
    for col, h in zip(badge_cols, hypothesis_data):
        with col:
            badge = render_verdict_badge(h["verdict"])
            st.markdown(f"**{h['hypothesis_id']}** {badge}", unsafe_allow_html=True)

    st.markdown("---")

    # ---- Hypothesis cards ----
    for h in hypothesis_data:
        verdict = h["verdict"]
        hid = h["hypothesis_id"]
        expanded = verdict == "supported"

        with st.expander(f"{hid}: {h['question']}", expanded=expanded):
            badge = render_verdict_badge(verdict)
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"**{h['summary']}**")
            st.markdown("---")
            _render_evidence(hid, h.get("evidence", {}))
