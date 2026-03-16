"""Reusable Plotly chart builders for the dashboard.

All functions return plotly.graph_objects.Figure instances and contain
no Streamlit rendering calls, making them independently testable.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from trade_analysis.dashboard.components.labels import col_label


# ---------------------------------------------------------------------------
# Parameter heatmap
# ---------------------------------------------------------------------------


def build_param_heatmap(
    df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str,
) -> go.Figure:
    """Build a heatmap of two swept parameters colored by *metric*.

    Pivots the DataFrame so param_x is on the x-axis and param_y is on the
    y-axis.  Cells show the mean of *metric* for each (x, y) combination.
    """
    pivot = df.pivot_table(index=param_y, columns=param_x, values=metric, aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    lx, ly, lm = col_label(param_x), col_label(param_y), col_label(metric)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale="RdYlGn",
            colorbar=dict(title=lm),
            hovertemplate=(
                f"{lx}: %{{x}}<br>{ly}: %{{y}}<br>{lm}: %{{z:.3f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{lm} by {lx} vs {ly}",
        xaxis_title=lx,
        yaxis_title=ly,
        height=450,
    )
    return fig


def build_single_param_bar(
    df: pd.DataFrame,
    param: str,
    metric: str,
) -> go.Figure:
    """Bar chart of a single parameter's average metric values."""
    grouped = df.groupby(param)[metric].mean().sort_index()
    lp, lm = col_label(param), col_label(metric)

    fig = go.Figure(
        go.Bar(
            x=[str(v) for v in grouped.index],
            y=grouped.values,
            marker_color="#2196F3",
            hovertemplate=f"{lp}: %{{x}}<br>{lm}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Average {lm} by {lp}",
        xaxis_title=lp,
        yaxis_title=lm,
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Distribution histogram
# ---------------------------------------------------------------------------


def build_distribution_histogram(
    df: pd.DataFrame,
    metric: str,
    title: str | None = None,
) -> go.Figure:
    """Histogram of a metric across all rows."""
    lm = col_label(metric)
    fig = go.Figure(
        go.Histogram(
            x=df[metric],
            nbinsx=30,
            marker_color="#4CAF50",
            hovertemplate=f"{lm}: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title or f"Distribution of {lm}",
        xaxis_title=lm,
        yaxis_title="Count",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Robustness chart
# ---------------------------------------------------------------------------


def build_robustness_chart(
    robustness_df: pd.DataFrame,
    param_name: str,
    metric: str,
) -> go.Figure:
    """Line chart showing metric vs param value with robustness coloring.

    Points are green if robust, red if isolated peak.
    """
    rdf = robustness_df.sort_values("param_value")
    lp, lm = col_label(param_name), col_label(metric)

    colors = [
        "#F44336" if row.get("is_isolated_peak", False)
        else "#4CAF50" if row.get("is_robust", True)
        else "#FF9800"
        for _, row in rdf.iterrows()
    ]

    fig = go.Figure()

    # Metric line
    fig.add_trace(go.Scatter(
        x=[str(v) for v in rdf["param_value"]],
        y=rdf["metric_avg"],
        mode="lines+markers",
        name=lm,
        marker=dict(color=colors, size=10),
        line=dict(color="#888"),
    ))

    # Neighbor average line
    if "neighbor_avg" in rdf.columns:
        fig.add_trace(go.Scatter(
            x=[str(v) for v in rdf["param_value"]],
            y=rdf["neighbor_avg"],
            mode="lines",
            name="Neighbor avg",
            line=dict(color="#BBDEFB", dash="dash"),
        ))

    fig.update_layout(
        title=f"Robustness: {lm} by {lp}",
        xaxis_title=lp,
        yaxis_title=lm,
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def build_equity_curve(
    trade_df: pd.DataFrame,
    pnl_col: str = "pnl_r",
) -> go.Figure:
    """Cumulative R-multiple equity curve."""
    cumulative = trade_df[pnl_col].cumsum()

    fig = go.Figure(
        go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative.values,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#2196F3"),
            hovertemplate="Trade #%{x}<br>Cumulative R: %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Equity Curve (Cumulative R)",
        xaxis_title="Trade #",
        yaxis_title="Cumulative R",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Breakdown bar charts
# ---------------------------------------------------------------------------


def build_breakdown_bars(
    breakdown: dict[str, dict],
    metric: str,
    title: str,
) -> go.Figure:
    """Grouped bar chart from a stats breakdown dict (by_regime, etc.).

    Each key in *breakdown* is a group name (e.g. "bull", "bear").
    Each value is a dict containing the metric key.
    """
    groups = sorted(breakdown.keys())
    values = [breakdown[g].get(metric, 0) for g in groups]
    lm = col_label(metric)

    fig = go.Figure(
        go.Bar(
            x=groups,
            y=values,
            marker_color="#2196F3",
            hovertemplate="%{x}<br>" + lm + ": %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=lm,
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Radar comparison
# ---------------------------------------------------------------------------


RADAR_METRICS = ["win_rate", "avg_r", "profit_factor", "total_r"]


def build_radar_comparison(
    selected_rows: list[dict],
    metrics: list[str] | None = None,
) -> go.Figure:
    """Radar chart comparing multiple parameter combos across metrics.

    Each entry in *selected_rows* should be a dict with metric keys.
    """
    metrics = metrics or RADAR_METRICS
    fig = go.Figure()

    for i, row in enumerate(selected_rows):
        label = row.get("_label", f"Combo {i + 1}")
        values = [row.get(m, 0) for m in metrics]
        # Close the polygon
        values.append(values[0])
        theta = [col_label(m) for m in metrics] + [col_label(metrics[0])]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            name=label,
            fill="toself",
            opacity=0.6,
        ))

    fig.update_layout(
        title="Strategy Comparison",
        polar=dict(radialaxis=dict(visible=True)),
        height=450,
    )
    return fig
