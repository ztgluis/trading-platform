"""Tests for M6 hypothesis result model and comparison utilities."""

import pandas as pd
import pytest

from trade_analysis.analyzer.hypothesis import (
    VERDICTS,
    HypothesisResult,
    compare_groups,
    compare_metrics_by_group,
    format_hypothesis_report,
)


# ===========================================================================
# HypothesisResult model
# ===========================================================================


class TestHypothesisResult:
    """Test HypothesisResult dataclass."""

    def test_create_result(self):
        """Basic creation works."""
        r = HypothesisResult(
            hypothesis_id="H1",
            question="Does trend filter help?",
            verdict="supported",
            evidence={"best_period": 20, "avg_r": 0.5},
            summary="Trend filter improves avg R by 0.3.",
        )
        assert r.hypothesis_id == "H1"
        assert r.verdict == "supported"

    def test_all_valid_verdicts(self):
        """All valid verdicts should work."""
        for verdict in VERDICTS:
            r = HypothesisResult(
                hypothesis_id="H1",
                question="test",
                verdict=verdict,
            )
            assert r.verdict == verdict

    def test_invalid_verdict_raises(self):
        """Invalid verdict should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid verdict"):
            HypothesisResult(
                hypothesis_id="H1",
                question="test",
                verdict="maybe",
            )

    def test_frozen(self):
        """HypothesisResult should be immutable."""
        r = HypothesisResult(
            hypothesis_id="H1",
            question="test",
            verdict="supported",
        )
        with pytest.raises(AttributeError):
            r.verdict = "refuted"

    def test_defaults(self):
        """Default evidence and summary."""
        r = HypothesisResult(
            hypothesis_id="H1",
            question="test",
            verdict="inconclusive",
        )
        assert r.evidence == {}
        assert r.summary == ""


# ===========================================================================
# compare_groups
# ===========================================================================


class TestCompareGroups:
    """Test group comparison utility."""

    def test_basic_comparison(self):
        """Compare two groups by a metric."""
        df = pd.DataFrame({
            "trend_ma_type": ["ema", "ema", "sma", "sma"],
            "total_r": [5.0, 7.0, 3.0, 4.0],
        })
        result = compare_groups(df, "trend_ma_type", "total_r")
        assert result["best_group"] == "ema"
        assert result["worst_group"] == "sma"
        assert result["groups"]["ema"]["mean"] == pytest.approx(6.0)
        assert result["groups"]["sma"]["mean"] == pytest.approx(3.5)
        assert result["spread"] == pytest.approx(2.5)

    def test_missing_column_returns_empty(self):
        """Missing group column returns empty dict."""
        df = pd.DataFrame({"total_r": [1.0, 2.0]})
        result = compare_groups(df, "nonexistent", "total_r")
        assert result == {}

    def test_missing_metric_returns_empty(self):
        """Missing metric column returns empty dict."""
        df = pd.DataFrame({"group": ["a", "b"]})
        result = compare_groups(df, "group", "nonexistent")
        assert result == {}

    def test_single_group(self):
        """Single group should have spread of 0."""
        df = pd.DataFrame({
            "group": ["a", "a", "a"],
            "total_r": [1.0, 2.0, 3.0],
        })
        result = compare_groups(df, "group", "total_r")
        assert result["spread"] == pytest.approx(0.0)

    def test_includes_count(self):
        """Each group should include the count of rows."""
        df = pd.DataFrame({
            "group": ["a", "a", "b"],
            "total_r": [1.0, 2.0, 3.0],
        })
        result = compare_groups(df, "group", "total_r")
        assert result["groups"]["a"]["count"] == 2
        assert result["groups"]["b"]["count"] == 1


# ===========================================================================
# compare_metrics_by_group
# ===========================================================================


class TestCompareMetricsByGroup:
    """Test multi-metric group comparison."""

    def test_compares_multiple_metrics(self):
        """Should return results for each metric."""
        df = pd.DataFrame({
            "group": ["a", "a", "b", "b"],
            "total_r": [5.0, 7.0, 3.0, 4.0],
            "win_rate": [0.6, 0.7, 0.4, 0.5],
        })
        result = compare_metrics_by_group(
            df, "group", metrics=["total_r", "win_rate"]
        )
        assert "total_r" in result
        assert "win_rate" in result

    def test_skips_missing_metrics(self):
        """Should skip metrics not in DataFrame."""
        df = pd.DataFrame({
            "group": ["a", "b"],
            "total_r": [1.0, 2.0],
        })
        result = compare_metrics_by_group(
            df, "group", metrics=["total_r", "nonexistent"]
        )
        assert "total_r" in result
        assert "nonexistent" not in result


# ===========================================================================
# format_hypothesis_report
# ===========================================================================


class TestFormatReport:
    """Test report formatting."""

    def test_report_with_results(self):
        """Report should include all hypothesis results."""
        results = [
            HypothesisResult("H1", "Does filter help?", "supported",
                             summary="Yes, filter improves by 0.3R."),
            HypothesisResult("H2", "EMA vs SMA?", "inconclusive",
                             summary="No significant difference."),
        ]
        report = format_hypothesis_report(results)
        assert "HYPOTHESIS EVALUATION REPORT" in report
        assert "H1" in report
        assert "SUPPORTED" in report
        assert "H2" in report
        assert "INCONCLUSIVE" in report

    def test_empty_results(self):
        """Empty results should still produce a report."""
        report = format_hypothesis_report([])
        assert "HYPOTHESIS EVALUATION REPORT" in report

    def test_verdict_icons(self):
        """Each verdict type should have an icon."""
        results = [
            HypothesisResult("H1", "q", "supported", summary="s"),
            HypothesisResult("H2", "q", "refuted", summary="s"),
            HypothesisResult("H3", "q", "inconclusive", summary="s"),
            HypothesisResult("H4", "q", "not_testable", summary="s"),
        ]
        report = format_hypothesis_report(results)
        assert "[+]" in report
        assert "[-]" in report
        assert "[?]" in report
        assert "[~]" in report
