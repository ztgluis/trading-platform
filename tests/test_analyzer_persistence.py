"""Tests for M6 Supabase persistence (mocked)."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from trade_analysis.analyzer.hypothesis import HypothesisResult
from trade_analysis.analyzer.persistence import (
    SupabaseClient,
    persist_grid_run,
    persist_hypothesis_results,
)


# ===========================================================================
# SupabaseClient
# ===========================================================================


class TestSupabaseClient:
    """Test client initialisation."""

    def test_disabled_without_credentials(self):
        """Client should be disabled when no env vars set."""
        with patch.dict("os.environ", {}, clear=True):
            client = SupabaseClient(url="", key="")
        assert client.enabled is False
        assert client.client is None

    def test_disabled_with_partial_credentials(self):
        """Client should be disabled with only URL."""
        client = SupabaseClient(url="https://example.supabase.co", key="")
        assert client.enabled is False

    def test_disabled_when_supabase_not_installed(self):
        """Client should be disabled if supabase package missing."""
        # supabase-py is not installed in test env, so providing
        # credentials should still result in disabled client
        client = SupabaseClient(
            url="https://example.supabase.co",
            key="fake-key",
        )
        assert client.enabled is False

    def test_enabled_property(self):
        """Enabled should reflect connection state."""
        client = SupabaseClient(url="", key="")
        assert client.enabled is False


# ===========================================================================
# Helpers for mocking
# ===========================================================================


def _mock_sb_client() -> SupabaseClient:
    """Create a SupabaseClient with a mocked raw client."""
    sb = SupabaseClient(url="", key="")
    sb._enabled = True
    sb._client = MagicMock()
    return sb


@dataclass(frozen=True)
class _FakeGridConfig:
    symbol: str = "AAPL"
    asset_class: str = "stock"
    timeframe: str = "Daily"
    parameters: dict = field(default_factory=lambda: {"rsi_period": [10, 14]})
    min_trades: int = 30
    rank_by: str = "total_r"


@dataclass(frozen=True)
class _FakeGridResult:
    total_combos: int = 2
    sufficient_combos: int = 2
    _df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame({
        "rsi_period": [10, 14],
        "total_trades": [50, 60],
        "win_rate": [0.5, 0.6],
        "avg_r": [0.1, 0.2],
        "total_r": [5.0, 12.0],
        "profit_factor": [1.5, 2.0],
        "max_drawdown_r": [-3.0, -2.0],
        "sufficient_trades": [True, True],
    }))

    def to_dataframe(self):
        return self._df


# ===========================================================================
# persist_grid_run
# ===========================================================================


class TestPersistGridRun:
    """Test grid run persistence."""

    def test_returns_none_when_disabled(self):
        """Should return None if client disabled."""
        sb = SupabaseClient(url="", key="")
        result = persist_grid_run(sb, _FakeGridConfig(), _FakeGridResult())
        assert result is None

    def test_inserts_run_and_results(self):
        """Should insert grid_runs row then grid_results rows."""
        sb = _mock_sb_client()

        # Mock insert responses
        run_response = MagicMock()
        run_response.data = [{"id": 42}]
        results_response = MagicMock()
        results_response.data = [{"id": 1}, {"id": 2}]

        run_table = MagicMock()
        run_table.insert.return_value.execute.return_value = run_response
        results_table = MagicMock()
        results_table.insert.return_value.execute.return_value = results_response

        def table_router(name):
            if name == "grid_runs":
                return run_table
            return results_table

        sb._client.table = table_router

        run_id = persist_grid_run(sb, _FakeGridConfig(), _FakeGridResult())
        assert run_id == 42

        # Verify grid_runs insert was called
        run_table.insert.assert_called_once()
        run_data = run_table.insert.call_args[0][0]
        assert run_data["symbol"] == "AAPL"
        assert run_data["total_combos"] == 2

        # Verify grid_results insert was called with 2 rows
        results_table.insert.assert_called_once()
        result_rows = results_table.insert.call_args[0][0]
        assert len(result_rows) == 2
        assert all(r["run_id"] == 42 for r in result_rows)


# ===========================================================================
# persist_hypothesis_results
# ===========================================================================


class TestPersistHypothesisResults:
    """Test hypothesis results persistence."""

    def test_returns_empty_when_disabled(self):
        """Should return empty list if client disabled."""
        sb = SupabaseClient(url="", key="")
        results = [
            HypothesisResult("H1", "test?", "supported", summary="yes"),
        ]
        ids = persist_hypothesis_results(sb, results)
        assert ids == []

    def test_inserts_results(self):
        """Should insert hypothesis_results rows."""
        sb = _mock_sb_client()

        response = MagicMock()
        response.data = [{"id": 100}, {"id": 101}]

        table = MagicMock()
        table.insert.return_value.execute.return_value = response
        sb._client.table = MagicMock(return_value=table)

        results = [
            HypothesisResult("H1", "Does filter help?", "supported",
                             evidence={"diff": 0.3}, summary="Yes."),
            HypothesisResult("H4", "Crossover?", "not_testable",
                             summary="Not implemented."),
        ]

        ids = persist_hypothesis_results(sb, results, grid_run_id=42)
        assert ids == [100, 101]

        # Verify insert call
        table.insert.assert_called_once()
        rows = table.insert.call_args[0][0]
        assert len(rows) == 2
        assert rows[0]["hypothesis_id"] == "H1"
        assert rows[0]["grid_run_id"] == 42
        assert rows[1]["verdict"] == "not_testable"

    def test_without_grid_run_id(self):
        """Should omit grid_run_id if not provided."""
        sb = _mock_sb_client()

        response = MagicMock()
        response.data = [{"id": 200}]

        table = MagicMock()
        table.insert.return_value.execute.return_value = response
        sb._client.table = MagicMock(return_value=table)

        results = [
            HypothesisResult("H1", "test?", "inconclusive"),
        ]

        persist_hypothesis_results(sb, results)
        rows = table.insert.call_args[0][0]
        assert "grid_run_id" not in rows[0]

    def test_empty_results(self):
        """Should return empty list for empty input."""
        sb = _mock_sb_client()
        ids = persist_hypothesis_results(sb, [])
        assert ids == []
