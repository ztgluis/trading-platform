"""Tests for live signal persistence.

Tests the persistence logic with a mocked SupabaseClient,
following the same pattern as test_analyzer_persistence.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.live.persistence import load_recent_signals, persist_signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_signal() -> dict:
    return {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "Daily",
        "bucket": "A",
        "bar_timestamp": "2024-06-15T00:00:00+00:00",
        "regime": "bull",
        "regime_distance_pct": 5.0,
        "regime_strongly_aligned": True,
        "signal_direction": "long",
        "conditions_met": 2,
        "trend_confirmed": True,
        "structure_confirmed": True,
        "structure_multi_method": False,
        "momentum_confirmed": False,
        "volume_spike": True,
        "signal_score": 4,
        "signal_tradeable": True,
        "entry_price": 195.50,
        "exit_stop": 190.00,
        "exit_target": 206.50,
        "exit_trail_be": 200.75,
        "exit_risk": 5.50,
        "exit_reward": 11.00,
        "exit_rr_ratio": 2.0,
        "config_hash": "abc123",
    }


def _mock_sb(enabled: bool = True) -> SupabaseClient:
    """Create a mock SupabaseClient."""
    sb = MagicMock(spec=SupabaseClient)
    sb.enabled = enabled
    sb._enabled = enabled

    # Mock the chained table().upsert().execute() / table().select()...execute()
    mock_table = MagicMock()
    sb.client.table.return_value = mock_table
    return sb


# ---------------------------------------------------------------------------
# persist_signal tests
# ---------------------------------------------------------------------------


class TestPersistSignal:
    def test_inserts_signal(self, sample_signal: dict) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": 42}]
        )

        result = persist_signal(sb, sample_signal)

        assert result == 42
        sb.client.table.assert_called_with("signals")
        table.upsert.assert_called_once()

    def test_returns_none_when_disabled(self, sample_signal: dict) -> None:
        sb = _mock_sb(enabled=False)
        result = persist_signal(sb, sample_signal)
        assert result is None

    def test_returns_none_when_no_data(self, sample_signal: dict) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.upsert.return_value.execute.return_value = MagicMock(data=[])

        result = persist_signal(sb, sample_signal)
        assert result is None


# ---------------------------------------------------------------------------
# load_recent_signals tests
# ---------------------------------------------------------------------------


class TestLoadRecentSignals:
    def test_loads_signals(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        mock_chain = table.select.return_value.order.return_value.limit.return_value
        mock_chain.execute.return_value = MagicMock(
            data=[{"id": 1, "symbol": "AAPL"}, {"id": 2, "symbol": "MSFT"}]
        )

        result = load_recent_signals(sb, limit=10)
        assert len(result) == 2
        assert result[0]["symbol"] == "AAPL"

    def test_returns_empty_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = load_recent_signals(sb)
        assert result == []

    def test_filters_by_symbol(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        mock_chain = table.select.return_value.order.return_value.limit.return_value
        mock_chain.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1, "symbol": "AAPL"}]
        )

        result = load_recent_signals(sb, symbol="AAPL")
        assert len(result) == 1

    def test_filters_tradeable_only(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        mock_chain = table.select.return_value.order.return_value.limit.return_value
        mock_chain.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1, "signal_tradeable": True}]
        )

        result = load_recent_signals(sb, tradeable_only=True)
        assert len(result) == 1
