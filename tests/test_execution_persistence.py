"""Tests for execution persistence (proposals, orders, fills, positions, kill switch).

Mocks SupabaseClient following the same pattern as test_live_persistence.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.execution.persistence import (
    close_position,
    create_fill,
    create_order,
    create_proposal,
    get_kill_switch_status,
    load_open_positions,
    load_pending_proposals,
    load_proposals,
    load_recent_orders,
    set_kill_switch,
    update_order_status,
    update_proposal_status,
    upsert_position,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_sb(enabled: bool = True) -> SupabaseClient:
    """Create a mock SupabaseClient."""
    sb = MagicMock(spec=SupabaseClient)
    sb.enabled = enabled
    sb._enabled = enabled

    mock_table = MagicMock()
    sb.client.table.return_value = mock_table
    return sb


@pytest.fixture()
def sample_proposal() -> dict:
    return {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "Daily",
        "direction": "long",
        "entry_price": 185.0,
        "stop_loss": 180.0,
        "target_price": 195.0,
        "rr_ratio": 2.0,
        "signal_score": 4,
        "regime": "bull",
        "config_hash": "abc123",
        "suggested_qty": 0,
        "status": "pending_approval",
    }


# ---------------------------------------------------------------------------
# Proposal tests
# ---------------------------------------------------------------------------


class TestCreateProposal:
    def test_creates_proposal(self, sample_proposal: dict) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        result = create_proposal(sb, sample_proposal)
        assert result == 1
        sb.client.table.assert_called_with("order_proposals")

    def test_returns_none_when_disabled(self, sample_proposal: dict) -> None:
        sb = _mock_sb(enabled=False)
        result = create_proposal(sb, sample_proposal)
        assert result is None

    def test_returns_none_when_no_data(self, sample_proposal: dict) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.insert.return_value.execute.return_value = MagicMock(data=[])

        result = create_proposal(sb, sample_proposal)
        assert result is None


class TestUpdateProposalStatus:
    def test_updates_status(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.update.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        now = datetime.now(tz=timezone.utc)
        result = update_proposal_status(sb, 1, "approved", decided_at=now)
        assert result is True

    def test_returns_false_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = update_proposal_status(sb, 1, "rejected")
        assert result is False


class TestLoadPendingProposals:
    def test_loads_pending(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.eq.return_value.order.return_value
        chain.execute.return_value = MagicMock(
            data=[{"id": 1, "symbol": "AAPL", "status": "pending_approval"}]
        )

        result = load_pending_proposals(sb)
        assert len(result) == 1

    def test_returns_empty_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = load_pending_proposals(sb)
        assert result == []

    def test_filters_by_symbol(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.eq.return_value.order.return_value
        chain.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1, "symbol": "AAPL"}]
        )

        result = load_pending_proposals(sb, symbol="AAPL")
        assert len(result) == 1


class TestLoadProposals:
    def test_loads_with_limit(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(
            data=[{"id": 1}, {"id": 2}]
        )

        result = load_proposals(sb, limit=10)
        assert len(result) == 2

    def test_filters_by_status(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.order.return_value.limit.return_value
        chain.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 1, "status": "approved"}]
        )

        result = load_proposals(sb, status_filter="approved")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------


class TestCreateOrder:
    def test_creates_order(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 10}]
        )

        order = {
            "proposal_id": 1,
            "symbol": "AAPL",
            "direction": "long",
            "qty": 10,
            "order_type": "market",
            "dry_run": True,
        }
        result = create_order(sb, order)
        assert result == 10
        sb.client.table.assert_called_with("orders")

    def test_returns_none_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = create_order(sb, {})
        assert result is None


class TestUpdateOrderStatus:
    def test_updates(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.update.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 10}]
        )

        result = update_order_status(sb, 10, "filled")
        assert result is True

    def test_returns_false_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = update_order_status(sb, 10, "filled")
        assert result is False


class TestLoadRecentOrders:
    def test_loads_orders(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(data=[{"id": 1}])

        result = load_recent_orders(sb)
        assert len(result) == 1

    def test_returns_empty_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = load_recent_orders(sb)
        assert result == []


# ---------------------------------------------------------------------------
# Fill tests
# ---------------------------------------------------------------------------


class TestCreateFill:
    def test_creates_fill(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 5}]
        )

        fill = {"order_id": 10, "fill_price": 185.25, "fill_qty": 10}
        result = create_fill(sb, fill)
        assert result == 5

    def test_returns_none_when_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = create_fill(sb, {})
        assert result is None


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------


class TestPositions:
    def test_upsert_position(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 3}]
        )

        pos = {
            "symbol": "AAPL",
            "direction": "long",
            "qty": 10,
            "avg_entry_price": 185.0,
            "current_stop": 180.0,
            "current_target": 195.0,
        }
        result = upsert_position(sb, pos)
        assert result == 3

    def test_close_position(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.update.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": 3}]
        )

        result = close_position(sb, 3, "target_hit")
        assert result is True

    def test_load_open_positions(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.is_.return_value.order.return_value
        chain.execute.return_value = MagicMock(
            data=[{"id": 1, "symbol": "AAPL"}]
        )

        result = load_open_positions(sb)
        assert len(result) == 1

    def test_positions_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        assert upsert_position(sb, {}) is None
        assert close_position(sb, 1, "stop") is False
        assert load_open_positions(sb) == []


# ---------------------------------------------------------------------------
# Kill Switch tests
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_get_status_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        assert get_kill_switch_status(sb) is False

    def test_get_status_engaged(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(
            data=[{"enabled": True}]
        )

        assert get_kill_switch_status(sb) is True

    def test_get_status_not_engaged(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(
            data=[{"enabled": False}]
        )

        assert get_kill_switch_status(sb) is False

    def test_get_status_no_rows(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        chain = table.select.return_value.order.return_value.limit.return_value
        chain.execute.return_value = MagicMock(data=[])

        assert get_kill_switch_status(sb) is False

    def test_set_kill_switch(self) -> None:
        sb = _mock_sb()
        table = sb.client.table.return_value
        table.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        result = set_kill_switch(sb, enabled=True, reason="emergency")
        assert result is True

    def test_set_kill_switch_disabled(self) -> None:
        sb = _mock_sb(enabled=False)
        result = set_kill_switch(sb, enabled=True)
        assert result is False
