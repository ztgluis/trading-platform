"""Tests for execution Pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from trade_analysis.execution.models import (
    Direction,
    Fill,
    LivePosition,
    Order,
    OrderProposal,
    OrderRequest,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    ProposalAction,
    ProposalStatus,
)


# ---------------------------------------------------------------------------
# OrderProposal
# ---------------------------------------------------------------------------


class TestOrderProposal:
    def test_minimal_proposal(self):
        p = OrderProposal(
            symbol="AAPL",
            asset_class="stock",
            timeframe="Daily",
            direction=Direction.LONG,
            entry_price=185.0,
            stop_loss=180.0,
            target_price=195.0,
        )
        assert p.symbol == "AAPL"
        assert p.direction == Direction.LONG
        assert p.status == ProposalStatus.PENDING
        assert p.id is None
        assert p.suggested_qty == 0

    def test_full_proposal(self):
        now = datetime.now(tz=timezone.utc)
        p = OrderProposal(
            id=42,
            symbol="MSFT",
            asset_class="stock",
            timeframe="Daily",
            direction=Direction.SHORT,
            entry_price=400.0,
            stop_loss=410.0,
            target_price=380.0,
            rr_ratio=2.0,
            signal_score=5,
            regime="bear",
            config_hash="abc123",
            signal_id=99,
            suggested_qty=10,
            status=ProposalStatus.APPROVED,
            created_at=now,
            decided_at=now,
        )
        assert p.id == 42
        assert p.status == ProposalStatus.APPROVED
        assert p.rr_ratio == 2.0

    def test_proposal_status_values(self):
        assert ProposalStatus.PENDING == "pending_approval"
        assert ProposalStatus.APPROVED == "approved"
        assert ProposalStatus.REJECTED == "rejected"
        assert ProposalStatus.EXPIRED == "expired"


# ---------------------------------------------------------------------------
# OrderRequest
# ---------------------------------------------------------------------------


class TestOrderRequest:
    def test_valid_request(self):
        r = OrderRequest(proposal_id=1, qty=10)
        assert r.order_type == OrderType.MARKET
        assert r.limit_price is None

    def test_limit_order(self):
        r = OrderRequest(
            proposal_id=1, qty=5, order_type=OrderType.LIMIT, limit_price=185.50
        )
        assert r.order_type == OrderType.LIMIT
        assert r.limit_price == 185.50

    def test_qty_must_be_positive(self):
        with pytest.raises(ValidationError):
            OrderRequest(proposal_id=1, qty=0)

    def test_qty_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            OrderRequest(proposal_id=1, qty=-5)


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


class TestOrder:
    def test_defaults(self):
        o = Order(
            proposal_id=1,
            symbol="AAPL",
            direction=Direction.LONG,
            qty=10,
            order_type=OrderType.MARKET,
        )
        assert o.status == OrderStatus.PLACED
        assert o.dry_run is True
        assert o.schwab_order_id is None

    def test_order_status_values(self):
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.PARTIAL == "partially_filled"
        assert OrderStatus.CANCELLED == "cancelled"


# ---------------------------------------------------------------------------
# Fill
# ---------------------------------------------------------------------------


class TestFill:
    def test_fill(self):
        f = Fill(order_id=1, fill_price=185.25, fill_qty=10)
        assert f.commission == 0.0
        assert f.filled_at is None

    def test_fill_with_commission(self):
        f = Fill(order_id=1, fill_price=185.25, fill_qty=10, commission=0.65)
        assert f.commission == 0.65


# ---------------------------------------------------------------------------
# LivePosition
# ---------------------------------------------------------------------------


class TestLivePosition:
    def test_open_position(self):
        pos = LivePosition(
            symbol="AAPL",
            direction=Direction.LONG,
            qty=10,
            avg_entry_price=185.0,
            current_stop=180.0,
            current_target=195.0,
        )
        assert pos.closed_at is None
        assert pos.close_reason is None
        assert pos.unrealized_pnl == 0.0


# ---------------------------------------------------------------------------
# PortfolioSnapshot
# ---------------------------------------------------------------------------


class TestPortfolioSnapshot:
    def test_defaults(self):
        snap = PortfolioSnapshot()
        assert snap.total_value == 0.0
        assert snap.open_positions == 0


# ---------------------------------------------------------------------------
# ProposalAction
# ---------------------------------------------------------------------------


class TestProposalAction:
    def test_approve(self):
        a = ProposalAction(proposal_id=1, action="approve", qty=10)
        assert a.action == "approve"

    def test_reject(self):
        a = ProposalAction(proposal_id=1, action="reject", reason="not confident")
        assert a.action == "reject"
        assert a.qty is None

    def test_invalid_action(self):
        with pytest.raises(ValidationError):
            ProposalAction(proposal_id=1, action="cancel")


# ---------------------------------------------------------------------------
# Direction / OrderType enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_direction_values(self):
        assert Direction.LONG == "long"
        assert Direction.SHORT == "short"

    def test_order_type_values(self):
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
