"""Pydantic models for order execution pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProposalStatus(str, Enum):
    """Lifecycle states for an order proposal."""

    PENDING = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderStatus(str, Enum):
    """Lifecycle states for an order placed with the broker."""

    PLACED = "placed"
    FILLED = "filled"
    PARTIAL = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(str, Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"


class Direction(str, Enum):
    """Trade direction."""

    LONG = "long"
    SHORT = "short"


# ---------------------------------------------------------------------------
# Order Proposal — generated from tradeable signals
# ---------------------------------------------------------------------------


class OrderProposal(BaseModel):
    """A proposed trade awaiting user approval."""

    id: int | None = None
    symbol: str
    asset_class: str
    timeframe: str
    direction: Direction
    entry_price: float
    stop_loss: float
    target_price: float
    rr_ratio: float = 0.0
    signal_score: int = 0
    regime: str | None = None
    config_hash: str = ""
    signal_id: int | None = None
    suggested_qty: int = 0
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime | None = None
    decided_at: datetime | None = None


# ---------------------------------------------------------------------------
# Order — placed with the broker (or simulated in dry-run)
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    """User request to approve a proposal and place an order."""

    proposal_id: int
    qty: int = Field(..., gt=0, description="Number of shares/units")
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = Field(
        default=None,
        description="Required when order_type is 'limit'",
    )


class Order(BaseModel):
    """A placed order (real or simulated)."""

    id: int | None = None
    proposal_id: int
    schwab_order_id: str | None = None
    symbol: str
    direction: Direction
    qty: int
    order_type: OrderType
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PLACED
    dry_run: bool = True
    placed_at: datetime | None = None
    updated_at: datetime | None = None


class Fill(BaseModel):
    """An order fill (partial or complete)."""

    id: int | None = None
    order_id: int
    fill_price: float
    fill_qty: int
    commission: float = 0.0
    filled_at: datetime | None = None


# ---------------------------------------------------------------------------
# Live Position — open position tracking
# ---------------------------------------------------------------------------


class LivePosition(BaseModel):
    """An open position (real or simulated)."""

    id: int | None = None
    symbol: str
    direction: Direction
    qty: int
    avg_entry_price: float
    current_stop: float
    current_target: float
    order_id: int | None = None
    unrealized_pnl: float = 0.0
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None


# ---------------------------------------------------------------------------
# Portfolio Snapshot
# ---------------------------------------------------------------------------


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio state."""

    total_value: float = 0.0
    cash_available: float = 0.0
    buying_power: float = 0.0
    open_positions: int = 0
    unrealized_pnl: float = 0.0
    timestamp: datetime | None = None


# ---------------------------------------------------------------------------
# Dashboard action — user approves/rejects a proposal
# ---------------------------------------------------------------------------


class ProposalAction(BaseModel):
    """Action taken on a proposal from the dashboard."""

    proposal_id: int
    action: str = Field(..., pattern="^(approve|reject)$")
    qty: int | None = Field(default=None, gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reason: str = ""
