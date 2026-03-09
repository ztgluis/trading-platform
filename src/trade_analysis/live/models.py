"""Pydantic models for webhook request/response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebhookRequest(BaseModel):
    """Incoming TradingView webhook payload."""

    secret: str = Field(..., description="Shared secret for authentication")
    symbol: str = Field(..., description="Ticker symbol, e.g. 'AAPL' or 'BTC/USDT'")
    timeframe: str = Field(..., description="Timeframe, e.g. 'Daily', '4H', 'Weekly'")
    action: str = Field(default="scan", description="Action to perform: 'scan'")


class SignalSummary(BaseModel):
    """Summary of a generated signal for the webhook response."""

    symbol: str
    timeframe: str
    direction: str | None = None
    score: int = 0
    tradeable: bool = False
    regime: str | None = None
    entry_price: float | None = None
    exit_stop: float | None = None
    exit_target: float | None = None
    rr_ratio: float | None = None
    bar_timestamp: str | None = None


class ProposalSummary(BaseModel):
    """Summary of a created order proposal."""

    proposal_id: int | None = None
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    target_price: float
    rr_ratio: float = 0.0
    signal_score: int = 0
    status: str = "pending_approval"


class WebhookResponse(BaseModel):
    """Response from the webhook endpoint."""

    status: str = Field(..., description="'ok', 'error', or 'no_signal'")
    message: str = Field(default="", description="Human-readable message")
    signal: SignalSummary | None = Field(default=None, description="Signal summary if generated")
    persisted: bool = Field(default=False, description="Whether the signal was saved to Supabase")
    proposal: ProposalSummary | None = Field(
        default=None, description="Order proposal if a tradeable signal was found"
    )


class ApproveRequest(BaseModel):
    """Request to approve an order proposal."""

    secret: str = Field(..., description="Shared secret for authentication")
    qty: int = Field(..., gt=0, description="Number of shares/units")
    order_type: str = Field(default="market", description="'market' or 'limit'")
    limit_price: float | None = None


class RejectRequest(BaseModel):
    """Request to reject an order proposal."""

    secret: str = Field(..., description="Shared secret for authentication")
    reason: str = Field(default="manual", description="Reason for rejection")


class KillSwitchRequest(BaseModel):
    """Request to toggle the kill switch."""

    secret: str = Field(..., description="Shared secret for authentication")
    enabled: bool = Field(..., description="True to engage, False to disengage")
    reason: str = Field(default="", description="Reason for toggling")


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""

    status: str = "ok"
    supabase_connected: bool = False
    symbols_configured: int = 0
    execution_mode: str = "dry_run"
    kill_switch_engaged: bool = False
