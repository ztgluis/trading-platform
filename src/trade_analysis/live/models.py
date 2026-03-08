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


class WebhookResponse(BaseModel):
    """Response from the webhook endpoint."""

    status: str = Field(..., description="'ok', 'error', or 'no_signal'")
    message: str = Field(default="", description="Human-readable message")
    signal: SignalSummary | None = Field(default=None, description="Signal summary if generated")
    persisted: bool = Field(default=False, description="Whether the signal was saved to Supabase")


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""

    status: str = "ok"
    supabase_connected: bool = False
    symbols_configured: int = 0
