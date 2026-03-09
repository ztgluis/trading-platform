"""API routes for the live runner."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from trade_analysis.execution.persistence import (
    get_kill_switch_status,
    load_open_positions,
    load_pending_proposals,
)
from trade_analysis.live.models import (
    ApproveRequest,
    HealthResponse,
    KillSwitchRequest,
    ProposalSummary,
    RejectRequest,
    SignalSummary,
    WebhookRequest,
    WebhookResponse,
)
from trade_analysis.live.persistence import persist_signal
from trade_analysis.live.signal_handler import handle_scan, lookup_symbol

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def _check_secret(secret: str, request: Request) -> str | None:
    """Validate the shared secret. Returns error message or None if OK."""
    config = request.app.state.live_config
    if config.webhook_secret and secret != config.webhook_secret:
        return "Invalid secret"
    return None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    config = request.app.state.live_config
    sb = request.app.state.supabase
    return HealthResponse(
        status="ok",
        supabase_connected=sb.enabled,
        symbols_configured=len(request.app.state.symbols),
        execution_mode="dry_run" if config.execution.dry_run else "live",
        kill_switch_engaged=get_kill_switch_status(sb),
    )


# ---------------------------------------------------------------------------
# Webhook (signal pipeline + proposal creation)
# ---------------------------------------------------------------------------


@router.post("/webhook", response_model=WebhookResponse)
async def webhook(payload: WebhookRequest, request: Request) -> WebhookResponse:
    """Receive a TradingView alert and run the signal engine.

    Validates the shared secret, looks up the symbol, runs the signal
    pipeline, persists tradeable signals, and creates order proposals.
    """
    config = request.app.state.live_config

    # ---- Auth ----
    auth_err = _check_secret(payload.secret, request)
    if auth_err:
        logger.warning("Invalid webhook secret from request")
        return WebhookResponse(status="error", message=auth_err)

    # ---- Symbol validation ----
    sym_config = lookup_symbol(payload.symbol, request.app.state.symbols)
    if sym_config is None:
        logger.warning("Unknown symbol: %s", payload.symbol)
        return WebhookResponse(
            status="error",
            message=f"Unknown symbol: {payload.symbol}",
        )

    # ---- Timeframe validation ----
    if payload.timeframe not in sym_config.timeframes:
        logger.warning(
            "Timeframe %s not configured for %s (available: %s)",
            payload.timeframe,
            payload.symbol,
            sym_config.timeframes,
        )
        return WebhookResponse(
            status="error",
            message=f"Timeframe {payload.timeframe} not configured for {payload.symbol}",
        )

    # ---- Action dispatch ----
    if payload.action != "scan":
        return WebhookResponse(
            status="error",
            message=f"Unknown action: {payload.action}. Supported: 'scan'",
        )

    # ---- Run signal pipeline ----
    try:
        signal_row = handle_scan(
            symbol=payload.symbol,
            timeframe=payload.timeframe,
            asset_class=sym_config.asset_class,
            dm=request.app.state.data_manager,
            signal_config=request.app.state.signal_config,
            force_refresh=config.force_refresh_ohlcv,
        )
    except Exception as exc:
        logger.exception("Signal pipeline failed for %s %s", payload.symbol, payload.timeframe)
        return WebhookResponse(status="error", message=f"Pipeline error: {exc}")

    # ---- No signal ----
    if signal_row is None:
        return WebhookResponse(status="no_signal", message="No signal on latest bar")

    # ---- Build summary ----
    summary = SignalSummary(
        symbol=signal_row["symbol"],
        timeframe=signal_row["timeframe"],
        direction=signal_row["signal_direction"],
        score=signal_row["signal_score"],
        tradeable=signal_row["signal_tradeable"],
        regime=signal_row["regime"],
        entry_price=signal_row.get("entry_price"),
        exit_stop=signal_row.get("exit_stop"),
        exit_target=signal_row.get("exit_target"),
        rr_ratio=signal_row.get("exit_rr_ratio"),
        bar_timestamp=signal_row.get("bar_timestamp"),
    )

    # ---- Persist signal ----
    persisted = False
    should_persist = signal_row["signal_tradeable"] or config.persist_all_signals
    if should_persist:
        sb = request.app.state.supabase
        signal_id = persist_signal(sb, signal_row)
        persisted = signal_id is not None
        if signal_id is not None:
            signal_row["signal_id"] = signal_id

    # ---- Create order proposal for tradeable signals ----
    proposal_summary = None
    if signal_row["signal_tradeable"]:
        executor = request.app.state.executor
        proposal = executor.create_proposal(signal_row)
        if proposal:
            proposal_summary = ProposalSummary(
                proposal_id=proposal.get("id"),
                symbol=proposal["symbol"],
                direction=proposal["direction"],
                entry_price=proposal["entry_price"],
                stop_loss=proposal["stop_loss"],
                target_price=proposal["target_price"],
                rr_ratio=proposal.get("rr_ratio", 0.0),
                signal_score=proposal.get("signal_score", 0),
            )

    return WebhookResponse(
        status="ok",
        message=(
            f"Signal: {signal_row['signal_direction']} {signal_row['symbol']} "
            f"score={signal_row['signal_score']} tradeable={signal_row['signal_tradeable']}"
        ),
        signal=summary,
        persisted=persisted,
        proposal=proposal_summary,
    )


# ---------------------------------------------------------------------------
# Proposal management endpoints
# ---------------------------------------------------------------------------


@router.get("/proposals")
async def list_proposals(request: Request):
    """List pending order proposals."""
    sb = request.app.state.supabase
    proposals = load_pending_proposals(sb)
    return {"status": "ok", "proposals": proposals}


@router.post("/proposals/{proposal_id}/approve")
async def approve_proposal(proposal_id: int, payload: ApproveRequest, request: Request):
    """Approve a proposal and place an order."""
    auth_err = _check_secret(payload.secret, request)
    if auth_err:
        return JSONResponse(status_code=401, content={"status": "error", "message": auth_err})

    executor = request.app.state.executor
    result = executor.approve_proposal(
        proposal_id=proposal_id,
        qty=payload.qty,
        order_type=payload.order_type,
        limit_price=payload.limit_price,
    )
    return result


@router.post("/proposals/{proposal_id}/reject")
async def reject_proposal(proposal_id: int, payload: RejectRequest, request: Request):
    """Reject a proposal."""
    auth_err = _check_secret(payload.secret, request)
    if auth_err:
        return JSONResponse(status_code=401, content={"status": "error", "message": auth_err})

    executor = request.app.state.executor
    updated = executor.reject_proposal(proposal_id, reason=payload.reason)
    return {"status": "ok" if updated else "error", "proposal_id": proposal_id}


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@router.get("/positions")
async def list_positions(request: Request):
    """List open positions."""
    sb = request.app.state.supabase
    positions = load_open_positions(sb)
    return {"status": "ok", "positions": positions}


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


@router.post("/kill-switch")
async def toggle_kill_switch(payload: KillSwitchRequest, request: Request):
    """Toggle the kill switch."""
    auth_err = _check_secret(payload.secret, request)
    if auth_err:
        return JSONResponse(status_code=401, content={"status": "error", "message": auth_err})

    executor = request.app.state.executor
    if payload.enabled:
        executor.risk.engage_kill_switch(payload.reason or "manual toggle")
    else:
        executor.risk.disengage_kill_switch()

    return {
        "status": "ok",
        "kill_switch_engaged": payload.enabled,
        "reason": payload.reason,
    }


# ---------------------------------------------------------------------------
# Execution health
# ---------------------------------------------------------------------------


@router.get("/execution/health")
async def execution_health(request: Request):
    """Schwab client status and risk manager state."""
    executor = request.app.state.executor
    return {
        "status": "ok",
        "schwab": executor.schwab.health_check(),
        "kill_switch_engaged": executor.risk.is_kill_switch_engaged(),
        "execution_config": {
            "dry_run": executor.risk.config.dry_run,
            "max_positions": executor.risk.config.max_positions,
            "max_per_symbol": executor.risk.config.max_per_symbol,
        },
    }
