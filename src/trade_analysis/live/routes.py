"""API routes for the live runner."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from trade_analysis.live.models import (
    HealthResponse,
    SignalSummary,
    WebhookRequest,
    WebhookResponse,
)
from trade_analysis.live.persistence import persist_signal
from trade_analysis.live.signal_handler import handle_scan, lookup_symbol

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        supabase_connected=request.app.state.supabase.enabled,
        symbols_configured=len(request.app.state.symbols),
    )


@router.post("/webhook", response_model=WebhookResponse)
async def webhook(payload: WebhookRequest, request: Request) -> WebhookResponse:
    """Receive a TradingView alert and run the signal engine.

    Validates the shared secret, looks up the symbol, runs the signal
    pipeline, and persists tradeable signals to Supabase.
    """
    config = request.app.state.live_config

    # ---- Auth ----
    if config.webhook_secret and payload.secret != config.webhook_secret:
        logger.warning("Invalid webhook secret from request")
        return WebhookResponse(status="error", message="Invalid secret")

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

    # ---- Persist ----
    persisted = False
    should_persist = signal_row["signal_tradeable"] or config.persist_all_signals
    if should_persist:
        sb = request.app.state.supabase
        signal_id = persist_signal(sb, signal_row)
        persisted = signal_id is not None

    return WebhookResponse(
        status="ok",
        message=(
            f"Signal: {signal_row['signal_direction']} {signal_row['symbol']} "
            f"score={signal_row['signal_score']} tradeable={signal_row['signal_tradeable']}"
        ),
        signal=summary,
        persisted=persisted,
    )
