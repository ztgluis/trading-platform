"""FastAPI application for the live runner webhook server.

Run with: uvicorn src.trade_analysis.live.app:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.config.loader import load_symbols
from trade_analysis.data_manager import DataManager
from trade_analysis.live.config import load_live_config
from trade_analysis.signals import load_signal_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    config = load_live_config()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    app.state.live_config = config
    app.state.data_manager = DataManager()
    app.state.supabase = SupabaseClient()
    app.state.signal_config = load_signal_config()
    app.state.symbols = load_symbols()

    logger.info("Live runner started on %s:%d", config.host, config.port)
    logger.info("Supabase: %s", "connected" if app.state.supabase.enabled else "disabled")
    logger.info("Symbols configured: %d", len(app.state.symbols))

    yield

    logger.info("Live runner shutting down")


app = FastAPI(
    title="Trade Analysis Live Runner",
    description="TradingView webhook → Signal Engine → Supabase",
    version="0.1.0",
    lifespan=lifespan,
)

# Import routes after app creation to avoid circular imports
from trade_analysis.live.routes import router  # noqa: E402

app.include_router(router)
