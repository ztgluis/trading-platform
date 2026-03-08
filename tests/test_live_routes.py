"""Tests for live runner API routes using FastAPI TestClient.

Uses mocked DataManager and SupabaseClient to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trade_analysis.config.loader import SymbolConfig
from trade_analysis.live.config import LiveConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_symbols() -> list[SymbolConfig]:
    return [
        SymbolConfig(
            ticker="AAPL",
            asset_class="stock",
            provider="yfinance",
            timeframes=["4H", "Daily"],
        ),
        SymbolConfig(
            ticker="BTC/USDT",
            asset_class="crypto",
            provider="ccxt",
            timeframes=["4H", "Daily", "Weekly"],
            exchange="binance",
        ),
    ]


@pytest.fixture()
def app_client(mock_symbols: list[SymbolConfig]) -> TestClient:
    """Create a TestClient with mocked app state."""
    from trade_analysis.live.app import app

    # Override app state directly
    app.state.live_config = LiveConfig(webhook_secret="test-secret")
    app.state.data_manager = MagicMock()
    app.state.supabase = MagicMock()
    app.state.supabase.enabled = True
    app.state.signal_config = MagicMock()
    app.state.symbols = mock_symbols

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, app_client: TestClient) -> None:
        response = app_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["supabase_connected"] is True
        assert data["symbols_configured"] == 2


# ---------------------------------------------------------------------------
# Webhook endpoint tests
# ---------------------------------------------------------------------------


class TestWebhook:
    def test_invalid_secret(self, app_client: TestClient) -> None:
        response = app_client.post("/webhook", json={
            "secret": "wrong",
            "symbol": "AAPL",
            "timeframe": "Daily",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Invalid secret" in data["message"]

    def test_unknown_symbol(self, app_client: TestClient) -> None:
        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "ZZZZ",
            "timeframe": "Daily",
        })
        data = response.json()
        assert data["status"] == "error"
        assert "Unknown symbol" in data["message"]

    def test_invalid_timeframe(self, app_client: TestClient) -> None:
        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "AAPL",
            "timeframe": "Monthly",
        })
        data = response.json()
        assert data["status"] == "error"
        assert "not configured" in data["message"]

    def test_unknown_action(self, app_client: TestClient) -> None:
        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "AAPL",
            "timeframe": "Daily",
            "action": "execute",
        })
        data = response.json()
        assert data["status"] == "error"
        assert "Unknown action" in data["message"]

    def test_missing_fields_returns_422(self, app_client: TestClient) -> None:
        response = app_client.post("/webhook", json={"secret": "test-secret"})
        assert response.status_code == 422

    @patch("trade_analysis.live.routes.handle_scan")
    def test_no_signal(
        self, mock_scan: MagicMock, app_client: TestClient
    ) -> None:
        mock_scan.return_value = None

        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "AAPL",
            "timeframe": "Daily",
        })
        data = response.json()
        assert data["status"] == "no_signal"

    @patch("trade_analysis.live.routes.persist_signal")
    @patch("trade_analysis.live.routes.handle_scan")
    def test_tradeable_signal_persisted(
        self,
        mock_scan: MagicMock,
        mock_persist: MagicMock,
        app_client: TestClient,
    ) -> None:
        mock_scan.return_value = {
            "symbol": "AAPL",
            "timeframe": "Daily",
            "signal_direction": "long",
            "signal_score": 4,
            "signal_tradeable": True,
            "regime": "bull",
            "entry_price": 195.0,
            "exit_stop": 190.0,
            "exit_target": 205.0,
            "exit_rr_ratio": 2.0,
            "bar_timestamp": "2024-06-15T00:00:00+00:00",
        }
        mock_persist.return_value = 42

        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "AAPL",
            "timeframe": "Daily",
        })
        data = response.json()
        assert data["status"] == "ok"
        assert data["persisted"] is True
        assert data["signal"]["direction"] == "long"
        assert data["signal"]["score"] == 4
        assert data["signal"]["tradeable"] is True

    @patch("trade_analysis.live.routes.persist_signal")
    @patch("trade_analysis.live.routes.handle_scan")
    def test_non_tradeable_signal_not_persisted_by_default(
        self,
        mock_scan: MagicMock,
        mock_persist: MagicMock,
        app_client: TestClient,
    ) -> None:
        mock_scan.return_value = {
            "symbol": "AAPL",
            "timeframe": "Daily",
            "signal_direction": "long",
            "signal_score": 2,
            "signal_tradeable": False,
            "regime": "bull",
            "bar_timestamp": "2024-06-15T00:00:00+00:00",
        }

        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "AAPL",
            "timeframe": "Daily",
        })
        data = response.json()
        assert data["status"] == "ok"
        assert data["persisted"] is False
        mock_persist.assert_not_called()

    @patch("trade_analysis.live.routes.handle_scan")
    def test_pipeline_error(
        self, mock_scan: MagicMock, app_client: TestClient
    ) -> None:
        mock_scan.side_effect = RuntimeError("data fetch failed")

        response = app_client.post("/webhook", json={
            "secret": "test-secret",
            "symbol": "AAPL",
            "timeframe": "Daily",
        })
        data = response.json()
        assert data["status"] == "error"
        assert "Pipeline error" in data["message"]

    def test_empty_secret_skips_auth(self, app_client: TestClient) -> None:
        """When webhook_secret is empty, all requests pass auth."""
        app_client.app.state.live_config = LiveConfig(webhook_secret="")

        with patch("trade_analysis.live.routes.handle_scan") as mock_scan:
            mock_scan.return_value = None
            response = app_client.post("/webhook", json={
                "secret": "anything",
                "symbol": "AAPL",
                "timeframe": "Daily",
            })
            data = response.json()
            assert data["status"] == "no_signal"

    def test_default_action_is_scan(self, app_client: TestClient) -> None:
        """action field defaults to 'scan' when not provided."""
        with patch("trade_analysis.live.routes.handle_scan") as mock_scan:
            mock_scan.return_value = None
            response = app_client.post("/webhook", json={
                "secret": "test-secret",
                "symbol": "AAPL",
                "timeframe": "Daily",
            })
            data = response.json()
            # If it reached handle_scan (not "Unknown action"), default worked
            assert data["status"] == "no_signal"
