"""Tests for live runner API routes using FastAPI TestClient.

Uses mocked DataManager and SupabaseClient to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trade_analysis.config.loader import SymbolConfig
from trade_analysis.execution.executor import OrderExecutor
from trade_analysis.execution.risk_manager import RiskManager
from trade_analysis.execution.schwab_client import SchwabClient
from trade_analysis.live.config import ExecutionConfig, LiveConfig


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
    sb = MagicMock()
    sb.enabled = True

    exec_config = ExecutionConfig()
    schwab = SchwabClient(dry_run=True)
    risk = MagicMock(spec=RiskManager)
    risk.config = exec_config
    risk.check_can_propose.return_value = (True, "OK")
    risk.is_kill_switch_engaged.return_value = False

    executor = MagicMock(spec=OrderExecutor)
    executor.schwab = schwab
    executor.risk = risk
    executor.create_proposal.return_value = {
        "id": 1,
        "symbol": "AAPL",
        "direction": "long",
        "entry_price": 195.0,
        "stop_loss": 190.0,
        "target_price": 205.0,
        "rr_ratio": 2.0,
        "signal_score": 4,
        "status": "pending_approval",
    }

    app.state.live_config = LiveConfig(webhook_secret="test-secret")
    app.state.data_manager = MagicMock()
    app.state.supabase = sb
    app.state.signal_config = MagicMock()
    app.state.symbols = mock_symbols
    app.state.executor = executor

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealth:
    @patch("trade_analysis.live.routes.get_kill_switch_status", return_value=False)
    def test_health_returns_ok(self, mock_ks, app_client: TestClient) -> None:
        response = app_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["supabase_connected"] is True
        assert data["symbols_configured"] == 2
        assert data["execution_mode"] == "dry_run"
        assert data["kill_switch_engaged"] is False


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

    @patch("trade_analysis.live.routes.persist_signal")
    @patch("trade_analysis.live.routes.handle_scan")
    def test_tradeable_signal_creates_proposal(
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
        assert data["proposal"] is not None
        assert data["proposal"]["symbol"] == "AAPL"
        assert data["proposal"]["direction"] == "long"
        # Verify executor was called
        app_client.app.state.executor.create_proposal.assert_called_once()


# ---------------------------------------------------------------------------
# Proposal endpoint tests
# ---------------------------------------------------------------------------


class TestProposalEndpoints:
    @patch("trade_analysis.live.routes.load_pending_proposals")
    def test_list_proposals(self, mock_load, app_client: TestClient) -> None:
        mock_load.return_value = [{"id": 1, "symbol": "AAPL"}]
        response = app_client.get("/proposals")
        data = response.json()
        assert data["status"] == "ok"
        assert len(data["proposals"]) == 1

    def test_approve_proposal(self, app_client: TestClient) -> None:
        app_client.app.state.executor.approve_proposal.return_value = {
            "status": "ok",
            "order_id": 10,
            "dry_run": True,
        }
        response = app_client.post("/proposals/1/approve", json={
            "secret": "test-secret",
            "qty": 10,
        })
        data = response.json()
        assert data["status"] == "ok"

    def test_approve_invalid_secret(self, app_client: TestClient) -> None:
        response = app_client.post("/proposals/1/approve", json={
            "secret": "wrong",
            "qty": 10,
        })
        assert response.status_code == 401

    def test_reject_proposal(self, app_client: TestClient) -> None:
        app_client.app.state.executor.reject_proposal.return_value = True
        response = app_client.post("/proposals/1/reject", json={
            "secret": "test-secret",
            "reason": "not confident",
        })
        data = response.json()
        assert data["status"] == "ok"

    def test_reject_invalid_secret(self, app_client: TestClient) -> None:
        response = app_client.post("/proposals/1/reject", json={
            "secret": "wrong",
        })
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Position endpoint tests
# ---------------------------------------------------------------------------


class TestPositionEndpoints:
    @patch("trade_analysis.live.routes.load_open_positions")
    def test_list_positions(self, mock_load, app_client: TestClient) -> None:
        mock_load.return_value = [{"id": 1, "symbol": "AAPL", "direction": "long"}]
        response = app_client.get("/positions")
        data = response.json()
        assert data["status"] == "ok"
        assert len(data["positions"]) == 1


# ---------------------------------------------------------------------------
# Kill switch endpoint tests
# ---------------------------------------------------------------------------


class TestKillSwitchEndpoint:
    def test_engage_kill_switch(self, app_client: TestClient) -> None:
        response = app_client.post("/kill-switch", json={
            "secret": "test-secret",
            "enabled": True,
            "reason": "emergency",
        })
        data = response.json()
        assert data["status"] == "ok"
        assert data["kill_switch_engaged"] is True

    def test_disengage_kill_switch(self, app_client: TestClient) -> None:
        response = app_client.post("/kill-switch", json={
            "secret": "test-secret",
            "enabled": False,
        })
        data = response.json()
        assert data["status"] == "ok"
        assert data["kill_switch_engaged"] is False

    def test_kill_switch_invalid_secret(self, app_client: TestClient) -> None:
        response = app_client.post("/kill-switch", json={
            "secret": "wrong",
            "enabled": True,
        })
        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Execution health
# ---------------------------------------------------------------------------


class TestExecutionHealth:
    def test_execution_health(self, app_client: TestClient) -> None:
        response = app_client.get("/execution/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "schwab" in data
