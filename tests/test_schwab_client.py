"""Tests for Schwab client wrapper with dry-run simulation."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from trade_analysis.execution.schwab_client import SchwabClient


# ---------------------------------------------------------------------------
# Dry-run mode tests
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_dry_run_is_default(self):
        client = SchwabClient()
        assert client.dry_run is True
        assert client.connected is False

    def test_connect_in_dry_run(self):
        client = SchwabClient(dry_run=True)
        result = client.connect()
        assert result is True  # dry-run always "connects"

    def test_place_market_order(self):
        client = SchwabClient(dry_run=True)
        result = client.place_equity_order(
            symbol="AAPL",
            direction="long",
            qty=10,
            order_type="market",
            entry_price=185.0,
        )
        assert result["dry_run"] is True
        assert result["status"] == "filled"
        assert result["fill_price"] == 185.0
        assert result["fill_qty"] == 10
        assert result["order_id"].startswith("DRY-")

    def test_place_limit_order(self):
        client = SchwabClient(dry_run=True)
        result = client.place_equity_order(
            symbol="MSFT",
            direction="short",
            qty=5,
            order_type="limit",
            limit_price=400.0,
        )
        assert result["fill_price"] == 400.0  # limit price used
        assert result["fill_qty"] == 5

    def test_order_counter_increments(self):
        client = SchwabClient(dry_run=True)
        r1 = client.place_equity_order("AAPL", "long", 1, entry_price=100.0)
        r2 = client.place_equity_order("MSFT", "long", 1, entry_price=200.0)
        assert r1["order_id"] == "DRY-000001"
        assert r2["order_id"] == "DRY-000002"

    def test_cancel_order_dry_run(self):
        client = SchwabClient(dry_run=True)
        assert client.cancel_order("DRY-000001") is True

    def test_get_order_status_dry_run(self):
        client = SchwabClient(dry_run=True)
        result = client.get_order_status("DRY-000001")
        assert result["status"] == "filled"
        assert result["dry_run"] is True

    def test_get_positions_dry_run(self):
        client = SchwabClient(dry_run=True)
        positions = client.get_account_positions()
        assert positions == []

    def test_get_balance_dry_run(self):
        client = SchwabClient(dry_run=True)
        balance = client.get_account_balance()
        assert balance["total_value"] == 100_000.0
        assert balance["cash_available"] == 100_000.0
        assert balance["buying_power"] == 200_000.0
        assert balance["dry_run"] is True

    def test_health_check_dry_run(self):
        client = SchwabClient(dry_run=True)
        health = client.health_check()
        assert health["dry_run"] is True
        assert health["connected"] is False
        assert health["account_hash"] is None


# ---------------------------------------------------------------------------
# Credential detection
# ---------------------------------------------------------------------------


class TestCredentials:
    @patch.dict(os.environ, {}, clear=True)
    def test_no_credentials_by_default(self):
        client = SchwabClient(dry_run=False)
        assert client.has_credentials is False

    def test_has_credentials_when_provided(self):
        client = SchwabClient(
            api_key="key",
            api_secret="secret",
            redirect_uri="https://localhost",
            dry_run=False,
        )
        assert client.has_credentials is True

    def test_connect_fails_without_credentials(self):
        client = SchwabClient(dry_run=False)
        result = client.connect()
        assert result is False

    @patch.dict("os.environ", {
        "SCHWAB_API_KEY": "env_key",
        "SCHWAB_API_SECRET": "env_secret",
        "SCHWAB_REDIRECT_URI": "https://localhost",
    })
    def test_reads_from_env_vars(self):
        client = SchwabClient(dry_run=False)
        assert client.has_credentials is True


# ---------------------------------------------------------------------------
# Real mode tests (mocked schwab-py)
# ---------------------------------------------------------------------------


class TestRealMode:
    def test_not_connected_returns_rejected(self):
        client = SchwabClient(dry_run=False)
        result = client.place_equity_order("AAPL", "long", 10)
        assert result["status"] == "rejected"
        assert result["dry_run"] is False
        assert "Not connected" in result.get("error", "")

    def test_cancel_order_not_connected(self):
        client = SchwabClient(dry_run=False)
        assert client.cancel_order("12345") is False

    def test_get_order_status_not_connected(self):
        client = SchwabClient(dry_run=False)
        result = client.get_order_status("12345")
        assert result["status"] == "unknown"

    def test_get_positions_not_connected(self):
        client = SchwabClient(dry_run=False)
        assert client.get_account_positions() == []

    def test_get_balance_not_connected(self):
        client = SchwabClient(dry_run=False)
        balance = client.get_account_balance()
        assert balance["total_value"] == 0.0

    def test_health_check_not_connected(self):
        client = SchwabClient(dry_run=False)
        health = client.health_check()
        assert health["dry_run"] is False
        assert health["connected"] is False


# ---------------------------------------------------------------------------
# Entry price fallback logic
# ---------------------------------------------------------------------------


class TestPriceFallback:
    def test_limit_price_takes_priority(self):
        client = SchwabClient(dry_run=True)
        result = client.place_equity_order(
            "AAPL", "long", 10,
            order_type="limit",
            limit_price=180.0,
            entry_price=185.0,
        )
        # limit_price should be used since it's first in the fallback chain
        assert result["fill_price"] == 180.0

    def test_entry_price_used_when_no_limit(self):
        client = SchwabClient(dry_run=True)
        result = client.place_equity_order(
            "AAPL", "long", 10,
            order_type="market",
            entry_price=185.0,
        )
        assert result["fill_price"] == 185.0

    def test_zero_when_no_prices(self):
        client = SchwabClient(dry_run=True)
        result = client.place_equity_order("AAPL", "long", 10)
        assert result["fill_price"] == 0.0
