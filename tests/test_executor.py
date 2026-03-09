"""Tests for OrderExecutor — proposal lifecycle, approval, rejection, expiry."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.execution.executor import OrderExecutor
from trade_analysis.execution.risk_manager import RiskManager
from trade_analysis.execution.schwab_client import SchwabClient
from trade_analysis.live.config import ExecutionConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_sb() -> SupabaseClient:
    sb = MagicMock(spec=SupabaseClient)
    sb.enabled = True
    return sb


def _config() -> ExecutionConfig:
    return ExecutionConfig(
        dry_run=True,
        max_positions=5,
        max_per_symbol=1,
        proposal_expiry_hours=24,
        default_order_type="market",
    )


@pytest.fixture()
def sample_signal() -> dict:
    return {
        "symbol": "AAPL",
        "asset_class": "stock",
        "timeframe": "Daily",
        "signal_direction": "long",
        "entry_price": 185.0,
        "exit_stop": 180.0,
        "exit_target": 195.0,
        "exit_rr_ratio": 2.0,
        "signal_score": 4,
        "regime": "bull",
        "config_hash": "abc123",
    }


def _make_executor(sb=None):
    sb = sb or _mock_sb()
    schwab = SchwabClient(dry_run=True)
    risk = RiskManager(_config(), sb)
    return OrderExecutor(schwab, risk, sb)


# ---------------------------------------------------------------------------
# Proposal creation
# ---------------------------------------------------------------------------


class TestCreateProposal:
    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals", return_value=[])
    @patch("trade_analysis.execution.executor.create_proposal", return_value=42)
    def test_creates_proposal(
        self, mock_create, mock_pp, mock_pos, mock_ks, sample_signal
    ):
        executor = _make_executor()
        result = executor.create_proposal(sample_signal)

        assert result is not None
        assert result["id"] == 42
        assert result["symbol"] == "AAPL"
        assert result["direction"] == "long"
        assert result["status"] == "pending_approval"

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=True)
    def test_blocked_by_kill_switch(self, mock_ks, sample_signal):
        executor = _make_executor()
        result = executor.create_proposal(sample_signal)
        assert result is None

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions")
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals", return_value=[])
    def test_blocked_by_position_limit(
        self, mock_pp, mock_pos, mock_ks, sample_signal
    ):
        mock_pos.return_value = [{"symbol": f"SYM{i}"} for i in range(5)]
        executor = _make_executor()
        result = executor.create_proposal(sample_signal)
        assert result is None

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals", return_value=[])
    @patch("trade_analysis.execution.executor.create_proposal", return_value=None)
    def test_proposal_without_supabase(
        self, mock_create, mock_pp, mock_pos, mock_ks, sample_signal
    ):
        """When Supabase is disabled, proposal is returned without id."""
        executor = _make_executor()
        result = executor.create_proposal(sample_signal)
        assert result is not None
        assert "id" not in result  # no id when persistence returns None


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


class TestApproveProposal:
    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    @patch("trade_analysis.execution.executor.load_proposals")
    @patch("trade_analysis.execution.executor.update_proposal_status")
    @patch("trade_analysis.execution.executor.create_order", return_value=10)
    @patch("trade_analysis.execution.executor.create_fill", return_value=5)
    @patch("trade_analysis.execution.executor.upsert_position", return_value=3)
    def test_approve_and_fill(
        self,
        mock_upsert,
        mock_fill,
        mock_order,
        mock_update,
        mock_load,
        mock_pos,
        mock_ks,
    ):
        mock_load.return_value = [
            {"id": 1, "symbol": "AAPL", "direction": "long",
             "entry_price": 185.0, "stop_loss": 180.0, "target_price": 195.0,
             "status": "pending_approval"}
        ]

        executor = _make_executor()
        result = executor.approve_proposal(1, qty=10)

        assert result["status"] == "ok"
        assert result["dry_run"] is True
        assert result["order_id"] == 10
        mock_update.assert_called_once()
        mock_order.assert_called_once()
        mock_fill.assert_called_once()
        mock_upsert.assert_called_once()

    @patch("trade_analysis.execution.executor.load_proposals", return_value=[])
    @patch("trade_analysis.execution.executor.load_pending_proposals", return_value=[])
    def test_proposal_not_found(self, mock_pending, mock_load):
        executor = _make_executor()
        result = executor.approve_proposal(999, qty=10)
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=True)
    @patch("trade_analysis.execution.executor.load_proposals")
    def test_blocked_by_kill_switch_at_execution(self, mock_load, mock_ks):
        mock_load.return_value = [
            {"id": 1, "symbol": "AAPL", "direction": "long",
             "status": "pending_approval"}
        ]

        executor = _make_executor()
        result = executor.approve_proposal(1, qty=10)
        assert result["status"] == "error"
        assert "Kill switch" in result["message"]


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


class TestRejectProposal:
    @patch("trade_analysis.execution.executor.update_proposal_status", return_value=True)
    def test_reject(self, mock_update):
        executor = _make_executor()
        result = executor.reject_proposal(1, reason="not confident")
        assert result is True
        mock_update.assert_called_once()

    @patch("trade_analysis.execution.executor.update_proposal_status", return_value=False)
    def test_reject_not_found(self, mock_update):
        executor = _make_executor()
        result = executor.reject_proposal(999)
        assert result is False


# ---------------------------------------------------------------------------
# Proposal expiry
# ---------------------------------------------------------------------------


class TestExpireStaleProposals:
    @patch("trade_analysis.execution.executor.load_pending_proposals")
    @patch("trade_analysis.execution.executor.update_proposal_status")
    def test_expires_old_proposals(self, mock_update, mock_load):
        old_time = (datetime.now(tz=timezone.utc) - timedelta(hours=25)).isoformat()
        mock_load.return_value = [
            {"id": 1, "created_at": old_time},
            {"id": 2, "created_at": old_time},
        ]
        mock_update.return_value = True

        executor = _make_executor()
        count = executor.expire_stale_proposals(max_age_hours=24)
        assert count == 2

    @patch("trade_analysis.execution.executor.load_pending_proposals")
    @patch("trade_analysis.execution.executor.update_proposal_status")
    def test_does_not_expire_fresh_proposals(self, mock_update, mock_load):
        fresh_time = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
        mock_load.return_value = [
            {"id": 1, "created_at": fresh_time},
        ]

        executor = _make_executor()
        count = executor.expire_stale_proposals(max_age_hours=24)
        assert count == 0
        mock_update.assert_not_called()

    @patch("trade_analysis.execution.executor.load_pending_proposals", return_value=[])
    def test_no_pending_proposals(self, mock_load):
        executor = _make_executor()
        count = executor.expire_stale_proposals()
        assert count == 0

    @patch("trade_analysis.execution.executor.load_pending_proposals")
    def test_handles_invalid_timestamp(self, mock_load):
        mock_load.return_value = [
            {"id": 1, "created_at": "invalid-date"},
        ]

        executor = _make_executor()
        count = executor.expire_stale_proposals()
        assert count == 0
