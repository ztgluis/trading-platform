"""Tests for RiskManager — position limits, kill switch, pre-trade checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.execution.risk_manager import RiskManager
from trade_analysis.live.config import ExecutionConfig


def _mock_sb() -> SupabaseClient:
    sb = MagicMock(spec=SupabaseClient)
    sb.enabled = True
    return sb


def _config(**overrides) -> ExecutionConfig:
    defaults = {
        "dry_run": True,
        "max_positions": 5,
        "max_per_symbol": 1,
        "proposal_expiry_hours": 24,
        "default_order_type": "market",
    }
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


# ---------------------------------------------------------------------------
# check_can_propose
# ---------------------------------------------------------------------------


class TestCheckCanPropose:
    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals", return_value=[])
    def test_allowed_when_no_limits_hit(self, mock_pp, mock_pos, mock_ks):
        rm = RiskManager(_config(), _mock_sb())
        allowed, reason = rm.check_can_propose("AAPL", "long")
        assert allowed is True
        assert reason == "OK"

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=True)
    def test_blocked_by_kill_switch(self, mock_ks):
        rm = RiskManager(_config(), _mock_sb())
        allowed, reason = rm.check_can_propose("AAPL", "long")
        assert allowed is False
        assert "Kill switch" in reason

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions")
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals", return_value=[])
    def test_blocked_by_max_positions(self, mock_pp, mock_pos, mock_ks):
        mock_pos.return_value = [
            {"symbol": f"SYM{i}"} for i in range(5)
        ]
        rm = RiskManager(_config(max_positions=5), _mock_sb())
        allowed, reason = rm.check_can_propose("AAPL", "long")
        assert allowed is False
        assert "max positions" in reason.lower()

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions")
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals", return_value=[])
    def test_blocked_by_per_symbol_limit(self, mock_pp, mock_pos, mock_ks):
        mock_pos.return_value = [{"symbol": "AAPL"}]
        rm = RiskManager(_config(max_per_symbol=1), _mock_sb())
        allowed, reason = rm.check_can_propose("AAPL", "long")
        assert allowed is False
        assert "AAPL" in reason

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals")
    def test_blocked_by_duplicate_proposal(self, mock_pp, mock_pos, mock_ks):
        mock_pp.return_value = [{"id": 99, "direction": "long"}]
        rm = RiskManager(_config(), _mock_sb())
        allowed, reason = rm.check_can_propose("AAPL", "long")
        assert allowed is False
        assert "Duplicate" in reason

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    @patch("trade_analysis.execution.risk_manager.load_pending_proposals")
    def test_different_direction_not_duplicate(self, mock_pp, mock_pos, mock_ks):
        mock_pp.return_value = [{"id": 99, "direction": "short"}]
        rm = RiskManager(_config(), _mock_sb())
        allowed, reason = rm.check_can_propose("AAPL", "long")
        assert allowed is True


# ---------------------------------------------------------------------------
# check_can_execute
# ---------------------------------------------------------------------------


class TestCheckCanExecute:
    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    @patch("trade_analysis.execution.risk_manager.load_open_positions", return_value=[])
    def test_allowed(self, mock_pos, mock_ks):
        rm = RiskManager(_config(), _mock_sb())
        allowed, reason = rm.check_can_execute({"symbol": "AAPL"})
        assert allowed is True

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=True)
    def test_blocked_by_kill_switch(self, mock_ks):
        rm = RiskManager(_config(), _mock_sb())
        allowed, reason = rm.check_can_execute({"symbol": "AAPL"})
        assert allowed is False


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    @patch("trade_analysis.execution.risk_manager.set_kill_switch")
    def test_engage(self, mock_set):
        rm = RiskManager(_config(), _mock_sb())
        rm.engage_kill_switch("emergency stop")
        mock_set.assert_called_once()
        args = mock_set.call_args
        assert args[1]["enabled"] is True or args[0][1] is True

    @patch("trade_analysis.execution.risk_manager.set_kill_switch")
    def test_disengage(self, mock_set):
        rm = RiskManager(_config(), _mock_sb())
        rm.disengage_kill_switch()
        mock_set.assert_called_once()

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=True)
    def test_is_engaged(self, mock_get):
        rm = RiskManager(_config(), _mock_sb())
        assert rm.is_kill_switch_engaged() is True

    @patch("trade_analysis.execution.risk_manager.get_kill_switch_status", return_value=False)
    def test_is_not_engaged(self, mock_get):
        rm = RiskManager(_config(), _mock_sb())
        assert rm.is_kill_switch_engaged() is False
