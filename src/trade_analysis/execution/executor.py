"""Order executor — proposal creation, approval, order placement.

Orchestrates the full lifecycle:
  tradeable signal → proposal → user approval → order → fill → position
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.execution.persistence import (
    close_position,
    create_fill,
    create_order,
    create_proposal,
    load_pending_proposals,
    load_proposals,
    update_order_status,
    update_proposal_status,
    upsert_position,
)
from trade_analysis.execution.risk_manager import RiskManager
from trade_analysis.execution.schwab_client import SchwabClient

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Manages the proposal → order → fill → position lifecycle.

    Parameters
    ----------
    schwab : SchwabClient
        Broker client (may be in dry-run mode).
    risk : RiskManager
        Pre-trade risk checks.
    sb : SupabaseClient
        Database persistence.
    """

    def __init__(
        self,
        schwab: SchwabClient,
        risk: RiskManager,
        sb: SupabaseClient,
    ) -> None:
        self._schwab = schwab
        self._risk = risk
        self._sb = sb

    @property
    def schwab(self) -> SchwabClient:
        return self._schwab

    @property
    def risk(self) -> RiskManager:
        return self._risk

    # ------------------------------------------------------------------
    # Proposal creation (from tradeable signal)
    # ------------------------------------------------------------------

    def create_proposal(self, signal_row: dict[str, Any]) -> dict[str, Any] | None:
        """Create an order proposal from a tradeable signal.

        Runs risk checks before creating. Returns the proposal dict
        (with id) if created, None if blocked by risk limits.
        """
        symbol = signal_row.get("symbol", "")
        direction = signal_row.get("signal_direction", "")

        # Risk check
        allowed, reason = self._risk.check_can_propose(symbol, direction)
        if not allowed:
            logger.info(
                "Proposal blocked for %s %s: %s", symbol, direction, reason
            )
            return None

        proposal = {
            "symbol": symbol,
            "asset_class": signal_row.get("asset_class", ""),
            "timeframe": signal_row.get("timeframe", ""),
            "direction": direction,
            "entry_price": signal_row.get("entry_price", 0.0),
            "stop_loss": signal_row.get("exit_stop", 0.0),
            "target_price": signal_row.get("exit_target", 0.0),
            "rr_ratio": signal_row.get("exit_rr_ratio", 0.0),
            "signal_score": signal_row.get("signal_score", 0),
            "regime": signal_row.get("regime", ""),
            "config_hash": signal_row.get("config_hash", ""),
            "signal_id": signal_row.get("signal_id"),
            "suggested_qty": 0,  # user decides qty on approval
            "status": "pending_approval",
        }

        proposal_id = create_proposal(self._sb, proposal)
        if proposal_id is not None:
            proposal["id"] = proposal_id
            logger.info(
                "Created proposal %s %s @ %.2f (id=%s)",
                symbol,
                direction,
                proposal["entry_price"],
                proposal_id,
            )
            return proposal

        # Supabase disabled — return proposal without id for logging
        logger.info(
            "Proposal created (not persisted): %s %s @ %.2f",
            symbol,
            direction,
            proposal["entry_price"],
        )
        return proposal

    # ------------------------------------------------------------------
    # Approval / Rejection
    # ------------------------------------------------------------------

    def approve_proposal(
        self,
        proposal_id: int,
        qty: int,
        order_type: str = "market",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """Approve a proposal and place an order.

        Returns a dict with order details and fill info.
        """
        # Load proposal
        proposals = load_proposals(self._sb, limit=1, status_filter="pending_approval")
        proposal = next((p for p in proposals if p.get("id") == proposal_id), None)

        if proposal is None:
            # Fall back to loading all pending and searching
            all_pending = load_pending_proposals(self._sb)
            proposal = next(
                (p for p in all_pending if p.get("id") == proposal_id), None
            )

        if proposal is None:
            return {"status": "error", "message": f"Proposal {proposal_id} not found or not pending."}

        # Re-check risk at execution time
        allowed, reason = self._risk.check_can_execute(proposal)
        if not allowed:
            return {"status": "error", "message": f"Blocked by risk manager: {reason}"}

        # Update proposal status
        now = datetime.now(tz=timezone.utc)
        update_proposal_status(self._sb, proposal_id, "approved", decided_at=now)

        # Place order via Schwab client
        symbol = proposal.get("symbol", "")
        direction = proposal.get("direction", "")

        order_result = self._schwab.place_equity_order(
            symbol=symbol,
            direction=direction,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            entry_price=proposal.get("entry_price"),
        )

        # Persist order record
        order_dict = {
            "proposal_id": proposal_id,
            "schwab_order_id": order_result.get("order_id"),
            "symbol": symbol,
            "direction": direction,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": order_result.get("status", "placed"),
            "dry_run": order_result.get("dry_run", True),
        }
        order_id = create_order(self._sb, order_dict)

        # If filled immediately (dry-run always fills instantly), create fill + position
        if order_result.get("status") == "filled" and order_id is not None:
            fill_dict = {
                "order_id": order_id,
                "fill_price": order_result.get("fill_price", 0.0),
                "fill_qty": order_result.get("fill_qty", qty),
                "commission": order_result.get("commission", 0.0),
            }
            create_fill(self._sb, fill_dict)

            # Open position
            position_dict = {
                "symbol": symbol,
                "direction": direction,
                "qty": qty,
                "avg_entry_price": order_result.get("fill_price", 0.0),
                "current_stop": proposal.get("stop_loss"),
                "current_target": proposal.get("target_price"),
                "order_id": order_id,
            }
            upsert_position(self._sb, position_dict)

        return {
            "status": "ok",
            "order_id": order_id,
            "schwab_order_id": order_result.get("order_id"),
            "order_status": order_result.get("status"),
            "fill_price": order_result.get("fill_price"),
            "dry_run": order_result.get("dry_run", True),
        }

    def reject_proposal(
        self,
        proposal_id: int,
        reason: str = "manual",
    ) -> bool:
        """Reject a proposal. Returns True if updated."""
        now = datetime.now(tz=timezone.utc)
        updated = update_proposal_status(
            self._sb, proposal_id, "rejected", decided_at=now
        )
        if updated:
            logger.info("Rejected proposal %s: %s", proposal_id, reason)
        return updated

    # ------------------------------------------------------------------
    # Proposal expiry
    # ------------------------------------------------------------------

    def expire_stale_proposals(self, max_age_hours: int = 24) -> int:
        """Expire pending proposals older than max_age_hours.

        Returns the number of expired proposals.
        """
        pending = load_pending_proposals(self._sb)
        now = datetime.now(tz=timezone.utc)
        expired_count = 0

        for proposal in pending:
            created_str = proposal.get("created_at", "")
            if not created_str:
                continue

            try:
                # Handle both timezone-aware and naive timestamps
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            age_hours = (now - created).total_seconds() / 3600
            if age_hours > max_age_hours:
                pid = proposal.get("id")
                if pid is not None:
                    update_proposal_status(self._sb, pid, "expired", decided_at=now)
                    expired_count += 1
                    logger.info(
                        "Expired proposal %s (age=%.1f hours)", pid, age_hours
                    )

        return expired_count
