"""Risk manager — position limits, kill switch, pre-trade checks."""

from __future__ import annotations

import logging
from typing import Any

from trade_analysis.analyzer.persistence import SupabaseClient
from trade_analysis.execution.persistence import (
    get_kill_switch_status,
    load_open_positions,
    load_pending_proposals,
    set_kill_switch,
)
from trade_analysis.live.config import ExecutionConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """Pre-trade risk checks: kill switch, position limits, duplicate detection.

    Parameters
    ----------
    config : ExecutionConfig
        Execution configuration with limits.
    sb : SupabaseClient
        Database connection (may be disabled).
    """

    def __init__(self, config: ExecutionConfig, sb: SupabaseClient) -> None:
        self._config = config
        self._sb = sb

    @property
    def config(self) -> ExecutionConfig:
        return self._config

    # ------------------------------------------------------------------
    # Pre-proposal checks
    # ------------------------------------------------------------------

    def check_can_propose(
        self,
        symbol: str,
        direction: str,
    ) -> tuple[bool, str]:
        """Check whether a new proposal can be created.

        Returns (allowed, reason).
        """
        # Kill switch
        if get_kill_switch_status(self._sb):
            return False, "Kill switch is engaged."

        # Position limits
        open_positions = load_open_positions(self._sb)

        if len(open_positions) >= self._config.max_positions:
            return False, (
                f"At max positions ({self._config.max_positions}). "
                "Close a position before adding new proposals."
            )

        # Per-symbol limit
        symbol_positions = [
            p for p in open_positions if p.get("symbol") == symbol
        ]
        if len(symbol_positions) >= self._config.max_per_symbol:
            return False, (
                f"Already have {len(symbol_positions)} position(s) for {symbol} "
                f"(max {self._config.max_per_symbol})."
            )

        # Duplicate pending proposal
        pending = load_pending_proposals(self._sb, symbol=symbol)
        duplicates = [
            p for p in pending
            if p.get("direction") == direction
        ]
        if duplicates:
            return False, (
                f"Duplicate pending proposal for {symbol} {direction} "
                f"(proposal id={duplicates[0].get('id')})."
            )

        return True, "OK"

    # ------------------------------------------------------------------
    # Pre-execution checks
    # ------------------------------------------------------------------

    def check_can_execute(
        self,
        proposal: dict[str, Any],
    ) -> tuple[bool, str]:
        """Re-check risk limits at execution time (approval moment).

        Returns (allowed, reason).
        """
        # Kill switch (re-check)
        if get_kill_switch_status(self._sb):
            return False, "Kill switch is engaged."

        # Position limits (re-check — may have changed since proposal creation)
        open_positions = load_open_positions(self._sb)

        if len(open_positions) >= self._config.max_positions:
            return False, f"At max positions ({self._config.max_positions})."

        symbol = proposal.get("symbol", "")
        symbol_positions = [
            p for p in open_positions if p.get("symbol") == symbol
        ]
        if len(symbol_positions) >= self._config.max_per_symbol:
            return False, (
                f"Already at max positions for {symbol} "
                f"({self._config.max_per_symbol})."
            )

        return True, "OK"

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def engage_kill_switch(self, reason: str = "manual") -> None:
        """Engage the kill switch. Blocks all new proposals and executions."""
        set_kill_switch(self._sb, enabled=True, reason=reason, toggled_by="user")
        logger.warning("Kill switch ENGAGED: %s", reason)

    def disengage_kill_switch(self) -> None:
        """Disengage the kill switch. Allows proposals and executions again."""
        set_kill_switch(self._sb, enabled=False, reason="disengaged", toggled_by="user")
        logger.info("Kill switch disengaged.")

    def is_kill_switch_engaged(self) -> bool:
        """Check if the kill switch is currently engaged."""
        return get_kill_switch_status(self._sb)
