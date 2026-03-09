"""Schwab API client wrapper with dry-run simulation.

In dry-run mode (default), all order/position/balance methods return
simulated responses and log what *would* have been executed.
When credentials are provided and dry_run=False, delegates to schwab-py.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SchwabClient:
    """Thin wrapper around schwab-py with dry-run simulation.

    Parameters
    ----------
    api_key : str | None
        Schwab API app key. Falls back to SCHWAB_API_KEY env var.
    api_secret : str | None
        Schwab API app secret. Falls back to SCHWAB_API_SECRET env var.
    redirect_uri : str | None
        OAuth redirect URI. Falls back to SCHWAB_REDIRECT_URI env var.
    token_path : str | Path
        Path to persist OAuth tokens (access + refresh).
    dry_run : bool
        When True, simulate all operations without touching the broker.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        redirect_uri: str | None = None,
        token_path: str | Path = "data/schwab_token.json",
        dry_run: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("SCHWAB_API_KEY", "")
        self._api_secret = api_secret or os.environ.get("SCHWAB_API_SECRET", "")
        self._redirect_uri = redirect_uri or os.environ.get(
            "SCHWAB_REDIRECT_URI", ""
        )
        self._token_path = Path(token_path)
        self._dry_run = dry_run
        self._client: Any = None
        self._account_hash: str | None = None

        # Track simulated state for dry-run mode
        self._dry_run_order_counter = 0
        self._dry_run_positions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def connected(self) -> bool:
        """Whether a live Schwab connection is available."""
        return self._client is not None

    @property
    def has_credentials(self) -> bool:
        """Whether API credentials are configured."""
        return bool(self._api_key and self._api_secret and self._redirect_uri)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish a Schwab API connection.

        Returns True if connected, False if credentials are missing or
        the connection failed.
        """
        if self._dry_run:
            logger.info("SchwabClient in dry-run mode — skipping real connection.")
            return True

        if not self.has_credentials:
            logger.warning(
                "Schwab credentials not configured. "
                "Set SCHWAB_API_KEY, SCHWAB_API_SECRET, SCHWAB_REDIRECT_URI."
            )
            return False

        try:
            import schwab

            if self._token_path.exists():
                self._client = schwab.auth.client_from_token_file(
                    str(self._token_path),
                    api_key=self._api_key,
                    app_secret=self._api_secret,
                )
            else:
                logger.warning(
                    "Token file %s not found. Run initial OAuth flow first.",
                    self._token_path,
                )
                return False

            # Get account hash for order placement
            accounts = self._client.get_account_numbers()
            if accounts.status_code == 200:
                acct_data = accounts.json()
                if acct_data:
                    self._account_hash = acct_data[0].get("hashValue")
                    logger.info("Connected to Schwab account.")
                    return True

            logger.warning("Failed to retrieve Schwab account numbers.")
            return False

        except Exception:
            logger.exception("Failed to connect to Schwab API.")
            return False

    # ------------------------------------------------------------------
    # Order Placement
    # ------------------------------------------------------------------

    def place_equity_order(
        self,
        symbol: str,
        direction: str,
        qty: int,
        order_type: str = "market",
        limit_price: float | None = None,
        entry_price: float | None = None,
    ) -> dict[str, Any]:
        """Place an equity order (or simulate in dry-run mode).

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. "AAPL").
        direction : str
            "long" (buy) or "short" (sell short).
        qty : int
            Number of shares.
        order_type : str
            "market" or "limit".
        limit_price : float | None
            Required when order_type is "limit".
        entry_price : float | None
            Expected entry price (used for dry-run simulation fill).

        Returns
        -------
        dict with keys: order_id, status, fill_price, fill_qty, dry_run
        """
        if self._dry_run:
            return self._simulate_order(
                symbol, direction, qty, order_type, limit_price, entry_price
            )

        return self._place_real_order(
            symbol, direction, qty, order_type, limit_price
        )

    def _simulate_order(
        self,
        symbol: str,
        direction: str,
        qty: int,
        order_type: str,
        limit_price: float | None,
        entry_price: float | None,
    ) -> dict[str, Any]:
        """Simulate an order placement in dry-run mode."""
        self._dry_run_order_counter += 1
        order_id = f"DRY-{self._dry_run_order_counter:06d}"

        fill_price = limit_price or entry_price or 0.0

        logger.info(
            "[DRY RUN] %s %d %s @ %s (order_type=%s, id=%s)",
            "BUY" if direction == "long" else "SELL SHORT",
            qty,
            symbol,
            f"${fill_price:.2f}" if fill_price else "MARKET",
            order_type,
            order_id,
        )

        return {
            "order_id": order_id,
            "status": "filled",
            "fill_price": fill_price,
            "fill_qty": qty,
            "commission": 0.0,
            "dry_run": True,
        }

    def _place_real_order(
        self,
        symbol: str,
        direction: str,
        qty: int,
        order_type: str,
        limit_price: float | None,
    ) -> dict[str, Any]:
        """Place a real order via schwab-py."""
        if not self._client or not self._account_hash:
            return {
                "order_id": None,
                "status": "rejected",
                "fill_price": None,
                "fill_qty": 0,
                "commission": 0.0,
                "dry_run": False,
                "error": "Not connected to Schwab API.",
            }

        try:
            from schwab.orders.equities import equity_buy_market, equity_sell_short_market
            from schwab.orders.equities import equity_buy_limit, equity_sell_short_limit

            if direction == "long":
                if order_type == "limit" and limit_price is not None:
                    order = equity_buy_limit(symbol, qty, limit_price)
                else:
                    order = equity_buy_market(symbol, qty)
            else:
                if order_type == "limit" and limit_price is not None:
                    order = equity_sell_short_limit(symbol, qty, limit_price)
                else:
                    order = equity_sell_short_market(symbol, qty)

            response = self._client.place_order(self._account_hash, order)

            if response.status_code in (200, 201):
                # Extract order ID from Location header
                order_id = response.headers.get("Location", "").split("/")[-1]
                logger.info(
                    "Placed %s order for %d %s (schwab_id=%s)",
                    direction,
                    qty,
                    symbol,
                    order_id,
                )
                return {
                    "order_id": order_id,
                    "status": "placed",
                    "fill_price": None,
                    "fill_qty": 0,
                    "commission": 0.0,
                    "dry_run": False,
                }
            else:
                logger.warning(
                    "Schwab order rejected: %s %s", response.status_code, response.text
                )
                return {
                    "order_id": None,
                    "status": "rejected",
                    "fill_price": None,
                    "fill_qty": 0,
                    "commission": 0.0,
                    "dry_run": False,
                    "error": f"HTTP {response.status_code}",
                }

        except Exception as exc:
            logger.exception("Error placing Schwab order.")
            return {
                "order_id": None,
                "status": "rejected",
                "fill_price": None,
                "fill_qty": 0,
                "commission": 0.0,
                "dry_run": False,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Order Management
    # ------------------------------------------------------------------

    def cancel_order(self, schwab_order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled."""
        if self._dry_run:
            logger.info("[DRY RUN] Cancel order %s", schwab_order_id)
            return True

        if not self._client or not self._account_hash:
            return False

        try:
            response = self._client.cancel_order(schwab_order_id, self._account_hash)
            return response.status_code in (200, 204)
        except Exception:
            logger.exception("Error cancelling order %s", schwab_order_id)
            return False

    def get_order_status(self, schwab_order_id: str) -> dict[str, Any]:
        """Get the current status of an order."""
        if self._dry_run:
            return {
                "order_id": schwab_order_id,
                "status": "filled",
                "dry_run": True,
            }

        if not self._client or not self._account_hash:
            return {"order_id": schwab_order_id, "status": "unknown", "dry_run": False}

        try:
            response = self._client.get_order(schwab_order_id, self._account_hash)
            if response.status_code == 200:
                data = response.json()
                return {
                    "order_id": schwab_order_id,
                    "status": data.get("status", "unknown").lower(),
                    "filled_qty": data.get("filledQuantity", 0),
                    "dry_run": False,
                }
            return {"order_id": schwab_order_id, "status": "unknown", "dry_run": False}
        except Exception:
            logger.exception("Error getting order status for %s", schwab_order_id)
            return {"order_id": schwab_order_id, "status": "unknown", "dry_run": False}

    # ------------------------------------------------------------------
    # Account / Positions
    # ------------------------------------------------------------------

    def get_account_positions(self) -> list[dict[str, Any]]:
        """Get open positions from the Schwab account."""
        if self._dry_run:
            return list(self._dry_run_positions)

        if not self._client or not self._account_hash:
            return []

        try:
            response = self._client.get_account(
                self._account_hash, fields=["positions"]
            )
            if response.status_code == 200:
                data = response.json()
                positions = data.get("securitiesAccount", {}).get("positions", [])
                return [
                    {
                        "symbol": p.get("instrument", {}).get("symbol", ""),
                        "qty": int(p.get("longQuantity", 0) or p.get("shortQuantity", 0)),
                        "direction": "long" if p.get("longQuantity", 0) > 0 else "short",
                        "avg_price": p.get("averagePrice", 0.0),
                        "market_value": p.get("marketValue", 0.0),
                        "current_price": p.get("currentDayProfitLossPercentage", 0.0),
                    }
                    for p in positions
                ]
            return []
        except Exception:
            logger.exception("Error fetching Schwab positions.")
            return []

    def get_account_balance(self) -> dict[str, Any]:
        """Get account balance summary."""
        if self._dry_run:
            return {
                "total_value": 100_000.0,
                "cash_available": 100_000.0,
                "buying_power": 200_000.0,
                "dry_run": True,
            }

        if not self._client or not self._account_hash:
            return {
                "total_value": 0.0,
                "cash_available": 0.0,
                "buying_power": 0.0,
                "dry_run": False,
            }

        try:
            response = self._client.get_account(self._account_hash)
            if response.status_code == 200:
                data = response.json()
                balances = data.get("securitiesAccount", {}).get(
                    "currentBalances", {}
                )
                return {
                    "total_value": balances.get("liquidationValue", 0.0),
                    "cash_available": balances.get("cashBalance", 0.0),
                    "buying_power": balances.get("buyingPower", 0.0),
                    "dry_run": False,
                }
            return {"total_value": 0.0, "cash_available": 0.0, "buying_power": 0.0, "dry_run": False}
        except Exception:
            logger.exception("Error fetching Schwab balance.")
            return {"total_value": 0.0, "cash_available": 0.0, "buying_power": 0.0, "dry_run": False}

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return client health status."""
        return {
            "connected": self.connected,
            "dry_run": self._dry_run,
            "has_credentials": self.has_credentials,
            "account_hash": self._account_hash[:8] + "..." if self._account_hash else None,
        }
