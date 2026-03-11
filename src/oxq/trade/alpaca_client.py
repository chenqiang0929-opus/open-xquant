"""Alpaca API client — REST + WebSocket communication layer."""

from __future__ import annotations

import os
from typing import Any

import httpx


class AlpacaAPIError(Exception):
    """Raised when an Alpaca API call fails."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Alpaca API error {status_code}: {detail}")


_PAPER_REST = "https://paper-api.alpaca.markets"
_LIVE_REST = "https://api.alpaca.markets"


class AlpacaClient:
    """Low-level Alpaca API client.

    Handles authentication, REST calls, and WebSocket streaming.
    All REST methods return raw dicts — business logic lives in LiveBroker.

    Parameters
    ----------
    api_key : str or None
        Alpaca API key. Falls back to ``ALPACA_API_KEY`` env var.
    secret_key : str or None
        Alpaca secret key. Falls back to ``ALPACA_SECRET_KEY`` env var.
    paper : bool
        If True (default), use Paper Trading endpoints.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not self._api_key or not self._secret_key:
            msg = "API key and secret key required. Set ALPACA_API_KEY/ALPACA_SECRET_KEY or pass explicitly."
            raise ValueError(msg)

        self._base_url = _PAPER_REST if paper else _LIVE_REST
        self._paper = paper
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
            },
            timeout=30.0,
        )

    # -- REST methods ----------------------------------------------------------

    def submit_order(self, order_params: dict[str, Any]) -> dict[str, Any]:
        """Submit an order via POST /v2/orders."""
        resp = self._http.post("/v2/orders", json=order_params)
        return self._handle(resp)

    def get_order(self, order_id: str) -> dict[str, Any]:
        """Get order status via GET /v2/orders/{id}."""
        resp = self._http.get(f"/v2/orders/{order_id}")
        return self._handle(resp)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order via DELETE /v2/orders/{id}."""
        resp = self._http.delete(f"/v2/orders/{order_id}")
        if resp.status_code == 204:
            return {}
        return self._handle(resp)

    def list_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """List open orders via GET /v2/orders."""
        params: dict[str, str] = {"status": "open"}
        if symbol:
            params["symbols"] = symbol
        resp = self._http.get("/v2/orders", params=params)
        return self._handle(resp)

    def get_positions(self) -> list[dict[str, Any]]:
        """List positions via GET /v2/positions."""
        resp = self._http.get("/v2/positions")
        return self._handle(resp)

    def get_account(self) -> dict[str, Any]:
        """Get account info via GET /v2/account."""
        resp = self._http.get("/v2/account")
        return self._handle(resp)

    # -- Helpers ---------------------------------------------------------------

    def _handle(self, resp: httpx.Response) -> Any:
        """Check response status and return JSON."""
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text)
            except Exception:
                detail = resp.text
            raise AlpacaAPIError(resp.status_code, detail)
        return resp.json()
