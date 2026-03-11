"""Tests for LiveBroker — Broker Protocol implementation over Alpaca."""

from __future__ import annotations

import queue
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oxq.core.types import Fill, Order
from oxq.trade.live_broker import LiveBroker


@pytest.fixture
def mock_client():
    """Return a LiveBroker with mocked AlpacaClient."""
    with patch("oxq.trade.live_broker.AlpacaClient") as MockClient:
        instance = MockClient.return_value
        instance.submit_order.return_value = {"id": "alpaca-001", "status": "accepted"}
        instance.start_trade_stream.return_value = None
        instance.stop_trade_stream.return_value = None
        broker = LiveBroker(api_key="k", secret_key="s")
        yield broker, instance


class TestSubmitOrder:
    def test_market_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        oid = broker.submit_order(order)
        assert oid == "alpaca-001"
        call_args = client.submit_order.call_args[0][0]
        assert call_args["symbol"] == "AAPL"
        assert call_args["side"] == "buy"
        assert call_args["qty"] == "100"
        assert call_args["type"] == "market"
        assert call_args["time_in_force"] == "day"

    def test_limit_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="GOOG", side="SELL", shares=50, order_type="limit", limit_price=Decimal("150.50"))
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "limit"
        assert call_args["limit_price"] == "150.50"

    def test_stop_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="TSLA", side="SELL", shares=10, order_type="stop", stop_price=Decimal("200.00"))
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "stop"
        assert call_args["stop_price"] == "200.00"

    def test_stop_limit_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="MSFT", side="BUY", shares=25, order_type="stop_limit", stop_price=Decimal("300.00"), limit_price=Decimal("305.00"))
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "stop_limit"
        assert call_args["stop_price"] == "300.00"
        assert call_args["limit_price"] == "305.00"

    def test_trailing_stop_order_mapping(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AMZN", side="SELL", shares=15, order_type="trailing_stop", trail_pct=0.05)
        broker.submit_order(order)
        call_args = client.submit_order.call_args[0][0]
        assert call_args["type"] == "trailing_stop"
        assert call_args["trail_percent"] == "5.0"

    def test_order_registered_in_orderbook(self, mock_client):
        broker, client = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        oid = broker.submit_order(order)
        open_orders = broker.get_open_orders("AAPL")
        assert len(open_orders) == 1
        assert open_orders[0].id == oid


class TestGetFills:
    def test_empty_fills(self, mock_client):
        broker, _ = mock_client
        assert broker.get_fills() == []

    def test_fill_from_websocket_callback(self, mock_client):
        broker, _ = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        broker.submit_order(order)
        broker._on_fill_event({
            "event": "fill",
            "order": {
                "id": "alpaca-001",
                "symbol": "AAPL",
                "side": "buy",
                "qty": "100",
                "type": "market",
                "filled_avg_price": "150.25",
                "filled_at": "2026-03-11T10:00:00Z",
            },
        })
        fills = broker.get_fills()
        assert len(fills) == 1
        assert fills[0].filled_price == Decimal("150.25")
        assert fills[0].order.symbol == "AAPL"
        assert fills[0].filled_at == "2026-03-11T10:00:00Z"

    def test_get_fills_clears_queue(self, mock_client):
        broker, _ = mock_client
        order = Order(symbol="AAPL", side="BUY", shares=100)
        broker.submit_order(order)
        broker._on_fill_event({
            "event": "fill",
            "order": {
                "id": "alpaca-001",
                "filled_avg_price": "150.25",
                "filled_at": "2026-03-11T10:00:00Z",
            },
        })
        broker.get_fills()
        assert broker.get_fills() == []
