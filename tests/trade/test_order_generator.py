"""Tests for OrderGenerator — generates trade plans from target weights."""

from __future__ import annotations

from decimal import Decimal

from oxq.core.types import Position
from oxq.trade.order_generator import generate_orders


class TestGenerateOrders:
    def test_basic_buy(self):
        """Target weight > 0, no existing position -> BUY."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 1
        assert result[0].order.symbol == "AAPL"
        assert result[0].order.side == "BUY"
        assert result[0].order.shares == 50
        assert result[0].target_shares == 50
        assert result[0].current_shares == 0
        assert result[0].target_weight == Decimal("0.5")
        assert result[0].estimated_amount == Decimal("5000")

    def test_basic_sell(self):
        """Target weight < current weight -> SELL delta."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.3")},
            positions={"AAPL": Position(symbol="AAPL", shares=50, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 1
        assert result[0].order.side == "SELL"
        assert result[0].order.shares == 20  # 50 - 30

    def test_no_change(self):
        """Target matches current -> no order."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={"AAPL": Position(symbol="AAPL", shares=50, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 0

    def test_clear_position(self):
        """Symbol in positions but not in target_weights -> full SELL."""
        result = generate_orders(
            target_weights={"GOOG": Decimal("0.5")},
            positions={"AAPL": Position(symbol="AAPL", shares=50, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100"), "GOOG": Decimal("200")},
            total_capital=Decimal("10000"),
        )
        sells = [r for r in result if r.order.side == "SELL"]
        assert len(sells) == 1
        assert sells[0].order.symbol == "AAPL"
        assert sells[0].order.shares == 50

    def test_lot_size(self):
        """lot_size=100 rounds down to nearest lot."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
            lot_size=100,
        )
        assert len(result) == 0  # 50 shares < 100 lot_size -> rounds to 0

    def test_lot_size_rounds_down(self):
        """lot_size=100, enough capital for 1 lot."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("1.0")},
            positions={},
            prices={"AAPL": Decimal("50")},
            total_capital=Decimal("10000"),
            lot_size=100,
        )
        assert result[0].order.shares == 200  # floor(200 / 100) * 100

    def test_multiple_symbols(self):
        """Multiple symbols with different weights."""
        result = generate_orders(
            target_weights={
                "AAPL": Decimal("0.4"),
                "GOOG": Decimal("0.3"),
            },
            positions={},
            prices={"AAPL": Decimal("100"), "GOOG": Decimal("150")},
            total_capital=Decimal("10000"),
        )
        assert len(result) == 2
        by_sym = {r.order.symbol: r for r in result}
        assert by_sym["AAPL"].order.shares == 40
        assert by_sym["GOOG"].order.shares == 20

    def test_market_order_type(self):
        """All generated orders use market order type."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.5")},
            positions={},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert result[0].order.order_type == "market"

    def test_planned_order_context(self):
        """PlannedOrder includes current_weight for review."""
        result = generate_orders(
            target_weights={"AAPL": Decimal("0.6")},
            positions={"AAPL": Position(symbol="AAPL", shares=30, avg_cost=Decimal("100"))},
            prices={"AAPL": Decimal("100")},
            total_capital=Decimal("10000"),
        )
        assert result[0].current_weight == Decimal("0.3")
        assert result[0].current_shares == 30
