"""Tests for OrderBook."""

from decimal import Decimal

from oxq.core.types import Order
from oxq.portfolio.orderbook import ManagedOrder, OrderBook


def test_add_order_returns_managed_order() -> None:
    book = OrderBook()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    managed = book.add(order, created_at="2024-01-02")
    assert isinstance(managed, ManagedOrder)
    assert managed.order is order
    assert managed.status == "open"
    assert managed.id.startswith("ord_")


def test_get_open_orders_all() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(Order(symbol="MSFT", side="BUY", shares=50), "2024-01-02")
    assert len(book.get_open_orders()) == 2


def test_get_open_orders_by_symbol() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(Order(symbol="MSFT", side="BUY", shares=50), "2024-01-02")
    assert len(book.get_open_orders(symbol="AAPL")) == 1


def test_cancel_orders() -> None:
    book = OrderBook()
    book.add(Order(symbol="AAPL", side="BUY", shares=100), "2024-01-02")
    book.add(
        Order(
            symbol="AAPL",
            side="SELL",
            shares=50,
            order_type="stop",
            stop_price=Decimal("140"),
        ),
        "2024-01-02",
    )
    canceled = book.cancel_orders("AAPL")
    assert len(canceled) == 2
    assert all(m.status == "canceled" for m in canceled)
    assert len(book.get_open_orders()) == 0


def test_fill_order() -> None:
    book = OrderBook()
    order = Order(symbol="AAPL", side="BUY", shares=100)
    managed = book.add(order, "2024-01-02")
    fill = book.fill(managed, price=Decimal("150"), date="2024-01-02")
    assert fill.filled_price == Decimal("150")
    assert managed.status == "filled"
    assert len(book.get_open_orders()) == 0


def test_dedup_replaces_same_symbol_side_type() -> None:
    book = OrderBook()
    o1 = Order(
        symbol="AAPL",
        side="SELL",
        shares=100,
        order_type="stop",
        stop_price=Decimal("140"),
    )
    o2 = Order(
        symbol="AAPL",
        side="SELL",
        shares=100,
        order_type="stop",
        stop_price=Decimal("145"),
    )
    book.add(o1, "2024-01-02")
    book.add(o2, "2024-01-03")
    open_orders = book.get_open_orders(symbol="AAPL")
    assert len(open_orders) == 1
    assert open_orders[0].order.stop_price == Decimal("145")
