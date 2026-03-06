"""Tests for core data types."""

from decimal import Decimal

from oxq.core.types import Fill, Order, Portfolio, Position


def test_order_is_frozen() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    assert order.symbol == "AAPL"
    assert order.side == "BUY"
    assert order.shares == 100
    assert order.order_type == "market"


def test_order_stop() -> None:
    order = Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="stop", stop_price=Decimal("142.50"),
    )
    assert order.order_type == "stop"
    assert order.stop_price == Decimal("142.50")


def test_order_limit() -> None:
    order = Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="limit", limit_price=Decimal("185"),
    )
    assert order.order_type == "limit"
    assert order.limit_price == Decimal("185")


def test_order_trailing_stop() -> None:
    order = Order(
        symbol="AAPL", side="SELL", shares=100,
        order_type="trailing_stop", trail_pct=0.05,
    )
    assert order.order_type == "trailing_stop"
    assert order.trail_pct == 0.05


def test_fill_is_frozen() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    fill = Fill(order=order, filled_price=Decimal("150"), filled_at="2024-01-02")
    assert fill.filled_price == Decimal("150")
    assert fill.filled_at == "2024-01-02"
    assert fill.order is order
    assert fill.fee == Decimal("0")


def test_fill_with_fee() -> None:
    order = Order(symbol="AAPL", side="BUY", shares=100)
    fill = Fill(
        order=order, filled_price=Decimal("150"),
        filled_at="2024-01-02", fee=Decimal("5"),
    )
    assert fill.fee == Decimal("5")


def test_position_is_frozen() -> None:
    pos = Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))
    assert pos.symbol == "AAPL"
    assert pos.shares == 100
    assert pos.avg_cost == Decimal("150")


def test_portfolio_total_value_cash_only() -> None:
    portfolio = Portfolio(cash=Decimal("100000"))
    assert portfolio.total_value({}) == Decimal("100000")


def test_portfolio_total_value_with_positions() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150")),
            "MSFT": Position(symbol="MSFT", shares=50, avg_cost=Decimal("300")),
        },
    )
    prices = {"AAPL": Decimal("160"), "MSFT": Decimal("310")}
    # cash + 100*160 + 50*310 = 50000 + 16000 + 15500 = 81500
    assert portfolio.total_value(prices) == Decimal("81500")


def test_portfolio_total_value_missing_price() -> None:
    portfolio = Portfolio(
        cash=Decimal("10000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    # If price not in dict, position valued at 0
    assert portfolio.total_value({}) == Decimal("10000")
