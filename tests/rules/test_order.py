"""Tests for Order Rules (stop-loss, take-profit, trailing stop)."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, Position
from oxq.rules.order import StopLossRule, TakeProfitRule, TrailingStopRule


def test_stop_loss_generates_stop_order() -> None:
    rule = StopLossRule(threshold=0.05)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 145.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.order_type == "stop"
    assert order.side == "SELL"
    assert order.shares == 100
    assert order.stop_price == Decimal("150") * (1 - Decimal("0.05"))


def test_stop_loss_no_position() -> None:
    rule = StopLossRule(threshold=0.05)
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 145.0})
    assert rule.evaluate("AAPL", row, portfolio) is None


def test_take_profit_generates_limit_order() -> None:
    rule = TakeProfitRule(threshold=0.15)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 170.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.order_type == "limit"
    assert order.side == "SELL"
    assert order.limit_price == Decimal("150") * (1 + Decimal("0.15"))


def test_take_profit_no_position() -> None:
    rule = TakeProfitRule(threshold=0.15)
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 170.0})
    assert rule.evaluate("AAPL", row, portfolio) is None


def test_trailing_stop_generates_trailing_order() -> None:
    rule = TrailingStopRule(trail_pct=0.05)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 155.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.order_type == "trailing_stop"
    assert order.trail_pct == 0.05
    assert order.shares == 100


def test_trailing_stop_no_position() -> None:
    rule = TrailingStopRule(trail_pct=0.05)
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 155.0})
    assert rule.evaluate("AAPL", row, portfolio) is None


def test_stop_loss_price_calculation() -> None:
    rule = StopLossRule(threshold=0.10)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("200"))},
    )
    row = pd.Series({"close": 195.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.stop_price == Decimal("200") * (1 - Decimal("0.10"))
    assert order.stop_price == Decimal("180")


def test_take_profit_price_calculation() -> None:
    rule = TakeProfitRule(threshold=0.20)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("200"))},
    )
    row = pd.Series({"close": 220.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.limit_price == Decimal("200") * (1 + Decimal("0.20"))
    assert order.limit_price == Decimal("240")


def test_stop_loss_uses_full_position_shares() -> None:
    rule = StopLossRule(threshold=0.05)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=250, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 145.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 250


def test_trailing_stop_uses_full_position_shares() -> None:
    rule = TrailingStopRule(trail_pct=0.05)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=300, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 155.0})
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 300
