"""Tests for Risk Rules."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import Order, Portfolio, Position
from oxq.rules.risk import DailyLossLimitRisk, MaxDrawdownRisk


def test_max_drawdown_no_trigger() -> None:
    rule = MaxDrawdownRisk(max_drawdown=0.15)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 150.0})
    order, hold = rule.evaluate("AAPL", row, portfolio)
    assert order is None
    assert hold is False


def test_max_drawdown_triggers() -> None:
    rule = MaxDrawdownRisk(max_drawdown=0.15)
    portfolio = Portfolio(
        cash=Decimal("0"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    # First call establishes peak at 100*200 = 20000
    row_high = pd.Series({"close": 200.0})
    rule.evaluate("AAPL", row_high, portfolio)

    # Now price drops: value = 100*160 = 16000, dd = 4000/20000 = 20% > 15%
    row_low = pd.Series({"close": 160.0})
    order, hold = rule.evaluate("AAPL", row_low, portfolio)
    assert hold is True
    assert order is not None
    assert order.side == "SELL"
    assert order.shares == 100


def test_max_drawdown_no_position_still_holds() -> None:
    rule = MaxDrawdownRisk(max_drawdown=0.15)
    portfolio = Portfolio(cash=Decimal("20000"))
    # Peak = 20000
    row = pd.Series({"close": 100.0})
    rule.evaluate("AAPL", row, portfolio)

    # Cash dropped to 16000 (simulating loss), dd = 20%
    portfolio.cash = Decimal("16000")
    order, hold = rule.evaluate("AAPL", row, portfolio)
    assert hold is True
    assert order is None  # No position to sell


def test_daily_loss_limit_no_trigger() -> None:
    rule = DailyLossLimitRisk(max_daily_loss=0.03)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    row = pd.Series({"close": 150.0}, name=pd.Timestamp("2024-01-02"))
    order, hold = rule.evaluate("AAPL", row, portfolio)
    assert order is None
    assert hold is False


def test_daily_loss_limit_triggers() -> None:
    rule = DailyLossLimitRisk(max_daily_loss=0.03)
    portfolio = Portfolio(
        cash=Decimal("0"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    # First eval on Day 1 establishes day_start_value = 100*200 = 20000
    row1 = pd.Series({"close": 200.0}, name=pd.Timestamp("2024-01-02"))
    rule.evaluate("AAPL", row1, portfolio)

    # Second eval same day, price dropped: value = 100*190 = 19000
    # daily loss = 1000/20000 = 5% > 3%
    row2 = pd.Series({"close": 190.0}, name=pd.Timestamp("2024-01-02"))
    order, hold = rule.evaluate("AAPL", row2, portfolio)
    assert hold is True
    assert order is None  # Only freeze, no liquidation
