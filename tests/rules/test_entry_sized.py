"""Tests for SizedEntryRule."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, Position
from oxq.rules.entry import SizedEntryRule


def test_sized_entry_basic() -> None:
    rule = SizedEntryRule(signal="cross", shares=100)
    row = pd.Series({"close": 150.0, "cross": True})
    portfolio = Portfolio(cash=Decimal("100000"))
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 100


def test_sized_entry_max_position() -> None:
    rule = SizedEntryRule(signal="cross", shares=100, max_position=500)
    row = pd.Series({"close": 150.0, "cross": True})
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=450, avg_cost=Decimal("140"))},
    )
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is None  # already holding, SizedEntryRule skips


def test_sized_entry_max_pct_equity() -> None:
    rule = SizedEntryRule(signal="cross", shares=500, max_pct_equity=0.2)
    row = pd.Series({"close": 100.0, "cross": True})
    portfolio = Portfolio(cash=Decimal("100000"))
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 200  # 20% of 100000 / 100


def test_sized_entry_no_signal() -> None:
    rule = SizedEntryRule(signal="cross", shares=100)
    row = pd.Series({"close": 150.0, "cross": False})
    portfolio = Portfolio(cash=Decimal("100000"))
    assert rule.evaluate("AAPL", row, portfolio) is None


def test_sized_entry_already_holding() -> None:
    rule = SizedEntryRule(signal="cross", shares=100)
    row = pd.Series({"close": 150.0, "cross": True})
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("140"))},
    )
    assert rule.evaluate("AAPL", row, portfolio) is None


def test_sized_entry_pct_equity_uses_bar_prices() -> None:
    """With bar_prices set, total_value includes other positions correctly."""
    rule = SizedEntryRule(signal="cross", shares=500, max_pct_equity=0.2)
    row = pd.Series({"close": 100.0, "cross": True})
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "MSFT": Position(symbol="MSFT", shares=200, avg_cost=Decimal("250")),
        },
        bar_prices={
            "AAPL": Decimal("100"),
            "MSFT": Decimal("300"),  # MSFT worth 200*300=60000
        },
    )
    # total = 50000 cash + 60000 MSFT = 110000
    # 20% of 110000 = 22000 → 22000/100 = 220 shares max
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 220


def test_sized_entry_pct_equity_without_bar_prices() -> None:
    """Without bar_prices, MSFT is valued at 0 → undersized order."""
    rule = SizedEntryRule(signal="cross", shares=500, max_pct_equity=0.2)
    row = pd.Series({"close": 100.0, "cross": True})
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "MSFT": Position(symbol="MSFT", shares=200, avg_cost=Decimal("250")),
        },
        # bar_prices not set → empty dict
    )
    # Without bar_prices: total = 50000 + 0 = 50000
    # 20% of 50000 = 10000 → 100 shares (undersized)
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 100  # much less than the correct 220
