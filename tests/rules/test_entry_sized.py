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
