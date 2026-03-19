"""Tests for pre-trade constraint rules."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, Position, RuleResult
from oxq.rules.constraint import BlacklistRule, MaxHoldingsRule, RebalanceFrequencyRule


# ---------------------------------------------------------------------------
# BlacklistRule
# ---------------------------------------------------------------------------


def test_blacklist_blocks_listed_symbol() -> None:
    rule = BlacklistRule(symbols={"AAPL", "TSLA"})
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("AAPL", row, portfolio)
    assert isinstance(result, RuleResult)
    assert result.weights == {"AAPL": 0.0}
    assert result.reason != ""


def test_blacklist_allows_unlisted_symbol() -> None:
    rule = BlacklistRule(symbols={"AAPL", "TSLA"})
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("GOOG", row, portfolio)
    assert result.weights is None


def test_blacklist_empty() -> None:
    rule = BlacklistRule(symbols=set())
    portfolio = Portfolio(cash=Decimal("100000"))
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("AAPL", row, portfolio)
    assert result.weights is None


# ---------------------------------------------------------------------------
# MaxHoldingsRule
# ---------------------------------------------------------------------------


def test_max_holdings_blocks_when_at_limit() -> None:
    rule = MaxHoldingsRule(max_holdings=2)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100")),
            "GOOG": Position(symbol="GOOG", shares=50, avg_cost=Decimal("200")),
        },
    )
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("MSFT", row, portfolio)
    assert result.weights == {"MSFT": 0.0}
    assert result.reason != ""


def test_max_holdings_allows_existing_position() -> None:
    rule = MaxHoldingsRule(max_holdings=2)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100")),
            "GOOG": Position(symbol="GOOG", shares=50, avg_cost=Decimal("200")),
        },
    )
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("AAPL", row, portfolio)
    assert result.weights is None


def test_max_holdings_allows_below_limit() -> None:
    rule = MaxHoldingsRule(max_holdings=3)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100")),
        },
    )
    row = pd.Series({"close": 150.0})
    result = rule.evaluate("MSFT", row, portfolio)
    assert result.weights is None


# ---------------------------------------------------------------------------
# RebalanceFrequencyRule
# ---------------------------------------------------------------------------


def test_rebalance_frequency_holds_within_interval() -> None:
    rule = RebalanceFrequencyRule(interval_days=5)
    portfolio = Portfolio(cash=Decimal("100000"))

    row1 = pd.Series({"close": 100.0}, name=pd.Timestamp("2024-01-02"))
    result1 = rule.evaluate("AAPL", row1, portfolio)
    assert result1.hold is False

    row2 = pd.Series({"close": 101.0}, name=pd.Timestamp("2024-01-03"))
    result2 = rule.evaluate("AAPL", row2, portfolio)
    assert result2.hold is True
    assert result2.reason != ""


def test_rebalance_frequency_allows_after_interval() -> None:
    rule = RebalanceFrequencyRule(interval_days=5)
    portfolio = Portfolio(cash=Decimal("100000"))

    row1 = pd.Series({"close": 100.0}, name=pd.Timestamp("2024-01-02"))
    rule.evaluate("AAPL", row1, portfolio)

    row2 = pd.Series({"close": 105.0}, name=pd.Timestamp("2024-01-08"))
    result = rule.evaluate("AAPL", row2, portfolio)
    assert result.hold is False
