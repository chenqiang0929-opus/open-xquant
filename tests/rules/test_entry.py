"""Tests for EntryRule and TargetValueEntryRule."""

import pandas as pd

from oxq.core.types import Portfolio, Position, Rule
from oxq.rules.entry import EntryRule, FullPositionEntryRule, TargetValueEntryRule


def test_entry_rule_satisfies_rule_protocol() -> None:
    assert isinstance(EntryRule(signal="sig"), Rule)


def test_entry_rule_buys_on_signal() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=100)
    row = pd.Series({"close": 150.0, "sma_10_x_sma_50": True})
    portfolio = Portfolio(cash=100_000.0)

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.symbol == "AAPL"
    assert order.side == "BUY"
    assert order.shares == 100


def test_entry_rule_no_signal_no_order() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=100)
    row = pd.Series({"close": 150.0, "sma_10_x_sma_50": False})
    portfolio = Portfolio(cash=100_000.0)

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_entry_rule_no_buy_if_already_holding() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=100)
    row = pd.Series({"close": 150.0, "sma_10_x_sma_50": True})
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=140.0)},
    )

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_entry_rule_buys_different_symbol() -> None:
    rule = EntryRule(signal="sma_10_x_sma_50", shares=50)
    row = pd.Series({"close": 300.0, "sma_10_x_sma_50": True})
    # Already holding AAPL, but evaluating MSFT
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=140.0)},
    )

    order = rule.evaluate("MSFT", row, portfolio)
    assert order is not None
    assert order.symbol == "MSFT"
    assert order.shares == 50


# ---------------------------------------------------------------------------
# TargetValueEntryRule
# ---------------------------------------------------------------------------


def test_target_value_entry_rule_satisfies_protocol() -> None:
    assert isinstance(TargetValueEntryRule(signal="sig", target_value=50_000), Rule)


def test_target_value_entry_rule_no_position() -> None:
    rule = TargetValueEntryRule(signal="cross", target_value=50_000)
    row = pd.Series({"close": 200.0, "cross": True})
    portfolio = Portfolio(cash=100_000.0)

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.side == "BUY"
    assert order.shares == 250  # 50000 / 200


def test_target_value_entry_rule_partial_position() -> None:
    rule = TargetValueEntryRule(signal="cross", target_value=50_000)
    row = pd.Series({"close": 200.0, "cross": True})
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=180.0)},
    )

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 150  # 250 - 100


def test_target_value_entry_rule_already_at_target() -> None:
    rule = TargetValueEntryRule(signal="cross", target_value=50_000)
    row = pd.Series({"close": 200.0, "cross": True})
    portfolio = Portfolio(
        cash=50_000.0,
        positions={"AAPL": Position(symbol="AAPL", shares=250, avg_cost=180.0)},
    )

    assert rule.evaluate("AAPL", row, portfolio) is None


def test_target_value_entry_rule_no_signal() -> None:
    rule = TargetValueEntryRule(signal="cross", target_value=50_000)
    row = pd.Series({"close": 200.0, "cross": False})
    portfolio = Portfolio(cash=100_000.0)

    assert rule.evaluate("AAPL", row, portfolio) is None


# ---------------------------------------------------------------------------
# FullPositionEntryRule
# ---------------------------------------------------------------------------


def test_full_position_entry_rule_satisfies_protocol() -> None:
    assert isinstance(FullPositionEntryRule(signal="sig"), Rule)


def test_full_position_entry_rule_buys_all_cash() -> None:
    rule = FullPositionEntryRule(signal="cross")
    row = pd.Series({"close": 200.0, "cross": True})
    portfolio = Portfolio(cash=100_000.0)

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.side == "BUY"
    assert order.shares == 500  # 100000 / 200


def test_full_position_entry_rule_partial_cash() -> None:
    rule = FullPositionEntryRule(signal="cross")
    row = pd.Series({"close": 200.0, "cross": True})
    portfolio = Portfolio(cash=30_000.0)

    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.shares == 150  # 30000 / 200


def test_full_position_entry_rule_no_cash() -> None:
    rule = FullPositionEntryRule(signal="cross")
    row = pd.Series({"close": 200.0, "cross": True})
    portfolio = Portfolio(cash=50.0)

    assert rule.evaluate("AAPL", row, portfolio) is None  # 50/200 = 0


def test_full_position_entry_rule_no_signal() -> None:
    rule = FullPositionEntryRule(signal="cross")
    row = pd.Series({"close": 200.0, "cross": False})
    portfolio = Portfolio(cash=100_000.0)

    assert rule.evaluate("AAPL", row, portfolio) is None
