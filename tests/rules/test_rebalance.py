"""Tests for RebalanceRule."""

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, Position, Rule
from oxq.rules.rebalance import RebalanceRule, _portfolio_value


def _make_row(date: pd.Timestamp, target_weight: float, close: float) -> pd.Series:
    s = pd.Series({"close": close, "target_weight": target_weight})
    s.name = date
    return s


def test_rebalance_rule_satisfies_rule_protocol() -> None:
    assert isinstance(RebalanceRule(weight_col="target_weight"), Rule)


def test_rebalance_rule_buys_to_target() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=1)
    portfolio = Portfolio(cash=Decimal("100000"))
    row = _make_row(pd.Timestamp("2024-01-01"), 0.5, 100.0)
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.side == "BUY"
    assert order.shares == 500  # 100000 * 0.5 / 100


def test_rebalance_rule_sells_overweight() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=1)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=800, avg_cost=Decimal("100"))},
    )
    row = _make_row(pd.Timestamp("2024-01-01"), 0.3, 100.0)
    # total_value = 50000 + 800*100 = 130000
    # target_shares = int(130000 * 0.3 / 100) = 390
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.side == "SELL"
    assert order.shares == 410


def test_rebalance_rule_zero_weight_sells_all() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=1)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=200, avg_cost=Decimal("100"))},
    )
    row = _make_row(pd.Timestamp("2024-01-01"), 0.0, 100.0)
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.side == "SELL"
    assert order.shares == 200


def test_rebalance_rule_no_change_needed() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=1)
    portfolio = Portfolio(
        cash=Decimal("10000"),
        positions={"AAPL": Position(symbol="AAPL", shares=500, avg_cost=Decimal("100"))},
    )
    # total=60000, weight=500*100/60000=0.8333, target_shares=int(60000*0.8333/100)=500
    row = _make_row(pd.Timestamp("2024-01-01"), 500 * 100 / 60_000, 100.0)
    assert rule.evaluate("AAPL", row, portfolio) is None


def test_rebalance_rule_frequency_gating() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=3)
    portfolio = Portfolio(cash=Decimal("100000"))
    dates = pd.bdate_range("2024-01-01", periods=6)
    orders = []
    for date in dates:
        row = _make_row(date, 0.5, 100.0)
        orders.append(rule.evaluate("AAPL", row, portfolio))
    # frequency=3: bar 3 and bar 6 trigger (bar_count % 3 == 0)
    assert orders[0] is None
    assert orders[1] is None
    assert orders[2] is not None  # bar 3
    assert orders[3] is None
    assert orders[4] is None
    assert orders[5] is not None  # bar 6


def test_rebalance_rule_multi_symbol_same_date() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=2)
    portfolio = Portfolio(cash=Decimal("100000"))
    d1, d2 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    # Date 1 (bar 1): not rebalance bar
    assert rule.evaluate("A", _make_row(d1, 0.5, 100.0), portfolio) is None
    assert rule.evaluate("B", _make_row(d1, 0.3, 200.0), portfolio) is None
    # Date 2 (bar 2): rebalance bar
    assert rule.evaluate("A", _make_row(d2, 0.5, 100.0), portfolio) is not None
    assert rule.evaluate("B", _make_row(d2, 0.3, 200.0), portfolio) is not None


def test_rebalance_rule_nan_weight_as_zero() -> None:
    rule = RebalanceRule(weight_col="target_weight", frequency=1)
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100"))},
    )
    row = _make_row(pd.Timestamp("2024-01-01"), float("nan"), 100.0)
    order = rule.evaluate("AAPL", row, portfolio)
    assert order is not None
    assert order.side == "SELL"
    assert order.shares == 100


def test_portfolio_value_with_bar_prices() -> None:
    portfolio = Portfolio(
        cash=Decimal("10000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("90")),
            "MSFT": Position(symbol="MSFT", shares=50, avg_cost=Decimal("200")),
        },
        bar_prices={
            "AAPL": Decimal("110"),
            "MSFT": Decimal("300"),  # real-time price, not avg_cost
        },
    )
    # AAPL: 100*110, MSFT: 50*300=15000, cash=10000 → total=36000
    assert _portfolio_value(portfolio, "AAPL", Decimal("110")) == Decimal("36000")


def test_portfolio_value_without_bar_prices() -> None:
    """Without bar_prices, only current symbol gets a price; others → 0."""
    portfolio = Portfolio(
        cash=Decimal("10000"),
        positions={
            "AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("90")),
            "MSFT": Position(symbol="MSFT", shares=50, avg_cost=Decimal("200")),
        },
    )
    # Only AAPL gets a price: 100*110=11000 + cash=10000 + MSFT=0 → 21000
    assert _portfolio_value(portfolio, "AAPL", Decimal("110")) == Decimal("21000")


def test_rebalance_rule_has_name() -> None:
    assert RebalanceRule(weight_col="tw").name == "RebalanceRule"
