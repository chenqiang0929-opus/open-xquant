"""Tests for position sizing functions."""

from decimal import Decimal

from oxq.core.types import Portfolio, Position
from oxq.rules.sizing import clip_to_max_position, clip_to_pct_equity


def test_clip_max_position_no_holding() -> None:
    portfolio = Portfolio(cash=Decimal("100000"))
    assert clip_to_max_position(100, "AAPL", portfolio, max_shares=500) == 100


def test_clip_max_position_partial() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=450, avg_cost=Decimal("150"))},
    )
    assert clip_to_max_position(100, "AAPL", portfolio, max_shares=500) == 50


def test_clip_max_position_at_limit() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=500, avg_cost=Decimal("150"))},
    )
    assert clip_to_max_position(100, "AAPL", portfolio, max_shares=500) == 0


def test_clip_pct_equity_no_holding() -> None:
    portfolio = Portfolio(cash=Decimal("100000"))
    prices = {"AAPL": Decimal("100")}
    # max 20% of 100000 = 20000, at $100 = 200 shares
    result = clip_to_pct_equity(300, "AAPL", Decimal("100"), portfolio, prices, max_pct=0.2)
    assert result == 200


def test_clip_pct_equity_partial_holding() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("100"))},
    )
    prices = {"AAPL": Decimal("100")}
    # total = 50000 + 100*100 = 60000, max 20% = 12000, current = 10000, room = 2000/100 = 20
    result = clip_to_pct_equity(100, "AAPL", Decimal("100"), portfolio, prices, max_pct=0.2)
    assert result == 20
