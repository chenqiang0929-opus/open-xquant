"""Tests for oxq.observe.export — RunResult directory export."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from oxq.core.types import BarSnapshot, Fill, Order, Portfolio, PositionSnapshot
from oxq.portfolio.analytics import RunResult


def _make_result() -> RunResult:
    """Minimal RunResult with all fields populated for export tests."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    mktdata = {
        "AAPL": pd.DataFrame(
            {"open": [149, 150, 151, 152, 153], "close": [150, 151, 152, 153, 154]},
            index=dates,
        ),
        "GOOG": pd.DataFrame(
            {"open": [99, 100, 101, 102, 103], "close": [100, 101, 102, 103, 104]},
            index=dates,
        ),
    }
    trades = [
        Fill(
            order=Order(symbol="AAPL", side="BUY", shares=100),
            filled_price=Decimal("150.00"),
            filled_at="2024-01-02",
        ),
        Fill(
            order=Order(
                symbol="GOOG",
                side="BUY",
                shares=50,
                order_type="limit",
                limit_price=Decimal("100.50"),
            ),
            filled_price=Decimal("100.50"),
            filled_at="2024-01-03",
            fee=Decimal("9.95"),
        ),
    ]
    equity = [(d, 100_000.0 + i * 500.0) for i, d in enumerate(dates)]
    snapshots = [
        BarSnapshot(
            date=dates[0],
            target_weights={"AAPL": 0.6, "GOOG": 0.4},
            adjusted_weights={"AAPL": 0.55, "GOOG": 0.35},
            positions={
                "AAPL": PositionSnapshot(shares=100, avg_cost=150.0),
                "GOOG": PositionSnapshot(shares=50, avg_cost=100.0),
            },
            cash=45000.0,
            total_value=100000.0,
        ),
        BarSnapshot(
            date=dates[1],
            target_weights={"AAPL": 0.5, "GOOG": 0.5},
            adjusted_weights={"AAPL": 0.5, "GOOG": 0.5},
            positions={
                "AAPL": PositionSnapshot(shares=100, avg_cost=150.0),
                "GOOG": PositionSnapshot(shares=50, avg_cost=100.0),
            },
            cash=44500.0,
            total_value=100500.0,
        ),
    ]
    return RunResult(
        portfolio=Portfolio(cash=Decimal("44500")),
        trades=trades,
        equity_curve=equity,
        mktdata=mktdata,
        snapshots=snapshots,
    )


class TestFlattenEquity:
    def test_basic(self) -> None:
        from oxq.observe.export import _flatten_equity

        result = _make_result()
        df = _flatten_equity(result.equity_curve)
        assert list(df.columns) == ["value"]
        assert len(df) == 5
        assert df.iloc[0]["value"] == 100_000.0
        assert df.iloc[-1]["value"] == 102_000.0

    def test_empty(self) -> None:
        from oxq.observe.export import _flatten_equity

        df = _flatten_equity([])
        assert list(df.columns) == ["value"]
        assert len(df) == 0


class TestFlattenTrades:
    def test_basic(self) -> None:
        from oxq.observe.export import _flatten_trades

        result = _make_result()
        df = _flatten_trades(result.trades)
        assert len(df) == 2
        assert list(df.columns) == [
            "filled_at", "symbol", "side", "shares", "order_type",
            "limit_price", "stop_price", "filled_price", "fee",
        ]
        # First trade: market order, no limit/stop
        row0 = df.iloc[0]
        assert row0["symbol"] == "AAPL"
        assert row0["side"] == "BUY"
        assert row0["shares"] == 100
        assert row0["filled_price"] == 150.0
        assert row0["fee"] == 0.0
        assert pd.isna(row0["limit_price"])
        assert pd.isna(row0["stop_price"])
        # Second trade: limit order
        row1 = df.iloc[1]
        assert row1["symbol"] == "GOOG"
        assert row1["order_type"] == "limit"
        assert row1["limit_price"] == 100.50
        assert row1["fee"] == 9.95

    def test_empty(self) -> None:
        from oxq.observe.export import _flatten_trades

        df = _flatten_trades([])
        assert len(df) == 0
        assert "symbol" in df.columns
