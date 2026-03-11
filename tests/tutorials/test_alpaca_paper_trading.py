"""Tests for Alpaca Paper Trading Tutorial.

Validates that all tutorial sections run correctly with local data
(no Alpaca API key required).
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Add examples to path so we can import the tutorial
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples" / "tutorials"))

from alpaca_paper_trading import (
    FakeMarketDataProvider,
    make_sample_data,
    make_strategy,
    section_1_fill_price_modes,
    section_2_order_generator,
    section_3_engine_step,
    section_4_execution_report,
)

from oxq.core import Engine
from oxq.core.types import Fill, Order, Position
from oxq.portfolio import ExecutionReport
from oxq.trade import FillPriceMode, SimBroker, generate_orders


class TestSampleData:
    def test_make_sample_data_shape(self):
        """Sample data has correct shape and columns."""
        data = make_sample_data()
        assert "AAPL" in data
        df = data["AAPL"]
        assert len(df) == 120
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_make_sample_data_trend(self):
        """Data follows expected trend pattern."""
        df = make_sample_data()["AAPL"]
        # Starts high, dips, rises, dips
        assert df["close"].iloc[0] == 200
        assert df["close"].iloc[49] == 102
        assert df["close"].iloc[89] == 180
        assert df["close"].iloc[119] == 122


class TestSection1FillPriceModes:
    def test_all_modes_produce_trades(self):
        """All 4 fill modes should produce at least 1 trade."""
        fills_by_mode = section_1_fill_price_modes()
        assert len(fills_by_mode) == 4
        for label, fills in fills_by_mode.items():
            assert len(fills) > 0, f"{label} produced no trades"

    def test_different_modes_different_prices(self):
        """Different modes should produce different fill prices."""
        data = make_sample_data()
        market = FakeMarketDataProvider(data)
        strategy = make_strategy()

        prices = {}
        for mode in FillPriceMode:
            broker = SimBroker(fill_price_mode=mode)
            result = Engine().run(
                strategy, market=market, broker=broker,
                start="2024-01-01", end="2024-12-31",
            )
            if result.trades:
                prices[mode] = result.trades[0].filled_price

        # At least CLOSE and NEXT_OPEN should differ
        if FillPriceMode.CLOSE in prices and FillPriceMode.NEXT_OPEN in prices:
            assert prices[FillPriceMode.CLOSE] != prices[FillPriceMode.NEXT_OPEN]


class TestSection2OrderGenerator:
    def test_generates_orders(self):
        """OrderGenerator produces expected orders."""
        planned = section_2_order_generator()
        assert len(planned) == 3  # AAPL adjust + GOOG buy + MSFT buy

        by_sym = {p.order.symbol: p for p in planned}
        assert "AAPL" in by_sym
        assert "GOOG" in by_sym
        assert "MSFT" in by_sym

    def test_order_context_fields(self):
        """PlannedOrder has correct context for human review."""
        positions = {"AAPL": Position(symbol="AAPL", shares=30, avg_cost=Decimal("150"))}
        target_weights = {"AAPL": Decimal("0.4"), "GOOG": Decimal("0.3")}
        prices = {"AAPL": Decimal("180"), "GOOG": Decimal("140")}
        total_capital = Decimal("100000")

        planned = generate_orders(
            target_weights=target_weights,
            positions=positions,
            prices=prices,
            total_capital=total_capital,
        )

        # AAPL: target = floor(100000 * 0.4 / 180) = 222, current = 30 → BUY 192
        aapl = next(p for p in planned if p.order.symbol == "AAPL")
        assert aapl.order.side == "BUY"
        assert aapl.current_shares == 30
        assert aapl.target_weight == Decimal("0.4")
        assert aapl.estimated_amount == aapl.order.shares * Decimal("180")

    def test_lot_size_100(self):
        """A-shares lot size (100) rounds correctly."""
        planned = generate_orders(
            target_weights={"000001": Decimal("0.5")},
            positions={},
            prices={"000001": Decimal("15")},
            total_capital=Decimal("100000"),
            lot_size=100,
        )
        assert len(planned) == 1
        # floor(100000 * 0.5 / 15 / 100) * 100 = floor(333.33/100)*100 = 300
        assert planned[0].order.shares == 3300
        assert planned[0].order.shares % 100 == 0


class TestSection3EngineStep:
    def test_step_matches_run(self):
        """setup() + step() produces identical results to run()."""
        section_3_engine_step()  # Includes its own assertion

    def test_step_incremental_equity(self):
        """Equity curve grows incrementally with each step()."""
        data = make_sample_data()
        market = FakeMarketDataProvider(data)
        strategy = make_strategy()

        engine = Engine()
        engine.setup(
            strategy=strategy, market=market, broker=SimBroker(),
            start="2024-01-01", end="2024-12-31",
        )

        for i, date in enumerate(engine.dates):
            engine.step(date)
            assert len(engine.result.equity_curve) == i + 1


class TestSection4ExecutionReport:
    def test_report_runs(self):
        """ExecutionReport section runs without errors."""
        section_4_execution_report()

    def test_report_with_simulated_fills(self):
        """ExecutionReport correctly computes slippage."""
        sim = [Fill(
            order=Order(symbol="AAPL", side="BUY", shares=100),
            filled_price=Decimal("150.00"),
            filled_at="2024-01-02",
        )]
        live = [Fill(
            order=Order(symbol="AAPL", side="BUY", shares=100),
            filled_price=Decimal("150.45"),
            filled_at="2024-01-02",
        )]

        report = ExecutionReport(sim_fills=sim, live_fills=live)
        assert len(report.comparisons) == 1
        c = report.comparisons[0]
        assert c.price_slippage == Decimal("0.45") / Decimal("150.00")
        assert c.shares_diff == 0

        s = report.summary()
        assert s["matched_trades"] == 1
        assert s["sim_only_trades"] == 0
        assert s["live_only_trades"] == 0

    def test_close_vs_next_open_slippage(self):
        """CLOSE vs NEXT_OPEN produces non-zero slippage."""
        data = make_sample_data()
        market = FakeMarketDataProvider(data)
        strategy = make_strategy()

        sim_result = Engine().run(
            strategy, market=market,
            broker=SimBroker(fill_price_mode=FillPriceMode.CLOSE),
            start="2024-01-01", end="2024-12-31",
        )
        live_result = Engine().run(
            strategy, market=market,
            broker=SimBroker(fill_price_mode=FillPriceMode.NEXT_OPEN),
            start="2024-01-01", end="2024-12-31",
        )

        report = ExecutionReport(
            sim_fills=sim_result.trades,
            live_fills=live_result.trades,
        )
        s = report.summary()
        assert s["matched_trades"] > 0
        # Slippage should be non-zero since fill prices differ
        assert s["avg_price_slippage"] != Decimal("0")
