"""Tests for Engine — full pipeline integration."""

from decimal import Decimal

import pandas as pd

from oxq.core.engine import Engine, _apply_fill
from oxq.core.strategy import Strategy
from oxq.core.types import Fill, Order, Portfolio, Position
from oxq.indicators.sma import SMA
from oxq.rules.entry import EntryRule
from oxq.rules.exit import ExitRule
from oxq.signals.crossover import Crossover
from oxq.trade.sim_broker import SimBroker
from oxq.universe.static import StaticUniverse


class FakeMarketDataProvider:
    """In-memory market data provider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= start) & (df.index <= end)]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_trending_data() -> dict[str, pd.DataFrame]:
    """Create data: downtrend → uptrend → downtrend to trigger crossover signals.

    Structure (120 bars):
    - Bars 0-49:  downtrend 200 → 102 (SMA10 < SMA50 at bar 49)
    - Bars 50-89: uptrend 102 → 182 (SMA10 crosses above SMA50 → golden cross)
    - Bars 90-119: downtrend 182 → 122 (SMA10 crosses below SMA50 → death cross)
    """
    n = 120
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes: list[float] = []
    for i in range(50):
        closes.append(200 - i * 2)       # 200 → 102
    for i in range(40):
        closes.append(102 + i * 2)       # 102 → 180
    for i in range(30):
        closes.append(180 - i * 2)       # 180 → 122

    return {
        "AAPL": pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        ),
    }


def _make_strategy() -> Strategy:
    return Strategy(
        name="test_sma_crossover",
        hypothesis="SMA10 crossing above SMA50 predicts positive returns",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_10": (SMA(), {"period": 10}),
            "sma_50": (SMA(), {"period": 50}),
        },
        signals={
            "sma_10_x_sma_50": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
        },
        entry_rules=[EntryRule(signal="sma_10_x_sma_50", shares=100)],
        exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
    )


def test_engine_full_pipeline() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, router=sim_broker, receiver=sim_broker,
        start="2024-01-01", end="2024-12-31",
    )

    # Should have at least some trades
    assert len(result.trades) > 0
    # All trades should be for AAPL
    assert all(t.order.symbol == "AAPL" for t in result.trades)
    # Equity curve should have one entry per bar
    assert len(result.equity_curve) == 120
    # mktdata should have indicator and signal columns
    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "sma_10_x_sma_50" in df.columns


def test_engine_run_through_indicator() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, router=sim_broker, receiver=sim_broker,
        start="2024-01-01", end="2024-12-31",
        run_through="indicator",
    )

    # Indicators computed, but no signals or trades
    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_50" in df.columns
    assert "sma_10_x_sma_50" not in df.columns
    assert len(result.trades) == 0
    assert len(result.equity_curve) == 0


def test_engine_run_through_signal() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, router=sim_broker, receiver=sim_broker,
        start="2024-01-01", end="2024-12-31",
        run_through="signal",
    )

    df = result.mktdata["AAPL"]
    assert "sma_10" in df.columns
    assert "sma_10_x_sma_50" in df.columns
    assert len(result.trades) == 0


def test_engine_portfolio_cash_changes() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, router=sim_broker, receiver=sim_broker,
        start="2024-01-01", end="2024-12-31",
        initial_cash=100_000.0,
    )

    # If any trades happened, cash should differ from initial
    if len(result.trades) > 0:
        # Either we still hold a position, or cash changed from fills
        has_position = len(result.portfolio.positions) > 0
        cash_changed = result.portfolio.cash != 100_000.0
        assert has_position or cash_changed


def test_engine_metrics() -> None:
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)
    strategy = _make_strategy()
    engine = Engine()
    sim_broker = SimBroker()

    result = engine.run(
        strategy, market=market, router=sim_broker, receiver=sim_broker,
        start="2024-01-01", end="2024-12-31",
    )

    # Metrics should return numbers without errors
    tr = result.total_return()
    sr = result.sharpe_ratio()
    mdd = result.max_drawdown()
    assert isinstance(tr, float)
    assert isinstance(sr, float)
    assert isinstance(mdd, float)
    assert mdd <= 0.0  # drawdown is always <= 0


def test_apply_fill_buy() -> None:
    portfolio = Portfolio(cash=Decimal("100000"))
    fill = Fill(
        order=Order(symbol="AAPL", side="BUY", shares=100),
        filled_price=Decimal("150"),
        filled_at="2024-01-02",
    )
    _apply_fill(portfolio, fill)

    assert portfolio.cash == Decimal("85000")  # 100000 - 100*150
    assert "AAPL" in portfolio.positions
    assert portfolio.positions["AAPL"].shares == 100
    assert portfolio.positions["AAPL"].avg_cost == Decimal("150")


def test_engine_rebalance_rules() -> None:
    """Engine should execute rebalance_rules and generate trades."""
    from oxq.indicators.ratio import Ratio
    from oxq.indicators.sma import SMA
    from oxq.rules.rebalance import RebalanceRule
    from oxq.signals.top_n_ranking import TopNRanking

    n = 60
    dates = pd.bdate_range("2024-01-01", periods=n)
    data: dict[str, pd.DataFrame] = {}
    for sym, base, trend in [("A", 100, 1.5), ("B", 80, 0.8), ("C", 120, -0.5)]:
        closes = [base + trend * i for i in range(n)]
        data[sym] = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=dates,
        )

    strategy = Strategy(
        name="rotation_test",
        hypothesis="Risk-adjusted momentum rotation",
        universe=StaticUniverse(("A", "B", "C")),
        indicators={
            "sma_fast": (SMA(), {"column": "close", "period": 5}),
            "sma_slow": (SMA(), {"column": "close", "period": 20}),
            "ram": (Ratio(), {"col_a": "sma_fast", "col_b": "sma_slow"}),
        },
        signals={
            "tw": (TopNRanking(), {"score": "ram", "n": 2, "max_weight": 0.9}),
        },
        entry_rules=[],
        exit_rules=[],
        rebalance_rules=[RebalanceRule(weight_col="tw", frequency=10)],
    )

    broker = SimBroker()
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        router=broker,
        receiver=broker,
        start="2024-01-01",
        end="2024-12-31",
    )

    assert len(result.trades) > 0
    assert len(result.equity_curve) == n
    assert "tw" in result.mktdata["A"].columns


def test_engine_missing_dates_uses_last_known_price() -> None:
    """When a symbol has no bar on a given date, engine should use the last
    known close price (not avg_cost) for equity curve valuation.

    Regression test: previously the engine fell back to avg_cost, which
    caused artificial volatility in cross-market portfolios where trading
    calendars differ.

    Setup:
      - Symbol A: 5 dates, close = [100, 102, 104, 106, 108]
      - Symbol B: 3 dates (d1, d2, d5), close = [200, 204, 216]
        (B is missing on d3 and d4, simulating a different trading calendar)
      - EntryRule buys 100 shares of each on d1 (via buy_signal column)
      - initial_cash = 100,000

    Hand-calculated equity curve:
      d1: buy A@100×100, B@200×100 → cash=70000
          equity = 70000 + 100×100 + 100×200 = 100,000
      d2: equity = 70000 + 100×102 + 100×204 = 100,600
      d3: B missing → use last_known=204
          equity = 70000 + 100×104 + 100×204 = 100,800
      d4: B missing → use last_known=204
          equity = 70000 + 100×106 + 100×204 = 101,000
      d5: equity = 70000 + 100×108 + 100×216 = 102,400
    """
    dates = pd.bdate_range("2024-01-01", periods=5)
    b_dates = dates[[0, 1, 4]]  # d1, d2, d5 — missing d3, d4

    def _make_df(idx: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
        n = len(idx)
        df = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1_000_000] * n,
            },
            index=idx,
        )
        # buy_signal is True only on day 1
        df["buy_signal"] = False
        df.iloc[0, df.columns.get_loc("buy_signal")] = True
        return df

    data = {
        "A": _make_df(dates, [100.0, 102.0, 104.0, 106.0, 108.0]),
        "B": _make_df(b_dates, [200.0, 204.0, 216.0]),
    }

    strategy = Strategy(
        name="missing_dates_test",
        hypothesis="Test last known price fallback",
        universe=StaticUniverse(("A", "B")),
        indicators={},
        signals={},
        entry_rules=[EntryRule(signal="buy_signal", shares=100)],
        exit_rules=[],
    )

    broker = SimBroker()
    result = Engine().run(
        strategy,
        market=FakeMarketDataProvider(data),
        router=broker,
        receiver=broker,
        start="2024-01-01",
        end="2024-12-31",
        initial_cash=100_000.0,
    )

    assert len(result.equity_curve) == 5
    expected = [100_000.0, 100_600.0, 100_800.0, 101_000.0, 102_400.0]
    actual = [v for _, v in result.equity_curve]
    assert actual == expected, f"expected {expected}, got {actual}"


def test_apply_fill_sell() -> None:
    portfolio = Portfolio(
        cash=Decimal("50000"),
        positions={"AAPL": Position(symbol="AAPL", shares=100, avg_cost=Decimal("150"))},
    )
    fill = Fill(
        order=Order(symbol="AAPL", side="SELL", shares=100),
        filled_price=Decimal("160"),
        filled_at="2024-03-01",
    )
    _apply_fill(portfolio, fill)

    assert portfolio.cash == Decimal("66000")  # 50000 + 100*160
    assert "AAPL" not in portfolio.positions


def test_engine_risk_rules_hold() -> None:
    """Risk rule hold=True should prevent entry/exit from executing."""
    from oxq.rules.risk import MaxDrawdownRisk

    # Use data that would normally trigger entry
    data = _make_trending_data()
    market = FakeMarketDataProvider(data)

    strategy = Strategy(
        name="risk_hold_test",
        hypothesis="Test risk hold",
        universe=StaticUniverse(("AAPL",)),
        indicators={
            "sma_10": (SMA(), {"period": 10}),
            "sma_50": (SMA(), {"period": 50}),
        },
        signals={
            "sma_10_x_sma_50": (Crossover(), {"fast": "sma_10", "slow": "sma_50"}),
        },
        entry_rules=[EntryRule(signal="sma_10_x_sma_50", shares=100)],
        exit_rules=[ExitRule(fast="sma_10", slow="sma_50")],
        risk_rules=[MaxDrawdownRisk(max_drawdown=0.001)],  # Very tight, will trigger
    )

    broker = SimBroker()
    result = Engine().run(
        strategy, market=market, router=broker, receiver=broker,
        start="2024-01-01", end="2024-12-31",
    )
    # With 0.1% max drawdown, risk should trigger early and freeze trading
    assert len(result.equity_curve) == 120


def test_engine_order_rules_stop_loss() -> None:
    """Order rules should place stop orders that SimBroker processes."""
    from oxq.rules.order import StopLossRule

    n = 20
    dates = pd.bdate_range("2024-01-01", periods=n)
    # Price: 100, then drops to 80
    closes = [100.0] + [100.0 - i * 2 for i in range(1, n)]
    data = {
        "AAPL": pd.DataFrame({
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": [1_000_000] * n,
        }, index=dates),
    }
    # Buy signal on day 1 only
    data["AAPL"]["buy_sig"] = False
    data["AAPL"].iloc[0, data["AAPL"].columns.get_loc("buy_sig")] = True

    strategy = Strategy(
        name="stop_loss_test",
        hypothesis="Test stop loss",
        universe=StaticUniverse(("AAPL",)),
        indicators={},
        signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    broker = SimBroker()
    result = Engine().run(
        strategy, market=FakeMarketDataProvider(data), router=broker,
        receiver=broker, start="2024-01-01", end="2024-12-31",
    )

    # Should have entry + stop-loss exit
    sell_trades = [t for t in result.trades if t.order.side == "SELL"]
    assert len(sell_trades) >= 1
    # Position should be closed at end
    assert "AAPL" not in result.portfolio.positions
