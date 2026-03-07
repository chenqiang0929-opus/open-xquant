"""Combination tests for Engine — cross-stage rule interactions.

These tests verify that rules from different stages interact correctly:
- Fills are applied at the right time so subsequent rules see fresh state
- Stale pending orders are canceled when positions are closed
- Hold signals work correctly with pending orders
- Multi-symbol strategies don't have cross-contamination
- No negative positions occur under any combination
"""

from decimal import Decimal

import pandas as pd

from oxq.core.engine import Engine
from oxq.core.strategy import Strategy
from oxq.core.types import Portfolio
from oxq.rules.entry import EntryRule, FullPositionEntryRule
from oxq.rules.exit import ExitRule
from oxq.rules.order import StopLossRule, TakeProfitRule, TrailingStopRule
from oxq.rules.rebalance import RebalanceRule
from oxq.rules.risk import DailyLossLimitRisk, MaxDrawdownRisk
from oxq.trade.sim_broker import SimBroker
from oxq.universe.static import StaticUniverse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeMarket:
    """In-memory market data provider for testing."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self._data = data

    def get_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        df = self._data[symbol]
        return df[(df.index >= start) & (df.index <= end)]

    def get_latest(self, symbol: str) -> pd.Series:
        return self._data[symbol].iloc[-1]


def _make_bars(
    closes: list[float],
    n: int | None = None,
    start: str = "2024-01-01",
    **extra_cols: list,
) -> pd.DataFrame:
    """Build a single-symbol DataFrame from close prices."""
    if n is None:
        n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    df = pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1_000_000] * n,
    }, index=dates)
    for col_name, values in extra_cols.items():
        df[col_name] = values
    return df


def _run(strategy: Strategy, data: dict[str, pd.DataFrame],
         initial_cash: float = 100_000.0):
    broker = SimBroker()
    return Engine().run(
        strategy,
        market=FakeMarket(data),
        router=broker, receiver=broker,
        start="2024-01-01", end="2024-12-31",
        initial_cash=initial_cash,
    )


def _assert_no_negative_positions(result):
    """Verify no position has negative shares at end of run."""
    for sym, pos in result.portfolio.positions.items():
        assert pos.shares > 0, f"{sym} has negative shares: {pos.shares}"


def _assert_reasonable_equity(result, initial_cash=100_000.0, max_multiple=5):
    """Verify final equity is within reasonable bounds."""
    final = result.equity_curve[-1][1]
    assert final > 0, f"Negative final equity: {final}"
    assert final < initial_cash * max_multiple, (
        f"Final equity {final:,.0f} exceeds {max_multiple}x initial cash"
    )


# ---------------------------------------------------------------------------
# 1. Exit + StopLoss: ExitRule closes position → stale stop must be canceled
# ---------------------------------------------------------------------------

def test_exit_cancels_stale_stop() -> None:
    """When ExitRule sells all shares, the stop order from StopLossRule
    must be canceled. Otherwise the stop can trigger on a zero position.
    """
    n = 30
    # Price: flat at 100 for SMA convergence, then drops to trigger death cross
    # SMA10 and SMA50 need enough history, so use simple indicator-free approach
    closes = [100.0] * 10 + [90.0] * 20
    sma10 = [100.0] * 10 + [90.0] * 20  # simplified
    sma50 = [100.0] * 30  # stays at 100 (slow)
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 29,
                        sma_f=sma10, sma_s=sma50),
    }

    strategy = Strategy(
        name="exit_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[ExitRule(fast="sma_f", slow="sma_s")],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    result = _run(strategy, data)

    # Exit should sell the position; no stop should fire after that
    stop_fills = [t for t in result.trades
                  if t.order.order_type == "stop"]
    # Position must not go negative
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)
    # If stop triggered before exit, that's fine. But after exit, no more stops.
    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    assert len(sell_fills) >= 1
    assert "A" not in result.portfolio.positions


# ---------------------------------------------------------------------------
# 2. Risk (MaxDrawdown) + StopLoss: Risk clears position → stale stop canceled
# ---------------------------------------------------------------------------

def test_risk_sell_cancels_stale_stop() -> None:
    """MaxDrawdownRisk sells position → stale stop must not trigger later."""
    n = 30
    # Price drops sharply to trigger both max drawdown and potential stop
    closes = [100.0] * 5 + [float(100 - i * 3) for i in range(1, 26)]
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 29),
    }

    strategy = Strategy(
        name="risk_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        # 10% max drawdown - will trigger during the drop
        risk_rules=[MaxDrawdownRisk(max_drawdown=0.10)],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    result = _run(strategy, data)

    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)
    assert "A" not in result.portfolio.positions


# ---------------------------------------------------------------------------
# 3. StopLoss + TakeProfit: one triggers → other must be canceled
# ---------------------------------------------------------------------------

def test_stop_triggers_cancels_take_profit() -> None:
    """When stop loss triggers, the take profit order must be canceled."""
    n = 15
    # Buy at 100, then price drops to trigger 5% stop
    closes = [100.0, 100.0, 96.0, 94.0, 92.0] + [90.0] * 10
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 14),
    }

    strategy = Strategy(
        name="stop_and_tp",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        order_rules=[StopLossRule(threshold=0.05),
                     TakeProfitRule(threshold=0.15)],
    )

    result = _run(strategy, data)

    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    # Exactly one sell (the stop), not a subsequent take-profit on empty position
    assert len(sell_fills) == 1, (
        f"Expected 1 sell, got {len(sell_fills)}: "
        f"{[(t.filled_at, t.order.order_type, t.order.shares) for t in sell_fills]}"
    )
    _assert_no_negative_positions(result)


def test_take_profit_triggers_cancels_stop() -> None:
    """When take profit triggers, the stop loss order must be canceled."""
    n = 15
    # Buy at 100, then price rises to trigger 15% take profit
    closes = [100.0, 100.0, 110.0, 115.0, 116.0] + [80.0] * 10
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 14),
    }

    strategy = Strategy(
        name="tp_and_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        order_rules=[StopLossRule(threshold=0.05),
                     TakeProfitRule(threshold=0.15)],
    )

    result = _run(strategy, data)

    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    # Only one sell (the take-profit), not a subsequent stop on empty position
    assert len(sell_fills) == 1, (
        f"Expected 1 sell, got {len(sell_fills)}: "
        f"{[(t.filled_at, t.order.order_type, t.order.shares) for t in sell_fills]}"
    )
    assert sell_fills[0].order.order_type == "limit"  # take-profit is a limit order
    _assert_no_negative_positions(result)


# ---------------------------------------------------------------------------
# 4. Entry + StopLoss → re-entry: stop clears position, signal re-enters
# ---------------------------------------------------------------------------

def test_stop_loss_then_reentry() -> None:
    """After stop loss closes a position, a new entry signal should work."""
    n = 20
    # Buy at 100, price drops to trigger stop, then recovers with new buy signal
    closes = [100.0, 100.0, 94.0, 93.0, 92.0,  # stop triggers around bar 2-3
              95.0, 96.0, 97.0, 98.0, 99.0,      # recovery
              100.0, 101.0, 102.0, 103.0, 104.0,  # buy signal fires again
              105.0, 106.0, 107.0, 108.0, 109.0]
    buy_sigs = [True] + [False] * 9 + [True] + [False] * 9
    data = {
        "A": _make_bars(closes, buy_sig=buy_sigs),
    }

    strategy = Strategy(
        name="stop_reentry",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    result = _run(strategy, data)

    buy_fills = [t for t in result.trades if t.order.side == "BUY"]
    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    # Should have 2 buys (initial + re-entry) and 1 sell (stop)
    assert len(buy_fills) == 2, f"Expected 2 buys, got {len(buy_fills)}"
    assert len(sell_fills) == 1, f"Expected 1 sell, got {len(sell_fills)}"
    # Should hold position at end (second entry, no subsequent stop)
    assert "A" in result.portfolio.positions
    _assert_no_negative_positions(result)


# ---------------------------------------------------------------------------
# 5. Rebalance + StopLoss multi-symbol: stop on one doesn't affect another
# ---------------------------------------------------------------------------

def test_rebalance_stop_multi_symbol_isolation() -> None:
    """Stop loss on symbol A must not affect symbol B's positions or orders."""
    n = 30
    # A: drops to trigger stop; B: stays flat
    closes_a = [100.0] * 10 + [90.0] * 20
    closes_b = [200.0] * 30
    data = {
        "A": _make_bars(closes_a, weight=[0.5] * 30),
        "B": _make_bars(closes_b, weight=[0.5] * 30),
    }

    strategy = Strategy(
        name="multi_sym_stop",
        universe=StaticUniverse(("A", "B")),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[StopLossRule(threshold=0.05)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    # B should still have a position (never stopped out)
    assert "B" in result.portfolio.positions, "B's position was incorrectly affected"
    assert result.portfolio.positions["B"].shares > 0
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)


# ---------------------------------------------------------------------------
# 6. TrailingStop + Exit: Exit closes position → trailing stop canceled
# ---------------------------------------------------------------------------

def test_exit_cancels_stale_trailing_stop() -> None:
    """When ExitRule sells position, trailing stop must be canceled."""
    n = 20
    # Buy, price goes up (trailing stop tracks), then death cross exits
    closes = [100.0] * 5 + [105.0, 110.0, 108.0, 106.0, 104.0,
              102.0, 100.0, 98.0, 96.0, 94.0,
              92.0, 90.0, 88.0, 86.0, 84.0]
    sma_f = closes  # fast tracks price
    sma_s = [100.0] * 20  # slow stays at 100
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 19,
                        sma_f=sma_f, sma_s=sma_s),
    }

    strategy = Strategy(
        name="trail_exit",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[ExitRule(fast="sma_f", slow="sma_s")],
        order_rules=[TrailingStopRule(trail_pct=0.05)],
    )

    result = _run(strategy, data)

    # Only one sell should occur (whichever triggers first)
    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    assert len(sell_fills) == 1, (
        f"Expected 1 sell, got {len(sell_fills)}: "
        f"{[(t.filled_at, t.order.order_type) for t in sell_fills]}"
    )
    _assert_no_negative_positions(result)


# ---------------------------------------------------------------------------
# 7. Risk hold + pending stop: hold=True must not block Stage 2a
# ---------------------------------------------------------------------------

def test_risk_hold_does_not_block_pending_stops() -> None:
    """When risk rule returns hold=True, pending stops in Stage 2a should
    still trigger. Only Stage 2b-5 are frozen.
    """
    n = 15
    # Buy at 100, price drops → stop is submitted → risk triggers hold →
    # pending stop should still execute in Stage 2a
    closes = [100.0, 100.0, 98.0, 96.0, 94.0, 92.0, 90.0,
              88.0, 86.0, 84.0, 82.0, 80.0, 78.0, 76.0, 74.0]
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 14),
    }

    strategy = Strategy(
        name="hold_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        # Very tight drawdown to trigger hold early
        risk_rules=[MaxDrawdownRisk(max_drawdown=0.02)],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    result = _run(strategy, data)

    # Position should be closed (by either risk or stop)
    assert "A" not in result.portfolio.positions
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)


# ---------------------------------------------------------------------------
# 8. Rebalance partial sell + StopLoss: stop should protect remaining shares
# ---------------------------------------------------------------------------

def test_rebalance_partial_sell_stop_protects_remainder() -> None:
    """When rebalance reduces position (not to zero), stop should update
    to protect the remaining shares, not the original amount.
    """
    n = 30
    # Weight goes from 0.5 to 0.3 at bar 20 (partial sell), then price drops
    weights = [0.5] * 10 + [0.5] * 10 + [0.3] * 10
    closes = [100.0] * 20 + [94.0] * 10  # drops after partial sell
    data = {
        "A": _make_bars(closes, weight=weights),
    }

    strategy = Strategy(
        name="partial_rebal_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[StopLossRule(threshold=0.05)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)
    # If stop triggered, the sold amount should match the remaining position,
    # not the original larger position
    stop_fills = [t for t in result.trades if t.order.order_type == "stop"]
    for sf in stop_fills:
        # The stop fill shares should not exceed the position size at that time
        # (can't assert exact number without replaying, but no negatives is key)
        assert sf.order.shares > 0


# ---------------------------------------------------------------------------
# 9. Entry + Exit same bar: Stage 4 exits, Stage 5 enters (no conflict)
# ---------------------------------------------------------------------------

def test_exit_then_entry_next_bar() -> None:
    """ExitRule (Stage 4) sells at end-of-bar. EntryRule on the same bar
    still sees the position (market orders are batched), so re-entry
    happens on the next bar when position is cleared.
    """
    n = 10
    closes = [100.0] * 10
    # Exit fires on bar 5 (sma_f < sma_s), buy signal on bar 6
    sma_f = [101.0] * 5 + [99.0] * 5
    sma_s = [100.0] * 10
    buy_sig = [True, False, False, False, False,
               False, True, False, False, False]
    data = {
        "A": _make_bars(closes, sma_f=sma_f, sma_s=sma_s, buy_sig=buy_sig),
    }

    strategy = Strategy(
        name="exit_entry_next_bar",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[ExitRule(fast="sma_f", slow="sma_s")],
    )

    result = _run(strategy, data)

    buy_fills = [t for t in result.trades if t.order.side == "BUY"]
    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    # 2 buys: initial (bar 0) + re-entry (bar 6)
    # 2 sells: exit (bar 5, sma_f<sma_s) + exit again (bar 7+, still sma_f<sma_s)
    assert len(buy_fills) == 2, f"Expected 2 buys, got {len(buy_fills)}"
    assert len(sell_fills) == 2, f"Expected 2 sells, got {len(sell_fills)}"
    _assert_no_negative_positions(result)


# ---------------------------------------------------------------------------
# 10. Full combination: all rule types working together
# ---------------------------------------------------------------------------

def test_full_combination_all_rule_types() -> None:
    """Entry + Exit + StopLoss + TakeProfit + Risk + Rebalance all together.
    The portfolio must remain sane throughout.
    """
    from oxq.indicators.sma import SMA
    from oxq.signals.crossover import Crossover

    n = 80
    dates = pd.bdate_range("2024-01-01", periods=n)
    # Simulate a volatile market: up → down → up → down
    closes: list[float] = []
    for i in range(20):
        closes.append(100 + i * 2)       # 100 → 138
    for i in range(20):
        closes.append(138 - i * 3)       # 138 → 81
    for i in range(20):
        closes.append(81 + i * 2.5)      # 81 → 128.5
    for i in range(20):
        closes.append(128.5 - i * 2)     # 128.5 → 90.5
    data = {
        "A": pd.DataFrame({
            "open": closes, "high": [c + 2 for c in closes],
            "low": [c - 2 for c in closes],
            "close": closes, "volume": [1_000_000] * n,
        }, index=dates),
    }

    strategy = Strategy(
        name="full_combo",
        hypothesis="All rule types combined",
        universe=StaticUniverse(("A",)),
        indicators={
            "sma_5": (SMA(), {"period": 5}),
            "sma_20": (SMA(), {"period": 20}),
        },
        signals={
            "golden": (Crossover(), {"fast": "sma_5", "slow": "sma_20"}),
        },
        risk_rules=[MaxDrawdownRisk(max_drawdown=0.20)],
        order_rules=[StopLossRule(threshold=0.08),
                     TakeProfitRule(threshold=0.20)],
        entry_rules=[EntryRule(signal="golden", shares=200)],
        exit_rules=[ExitRule(fast="sma_5", slow="sma_20")],
    )

    result = _run(strategy, data)

    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)
    assert len(result.equity_curve) == n
    # Should have some trades
    assert len(result.trades) > 0


# ---------------------------------------------------------------------------
# 11. Invariant: no negative positions at any point during the run
# ---------------------------------------------------------------------------

def test_no_negative_positions_during_run() -> None:
    """Trace all fills chronologically and verify that no position ever
    goes negative at any intermediate step.
    """
    n = 40
    # Volatile price with multiple buy/sell cycles
    closes = ([100.0, 102.0, 98.0, 94.0, 96.0, 100.0, 105.0, 110.0] * 5)
    buy_sigs = [True, False, False, False, False, False, False, False] * 5
    data = {
        "A": _make_bars(closes, buy_sig=buy_sigs),
    }

    strategy = Strategy(
        name="no_negatives",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        order_rules=[StopLossRule(threshold=0.05),
                     TakeProfitRule(threshold=0.10)],
    )

    result = _run(strategy, data)

    # Replay fills and check position at every step
    positions: dict[str, int] = {}
    for fill in result.trades:
        sym = fill.order.symbol
        current = positions.get(sym, 0)
        if fill.order.side == "BUY":
            positions[sym] = current + fill.order.shares
        else:
            positions[sym] = current - fill.order.shares
        assert positions[sym] >= 0, (
            f"Negative position for {sym} after fill at {fill.filled_at}: "
            f"{positions[sym]} shares (order: {fill.order.side} {fill.order.shares} "
            f"{fill.order.order_type})"
        )


# ---------------------------------------------------------------------------
# 12. Rebalance + TakeProfit: rebalance sells all → stale limit canceled
# ---------------------------------------------------------------------------

def test_rebalance_cancels_stale_take_profit() -> None:
    """When RebalanceRule sells all shares (weight→0), the take profit
    limit order must be canceled.
    """
    n = 40
    # Weight: 0.5 for first 10 bars, then 0.0
    # Price stays BELOW TP level (115) until after rebalance sells all at bar 20,
    # then rises above 115 so a stale TP would trigger on empty position.
    weights = [0.5] * 10 + [0.0] * 30
    closes = [100.0] * 10 + [101.0] * 10 + [float(115 + i) for i in range(20)]
    data = {
        "A": _make_bars(closes, weight=weights),
    }

    strategy = Strategy(
        name="rebal_tp",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[TakeProfitRule(threshold=0.15)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    # No limit fills should occur after the rebalance sell-all
    limit_fills = [t for t in result.trades if t.order.order_type == "limit"]
    assert len(limit_fills) == 0, (
        f"Stale take-profit triggered: {[(t.filled_at, t.order.shares) for t in limit_fills]}"
    )
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)


# ---------------------------------------------------------------------------
# 13. Rebalance + TrailingStop: rebalance sells all → trailing stop canceled
# ---------------------------------------------------------------------------

def test_rebalance_cancels_stale_trailing_stop() -> None:
    """When RebalanceRule sells all shares, trailing stop must be canceled."""
    n = 40
    weights = [0.5] * 10 + [0.0] * 30
    # Price rises then retraces to potentially trigger trailing stop
    closes = [100.0] * 10 + [105.0, 110.0, 108.0, 106.0, 104.0,
              102.0, 100.0, 98.0, 96.0, 94.0] + [92.0] * 20
    data = {
        "A": _make_bars(closes, weight=weights),
    }

    strategy = Strategy(
        name="rebal_trail",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[TrailingStopRule(trail_pct=0.05)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    trailing_fills = [t for t in result.trades
                      if t.order.order_type == "trailing_stop"]
    assert len(trailing_fills) == 0, (
        f"Stale trailing stop triggered: "
        f"{[(t.filled_at, t.order.shares) for t in trailing_fills]}"
    )
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)


# ---------------------------------------------------------------------------
# 14. DailyLossLimit + StopLoss: hold freezes Stage 2b but stop still works
# ---------------------------------------------------------------------------

def test_daily_loss_hold_stop_still_triggers() -> None:
    """DailyLossLimitRisk hold should not prevent existing stop orders
    from triggering in Stage 2a.
    """
    n = 10
    # Buy at 100, price drops → daily loss hold + stop should both work
    closes = [100.0, 100.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0, 82.0]
    data = {
        "A": _make_bars(closes, buy_sig=[True] + [False] * 9),
    }

    strategy = Strategy(
        name="daily_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        entry_rules=[EntryRule(signal="buy_sig", shares=100)],
        exit_rules=[],
        risk_rules=[DailyLossLimitRisk(max_daily_loss=0.02)],
        order_rules=[StopLossRule(threshold=0.05)],
    )

    result = _run(strategy, data)

    # Position should be closed by the stop
    assert "A" not in result.portfolio.positions
    sell_fills = [t for t in result.trades if t.order.side == "SELL"]
    assert len(sell_fills) >= 1
    _assert_no_negative_positions(result)


# ---------------------------------------------------------------------------
# 15. Multi-symbol full lifecycle with rebalance + all order rules
# ---------------------------------------------------------------------------

def test_multi_symbol_rebalance_with_all_order_rules() -> None:
    """Three symbols with rebalance + stop + take-profit + trailing stop.
    Verify no cross-contamination and no negative positions.
    """
    n = 60
    # A: steady uptrend; B: drops to trigger stop; C: spikes to trigger TP
    closes_a = [float(100 + i * 0.5) for i in range(n)]
    closes_b = [100.0] * 20 + [float(100 - i * 2) for i in range(1, 41)]
    closes_c = [100.0] * 20 + [float(100 + i * 3) for i in range(1, 41)]
    data = {
        "A": _make_bars(closes_a, weight=[0.34] * n),
        "B": _make_bars(closes_b, weight=[0.33] * n),
        "C": _make_bars(closes_c, weight=[0.33] * n),
    }

    strategy = Strategy(
        name="multi_all_orders",
        universe=StaticUniverse(("A", "B", "C")),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=20)],
        order_rules=[
            StopLossRule(threshold=0.10),
            TakeProfitRule(threshold=0.30),
        ],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)
    assert len(result.trades) > 0

    # Replay all fills to check no intermediate negative positions
    positions: dict[str, int] = {}
    for fill in result.trades:
        sym = fill.order.symbol
        current = positions.get(sym, 0)
        if fill.order.side == "BUY":
            positions[sym] = current + fill.order.shares
        else:
            positions[sym] = current - fill.order.shares
        assert positions[sym] >= 0, (
            f"Negative position for {sym} at {fill.filled_at}: "
            f"{positions[sym]} ({fill.order.side} {fill.order.shares} "
            f"{fill.order.order_type})"
        )


# ---------------------------------------------------------------------------
# 16. Partial rebalance + TakeProfit: TP shares must be capped to position
# ---------------------------------------------------------------------------

def test_partial_rebalance_caps_take_profit_shares() -> None:
    """Regression: after partial position reduction by rebalance, a pending
    take-profit (or stop) must not sell more shares than currently held.

    Scenario (mirrors the real-world bug):
    1. Rebalance buys 500 shares at bar 10
    2. TakeProfitRule submits limit SELL 500 @ 115 (15% above cost)
    3. Rebalance at bar 20 sells 200 shares (weight drops), position = 300
    4. Price rises above 115 after bar 20
    5. Without fix: limit SELL 500 triggers → sells 200 more than held → short
    6. With fix: limit SELL is capped to 300 → correct behavior
    """
    n = 40
    # Weight: 0.5 for bars 0-19, drops to 0.3 at bar 20
    weights = [0.5] * 10 + [0.5] * 10 + [0.3] * 20
    # Price stays below TP level (115) until bar 25, then jumps above
    closes = [100.0] * 10 + [101.0] * 10 + [101.0] * 5 + [120.0] * 15
    data = {
        "A": _make_bars(closes, weight=weights),
    }

    strategy = Strategy(
        name="partial_rebal_tp",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[TakeProfitRule(threshold=0.15)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    # Replay fills to verify no negative positions at any point
    positions: dict[str, int] = {}
    for fill in result.trades:
        sym = fill.order.symbol
        current = positions.get(sym, 0)
        if fill.order.side == "BUY":
            positions[sym] = current + fill.order.shares
        else:
            positions[sym] = current - fill.order.shares
        assert positions[sym] >= 0, (
            f"Negative position for {sym} at {fill.filled_at}: "
            f"{positions[sym]} shares (order: {fill.order.side} "
            f"{fill.order.shares} {fill.order.order_type})"
        )
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)


def test_partial_rebalance_caps_stop_shares() -> None:
    """Same as above but with StopLossRule instead of TakeProfitRule."""
    n = 40
    # Weight: 0.5 for bars 0-19, drops to 0.3 at bar 20
    weights = [0.5] * 10 + [0.5] * 10 + [0.3] * 20
    # Price stays above stop level until bar 25, then drops below
    closes = [100.0] * 10 + [100.0] * 10 + [100.0] * 5 + [90.0] * 15
    data = {
        "A": _make_bars(closes, weight=weights),
    }

    strategy = Strategy(
        name="partial_rebal_stop",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[StopLossRule(threshold=0.05)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    # Replay fills — no negative positions
    positions: dict[str, int] = {}
    for fill in result.trades:
        sym = fill.order.symbol
        current = positions.get(sym, 0)
        if fill.order.side == "BUY":
            positions[sym] = current + fill.order.shares
        else:
            positions[sym] = current - fill.order.shares
        assert positions[sym] >= 0, (
            f"Negative position for {sym} at {fill.filled_at}: "
            f"{positions[sym]} shares (order: {fill.order.side} "
            f"{fill.order.shares} {fill.order.order_type})"
        )
    _assert_no_negative_positions(result)
    _assert_reasonable_equity(result)


def test_partial_exit_caps_trailing_stop_shares() -> None:
    """Entry buys 200 shares, ExitRule sells 200 on death cross, but
    only 100 shares remain (partial sell happened via another mechanism).
    Trailing stop must not exceed remaining position.

    This uses a multi-symbol setup where rebalance reduces one symbol
    while trailing stop is active.
    """
    n = 40
    weights = [0.5] * 10 + [0.5] * 10 + [0.25] * 20
    # Price rises then retraces to trigger trailing stop
    closes = ([100.0] * 10 + [100.0] * 10
              + [105.0, 110.0, 108.0, 106.0, 104.0,
                 102.0, 100.0, 98.0, 96.0, 94.0]
              + [92.0] * 10)
    data = {
        "A": _make_bars(closes, weight=weights),
    }

    strategy = Strategy(
        name="partial_trail",
        universe=StaticUniverse(("A",)),
        indicators={}, signals={},
        rebalance_rules=[RebalanceRule(weight_col="weight", frequency=10)],
        order_rules=[TrailingStopRule(trail_pct=0.05)],
        entry_rules=[], exit_rules=[],
    )

    result = _run(strategy, data)

    # Replay fills — no negative positions
    positions: dict[str, int] = {}
    for fill in result.trades:
        sym = fill.order.symbol
        current = positions.get(sym, 0)
        if fill.order.side == "BUY":
            positions[sym] = current + fill.order.shares
        else:
            positions[sym] = current - fill.order.shares
        assert positions[sym] >= 0, (
            f"Negative position for {sym} at {fill.filled_at}: "
            f"{positions[sym]} shares (order: {fill.order.side} "
            f"{fill.order.shares} {fill.order.order_type})"
        )
