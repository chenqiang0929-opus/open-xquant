"""Universal strategy execution engine.

Executes the 4-phase pipeline: Universe -> Indicator -> Signal -> Rule.
Provider-agnostic — the same engine serves backtest, paper trading,
and live trading.  The difference is which providers you plug in.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Literal

import pandas as pd

from oxq.core.strategy import Strategy
from oxq.core.types import Broker, Fill, Portfolio, Position
from oxq.data.providers import MarketDataProvider
from oxq.portfolio.analytics import RunResult

logger = logging.getLogger(__name__)


class Engine:
    """Universal strategy execution engine.

    Executes the 4-phase pipeline: Universe -> Indicator -> Signal -> Rule.
    Provider-agnostic: the same engine serves backtest, paper trading,
    and live trading — the difference is which providers you plug in.

    Example — backtest mode::

        engine = Engine()
        result = engine.run(
            strategy,
            market=LocalMarketDataProvider(),
            broker=sim_broker,
            start="2023-01-01",
            end="2024-12-31",
        )

    Example — step-by-step mode (live trading)::

        engine = Engine()
        engine.setup(strategy=strategy, market=market, broker=broker,
                     start="2024-01-01", end="2024-12-31")
        for date in engine.dates:
            engine.step(date)
        result = engine.result
    """

    def setup(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker: Broker,
        start: str,
        end: str,
        initial_cash: float = 100_000.0,
        run_through: Literal["indicator", "signal"] | None = None,
    ) -> None:
        """Initialize engine state and run vectorized phases.

        Parameters
        ----------
        strategy : Strategy
            The strategy definition.
        market : MarketDataProvider
            Data provider for loading bars.
        broker : Broker
            Unified broker interface (order routing, fills, lifecycle).
        start, end : str
            Date range.
        initial_cash : float
            Starting cash.
        run_through : str | None
            Stop after this phase: ``"indicator"`` or ``"signal"``.
            Controls which vectorized phases are computed.
        """
        self._strategy = strategy
        self._broker = broker
        self._portfolio = Portfolio(cash=Decimal(str(initial_cash)))

        # -- Phase 0: Universe ------------------------------------------------
        self._universe = strategy.universe.get_universe(as_of_date=end)

        self._mktdata: dict[str, pd.DataFrame] = {}
        for symbol in self._universe.symbols:
            self._mktdata[symbol] = market.get_bars(symbol, start, end).copy()

        # -- Benchmark data (recorded for post-run analysis) ----------
        self._benchmark_prices: dict[str, pd.Series] = {}
        for bench_symbol in strategy.benchmarks:
            bench_bars = market.get_bars(bench_symbol, start, end)
            self._benchmark_prices[bench_symbol] = bench_bars["close"].copy()

        # -- Phase 1: Indicator (vectorized, per symbol) ----------------------
        for symbol in self._universe.symbols:
            for ind_name, (indicator, params) in strategy.indicators.items():
                for dep_col in getattr(indicator, "depends_on", ()):
                    if dep_col not in self._mktdata[symbol].columns:
                        logger.warning(
                            "Indicator '%s' depends on column '%s' which does "
                            "not yet exist in mktdata. Ensure the producing "
                            "indicator is registered first.",
                            ind_name,
                            dep_col,
                        )
                self._mktdata[symbol][ind_name] = indicator.compute(
                    self._mktdata[symbol], **params,
                )

        if run_through == "indicator":
            return

        # -- Phase 2: Signal (vectorized, cross-sectional) --------------------
        for sig_name, (signal, params) in strategy.signals.items():
            results = signal.compute(self._mktdata, **params)
            for symbol, series in results.items():
                self._mktdata[symbol][sig_name] = series

        # -- Phase 3 state init -----------------------------------------------
        self._trades: list[Fill] = []
        self._equity_curve: list[tuple[object, float]] = []
        self._last_known_price: dict[str, float] = {}

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Union of all dates across symbols in mktdata."""
        symbols = self._universe.symbols
        result = self._mktdata[symbols[0]].index
        for sym in symbols[1:]:
            result = result.union(self._mktdata[sym].index)
        return result

    @property
    def result(self) -> RunResult:
        """Current result based on accumulated state."""
        return RunResult(
            portfolio=self._portfolio,
            trades=self._trades,
            equity_curve=self._equity_curve,
            mktdata=self._mktdata,
            benchmark_prices=self._benchmark_prices,
        )

    def step(self, date: pd.Timestamp) -> None:
        """Process a single bar through the Phase 3 rule state machine."""
        universe = self._universe
        strategy = self._strategy
        broker = self._broker
        portfolio = self._portfolio
        mktdata = self._mktdata

        # Build bar-wide prices dict for portfolio valuation
        bar_prices: dict[str, Decimal] = {}
        for s in universe.symbols:
            if date in mktdata[s].index:
                bar_prices[s] = Decimal(
                    str(float(mktdata[s].loc[date, "close"])),
                )
            elif s in self._last_known_price:
                bar_prices[s] = Decimal(str(self._last_known_price[s]))
        portfolio.bar_prices = bar_prices

        # ── Stage 1: Risk Rules ──────────────────────────────────────
        hold = False
        for rule in strategy.risk_rules:
            for symbol in universe.symbols:
                if date not in mktdata[symbol].index:
                    continue
                row = mktdata[symbol].loc[date]
                result_tuple = rule.evaluate(  # type: ignore[call-arg]
                    symbol, row, portfolio, prices=bar_prices,
                )
                order, should_hold = result_tuple  # type: ignore[misc]
                if should_hold:
                    hold = True
                if order:
                    broker.submit_order(order)

        # ── Stage 2a: Process pending orders (even if hold) ──────────
        # Sync pending SELL orders with current positions:
        # - No position → cancel all pending SELLs
        # - Position reduced → cap pending SELL shares to position size
        for sym in {
            m.order.symbol for m in broker.get_open_orders()
            if m.order.side == "SELL" and m.order.order_type != "market"
        }:
            pos = portfolio.positions.get(sym)
            broker.cap_pending_sells(sym, pos.shares if pos else 0)

        broker.on_bar_open(mktdata, date)

        # Apply fills from pending orders immediately so that
        # subsequent rules see up-to-date portfolio state.
        for fill in broker.get_fills():
            _apply_fill(portfolio, fill)
            self._trades.append(fill)

        # ── Stage 2b: Order Rules (skip if hold) ─────────────────────
        if not hold:
            for rule in strategy.order_rules:
                for symbol in universe.symbols:
                    if date not in mktdata[symbol].index:
                        continue
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        broker.submit_order(order)

        # ── Stage 3: Rebalance Rules (skip if hold) ──────────────────
        if not hold:
            for rule in strategy.rebalance_rules:
                for symbol in universe.symbols:
                    if date not in mktdata[symbol].index:
                        continue
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        broker.submit_order(order)

        # ── Stage 4: Exit Rules (skip if hold) ───────────────────────
        if not hold:
            for rule in strategy.exit_rules:
                for symbol in universe.symbols:
                    if date not in mktdata[symbol].index:
                        continue
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        broker.submit_order(order)

        # ── Stage 5: Entry Rules (skip if hold) ──────────────────────
        if not hold:
            for rule in strategy.entry_rules:
                for symbol in universe.symbols:
                    if date not in mktdata[symbol].index:
                        continue
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        broker.submit_order(order)

        # ── Fill market orders + collect all fills ────────────────────
        broker.on_bar_close(mktdata, date)

        for fill in broker.get_fills():
            _apply_fill(portfolio, fill)
            self._trades.append(fill)

        # Record equity curve (float for numpy analytics)
        prices: dict[str, Decimal] = {}
        for s in universe.symbols:
            if date in mktdata[s].index:
                close = Decimal(str(float(mktdata[s].loc[date, "close"])))
                self._last_known_price[s] = float(close)
                prices[s] = close
            elif s in self._last_known_price:
                prices[s] = Decimal(str(self._last_known_price[s]))
        self._equity_curve.append((date, float(portfolio.total_value(prices))))

    def run(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        broker: Broker,
        start: str,
        end: str,
        initial_cash: float = 100_000.0,
        run_through: Literal["indicator", "signal"] | None = None,
    ) -> RunResult:
        """Run the 4-phase strategy pipeline.

        Parameters
        ----------
        strategy : Strategy
            The strategy definition.
        market : MarketDataProvider
            Data provider for loading bars.
        broker : Broker
            Unified broker interface (order routing, fills, lifecycle).
        start, end : str
            Date range.
        initial_cash : float
            Starting cash.
        run_through : str | None
            Stop after this phase: ``"indicator"`` or ``"signal"``.
            ``None`` runs the full pipeline including rules.
        """
        self.setup(
            strategy=strategy, market=market, broker=broker,
            start=start, end=end, initial_cash=initial_cash,
            run_through=run_through,
        )

        if run_through == "indicator":
            return RunResult(
                portfolio=self._portfolio, trades=[], equity_curve=[],
                mktdata=self._mktdata,
                benchmark_prices=self._benchmark_prices,
            )

        if run_through == "signal":
            return RunResult(
                portfolio=self._portfolio, trades=[], equity_curve=[],
                mktdata=self._mktdata,
                benchmark_prices=self._benchmark_prices,
            )

        for date in self.dates:
            self.step(date)

        return self.result


def _apply_fill(portfolio: Portfolio, fill: Fill) -> None:
    """Update portfolio state based on a fill."""
    order = fill.order
    symbol = order.symbol
    cost = fill.filled_price * order.shares

    if order.side == "BUY":
        portfolio.cash -= cost + fill.fee
        if symbol in portfolio.positions:
            old = portfolio.positions[symbol]
            total_shares = old.shares + order.shares
            total_cost = old.avg_cost * old.shares + cost
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=total_shares,
                avg_cost=total_cost / total_shares,
            )
        else:
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                shares=order.shares,
                avg_cost=fill.filled_price,
            )
    elif order.side == "SELL":
        portfolio.cash += cost - fill.fee
        if symbol in portfolio.positions:
            old = portfolio.positions[symbol]
            remaining = old.shares - order.shares
            if remaining <= 0:
                del portfolio.positions[symbol]
            else:
                portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    shares=remaining,
                    avg_cost=old.avg_cost,
                )
