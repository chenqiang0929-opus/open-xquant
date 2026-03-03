"""Universal strategy execution engine.

Executes the 4-phase pipeline: Universe → Indicator → Signal → Rule.
Provider-agnostic — the same engine serves backtest, paper trading,
and live trading.  The difference is which providers you plug in.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from oxq.core.strategy import Strategy
from oxq.core.types import Fill, FillReceiver, OrderRouter, Portfolio, Position
from oxq.data.providers import MarketDataProvider
from oxq.portfolio.analytics import RunResult


class Engine:
    """Universal strategy execution engine.

    Executes the 4-phase pipeline: Universe → Indicator → Signal → Rule.
    Provider-agnostic: the same engine serves backtest, paper trading,
    and live trading — the difference is which providers you plug in.

    Example — backtest mode::

        engine = Engine()
        result = engine.run(
            strategy,
            market=LocalMarketDataProvider(),
            router=sim_broker,
            receiver=sim_broker,
            start="2023-01-01",
            end="2024-12-31",
        )
    """

    def run(
        self,
        strategy: Strategy,
        market: MarketDataProvider,
        router: OrderRouter,
        receiver: FillReceiver,
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
        router : OrderRouter
            Order submission interface.
        receiver : FillReceiver
            Fill retrieval interface.
        start, end : str
            Date range.
        initial_cash : float
            Starting cash.
        run_through : str | None
            Stop after this phase: ``"indicator"`` or ``"signal"``.
            ``None`` runs the full pipeline including rules.
        """
        portfolio = Portfolio(cash=initial_cash)

        # ── Phase 0: Universe ──────────────────────────────────────────
        universe = strategy.universe.get_universe(as_of_date=end)

        mktdata: dict[str, pd.DataFrame] = {}
        for symbol in universe.symbols:
            mktdata[symbol] = market.get_bars(symbol, start, end).copy()

        # ── Phase 1: Indicator (vectorized, per symbol) ────────────────
        for symbol in universe.symbols:
            for ind_name, (indicator, params) in strategy.indicators.items():
                mktdata[symbol][ind_name] = indicator.compute(
                    mktdata[symbol], **params,
                )

        if run_through == "indicator":
            return RunResult(
                portfolio=portfolio, trades=[], equity_curve=[],
                mktdata=mktdata,
            )

        # ── Phase 2: Signal (vectorized, cross-sectional) ──────────────
        for sig_name, (signal, params) in strategy.signals.items():
            results = signal.compute(mktdata, **params)
            for symbol, series in results.items():
                mktdata[symbol][sig_name] = series

        if run_through == "signal":
            return RunResult(
                portfolio=portfolio, trades=[], equity_curve=[],
                mktdata=mktdata,
            )

        # ── Phase 3: Rule (bar-by-bar state machine) ───────────────────
        dates = mktdata[universe.symbols[0]].index
        trades: list[Fill] = []
        equity_curve: list[tuple[object, float]] = []

        for date in dates:
            # Rebalance rules (priority 3)
            for rule in strategy.rebalance_rules:
                for symbol in universe.symbols:
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        router.submit_order(order)

            # Exit rules (priority 4)
            for rule in strategy.exit_rules:
                for symbol in universe.symbols:
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        router.submit_order(order)

            # Entry rules second (lower priority)
            for rule in strategy.entry_rules:
                for symbol in universe.symbols:
                    row = mktdata[symbol].loc[date]
                    order = rule.evaluate(symbol, row, portfolio)
                    if order:
                        router.submit_order(order)

            # Process fills — SimBroker needs an explicit step call to
            # simulate fills at bar close; live brokers fill asynchronously.
            if hasattr(router, "fill_pending_orders"):
                router.fill_pending_orders(mktdata, date)

            for fill in receiver.get_fills():
                _apply_fill(portfolio, fill)
                trades.append(fill)

            # Record equity curve
            prices = {
                s: float(mktdata[s].loc[date, "close"])
                for s in universe.symbols
            }
            equity_curve.append((date, portfolio.total_value(prices)))

        return RunResult(
            portfolio=portfolio,
            trades=trades,
            equity_curve=equity_curve,
            mktdata=mktdata,
        )


def _apply_fill(portfolio: Portfolio, fill: Fill) -> None:
    """Update portfolio state based on a fill."""
    order = fill.order
    symbol = order.symbol
    cost = fill.filled_price * order.shares

    if order.side == "BUY":
        portfolio.cash -= cost
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
        portfolio.cash += cost
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
