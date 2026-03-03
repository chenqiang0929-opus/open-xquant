"""Rebalance rule — generates orders to align portfolio with target weights."""

from __future__ import annotations

import pandas as pd

from oxq.core.types import Order, Portfolio


class RebalanceRule:
    """Rebalance portfolio to match target weights on a periodic schedule.

    Reads ``weight_col`` from each bar (produced by a Signal like
    TopNRanking). Every ``frequency`` bars, computes target shares and
    generates BUY/SELL orders to converge.
    """

    name = "RebalanceRule"

    def __init__(self, weight_col: str, frequency: int = 10) -> None:
        self.weight_col = weight_col
        self.frequency = frequency
        self._bar_count: int = 0
        self._current_date: object | None = None

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
    ) -> Order | None:
        """Generate a rebalance order if this is a rebalance bar."""
        bar_date = row.name
        if bar_date != self._current_date:
            self._current_date = bar_date
            self._bar_count += 1

        if self._bar_count % self.frequency != 0:
            return None

        target_weight = float(row.get(self.weight_col, 0.0))
        if pd.isna(target_weight):
            target_weight = 0.0

        price = float(row["close"])
        if price <= 0:
            return None

        portfolio_value = _estimate_portfolio_value(portfolio, symbol, price)
        if portfolio_value <= 0:
            return None

        target_shares = int(portfolio_value * target_weight / price)
        current_shares = 0
        if symbol in portfolio.positions:
            current_shares = portfolio.positions[symbol].shares

        delta = target_shares - current_shares
        if delta > 0:
            return Order(symbol=symbol, side="BUY", shares=delta)
        elif delta < 0:
            return Order(symbol=symbol, side="SELL", shares=abs(delta))
        return None


def _estimate_portfolio_value(
    portfolio: Portfolio,
    current_symbol: str,
    current_price: float,
) -> float:
    """Estimate total portfolio value.

    Uses market price for current symbol, avg_cost for others.
    """
    total = portfolio.cash
    for sym, pos in portfolio.positions.items():
        if sym == current_symbol:
            total += pos.shares * current_price
        else:
            total += pos.shares * pos.avg_cost
    return total
