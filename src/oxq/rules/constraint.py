"""Pre-trade constraint rules — blacklist, max holdings, rebalance frequency."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Portfolio, RuleResult


class BlacklistRule:
    """Blocks trading for blacklisted symbols by setting their weight to 0."""

    name = "BlacklistRule"

    def __init__(self, symbols: set[str]) -> None:
        self.symbols = symbols

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        if symbol in self.symbols:
            return RuleResult(
                weights={symbol: 0.0},
                reason=f"{symbol} is blacklisted",
            )
        return RuleResult()


class MaxHoldingsRule:
    """Blocks new positions when portfolio is at max holdings limit."""

    name = "MaxHoldingsRule"

    def __init__(self, max_holdings: int) -> None:
        self.max_holdings = max_holdings

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        if symbol in portfolio.positions:
            return RuleResult()
        if len(portfolio.positions) >= self.max_holdings:
            return RuleResult(
                weights={symbol: 0.0},
                reason=f"max holdings {self.max_holdings} reached, blocking {symbol}",
            )
        return RuleResult()


class RebalanceFrequencyRule:
    """Freezes trading within a rebalance interval.

    Allows trading on the first bar, then blocks until interval_days have passed.
    """

    name = "RebalanceFrequencyRule"

    def __init__(self, interval_days: int = 5) -> None:
        self.interval_days = interval_days
        self._last_rebalance_date: pd.Timestamp | None = None

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> RuleResult:
        bar_date = row.name if hasattr(row, "name") else None
        if bar_date is None:
            return RuleResult()

        if self._last_rebalance_date is None:
            self._last_rebalance_date = bar_date
            return RuleResult()

        days_since = (bar_date - self._last_rebalance_date).days
        if days_since >= self.interval_days:
            self._last_rebalance_date = bar_date
            return RuleResult()

        return RuleResult(
            hold=True,
            reason=f"rebalance interval: {days_since}d < {self.interval_days}d",
        )
