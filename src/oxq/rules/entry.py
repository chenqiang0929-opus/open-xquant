"""Entry rules — generate BUY orders when a signal fires."""

from __future__ import annotations

import pandas as pd

from oxq.core.types import Order, Portfolio


class EntryRule:
    """Buy when the named signal column is True and no position is held."""

    name = "EntryRule"

    def __init__(self, signal: str, shares: int = 100) -> None:
        self.signal = signal
        self.shares = shares

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        if row.get(self.signal) and symbol not in portfolio.positions:
            return Order(symbol=symbol, side="BUY", shares=self.shares)
        return None


class TargetValueEntryRule:
    """Buy to reach a target market value when the signal fires."""

    name = "TargetValueEntryRule"

    def __init__(self, signal: str, target_value: float) -> None:
        self.signal = signal
        self.target_value = target_value

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        if not row.get(self.signal):
            return None
        price = float(row["close"])
        current_shares = 0
        if symbol in portfolio.positions:
            current_shares = portfolio.positions[symbol].shares
        target_shares = int(self.target_value / price)
        shares_to_buy = target_shares - current_shares
        if shares_to_buy <= 0:
            return None
        return Order(symbol=symbol, side="BUY", shares=shares_to_buy)


class FullPositionEntryRule:
    """Buy with all available cash when the signal fires."""

    name = "FullPositionEntryRule"

    def __init__(self, signal: str) -> None:
        self.signal = signal

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        if not row.get(self.signal):
            return None
        price = float(row["close"])
        shares = int(portfolio.cash / price)
        if shares <= 0:
            return None
        return Order(symbol=symbol, side="BUY", shares=shares)
