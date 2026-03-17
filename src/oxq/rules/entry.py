"""Entry rules — generate BUY orders when a signal fires."""

from __future__ import annotations

from decimal import Decimal

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
        shares = int(float(portfolio.cash) / price)
        if shares <= 0:
            return None
        return Order(symbol=symbol, side="BUY", shares=shares)


class SizedEntryRule:
    """Buy when a signal fires, with optional position sizing constraints.

    Extends the basic EntryRule with two optional sizing guards:

    - ``max_position``: caps the total number of shares held
    - ``max_pct_equity``: caps the position value as a percentage
      of total portfolio equity

    Both constraints are applied sequentially — the final order size
    is the minimum allowed by all active constraints. If the result
    is zero or negative, no order is generated.

    Parameters
    ----------
    signal : str
        Name of the signal column to monitor (must be boolean).
    shares : int
        Base number of shares to buy (before sizing constraints).
        Default is 100.
    max_position : int or None
        Maximum total shares allowed for this symbol.
    max_pct_equity : float or None
        Maximum position value as a fraction of total equity.

    Attributes
    ----------
    name : str
        Rule identifier, always ``"SizedEntryRule"``.

    Examples
    --------
    >>> SizedEntryRule(signal="golden_cross", shares=100, max_position=500)
    >>> SizedEntryRule(signal="golden_cross", shares=100, max_pct_equity=0.2)
    """

    name = "SizedEntryRule"

    def __init__(
        self,
        signal: str,
        shares: int = 100,
        max_position: int | None = None,
        max_pct_equity: float | None = None,
    ) -> None:
        self.signal = signal
        self.shares = shares
        self.max_position = max_position
        self.max_pct_equity = max_pct_equity

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        """Evaluate signal and apply sizing constraints."""
        if not row.get(self.signal) or symbol in portfolio.positions:
            return None

        shares = self.shares

        if self.max_position is not None:
            from oxq.rules.sizing import clip_to_max_position
            shares = clip_to_max_position(shares, symbol, portfolio, self.max_position)

        if self.max_pct_equity is not None:
            from oxq.rules.sizing import clip_to_pct_equity
            price = Decimal(str(float(row["close"])))
            if not price.is_finite():
                return None
            prices = dict(portfolio.bar_prices) if portfolio.bar_prices else {}
            prices[symbol] = price  # ensure current symbol has latest price
            shares = clip_to_pct_equity(
                shares, symbol, price, portfolio, prices, self.max_pct_equity,
            )

        if shares <= 0:
            return None
        return Order(symbol=symbol, side="BUY", shares=shares)
