"""Order rules — generate stop/limit/trailing_stop orders for position protection."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Order, Portfolio


class StopLossRule:
    """Place a stop order to sell when unrealized loss exceeds a threshold.

    When a position is held, generates a stop SELL order with
    ``stop_price = avg_cost * (1 - threshold)``. The order is
    submitted to SimBroker's order book and triggers when the
    market price drops to or below the stop price.

    If a stop order already exists for the same symbol, SimBroker
    replaces it (no duplicates).

    Parameters
    ----------
    threshold : float
        Maximum allowed loss as a decimal fraction of entry cost.
        Default is 0.05 (5%). Must be between 0 and 1.

    Attributes
    ----------
    name : str
        Rule identifier, always ``"StopLossRule"``.

    Examples
    --------
    >>> strategy = Strategy(
    ...     order_rules=[StopLossRule(threshold=0.05)],
    ...     ...
    ... )

    Notes
    -----
    The stop price is based on the position's ``avg_cost``, which
    may change if the position is scaled into over multiple entries.
    Each bar, a new stop order is submitted with the current
    avg_cost, ensuring the stop level stays correct after averaging.
    """

    name = "StopLossRule"

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        """Generate a stop order if a position is held.

        Returns
        -------
        Order or None
            A stop SELL order, or None if no position exists.
        """
        if symbol not in portfolio.positions:
            return None
        pos = portfolio.positions[symbol]
        stop_price = pos.avg_cost * (1 - Decimal(str(self.threshold)))
        return Order(
            symbol=symbol,
            side="SELL",
            shares=pos.shares,
            order_type="stop",
            stop_price=stop_price,
        )


class TakeProfitRule:
    """Place a limit order to sell when unrealized profit exceeds a threshold.

    When a position is held, generates a limit SELL order with
    ``limit_price = avg_cost * (1 + threshold)``. The order is
    submitted to SimBroker's order book and triggers when the
    market price rises to or above the limit price.

    Parameters
    ----------
    threshold : float
        Profit target as a decimal fraction of entry cost.
        Default is 0.15 (15%). Must be between 0 and 1.

    Attributes
    ----------
    name : str
        Rule identifier, always ``"TakeProfitRule"``.

    Examples
    --------
    >>> strategy = Strategy(
    ...     order_rules=[TakeProfitRule(threshold=0.15)],
    ...     ...
    ... )
    """

    name = "TakeProfitRule"

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        """Generate a limit order if a position is held.

        Returns
        -------
        Order or None
            A limit SELL order, or None if no position exists.
        """
        if symbol not in portfolio.positions:
            return None
        pos = portfolio.positions[symbol]
        limit_price = pos.avg_cost * (1 + Decimal(str(self.threshold)))
        return Order(
            symbol=symbol,
            side="SELL",
            shares=pos.shares,
            order_type="limit",
            limit_price=limit_price,
        )


class TrailingStopRule:
    """Place a trailing stop order that follows the price upward.

    When a position is held, generates a trailing_stop SELL order.
    SimBroker tracks the high-water mark internally and triggers
    the order when the price retraces ``trail_pct`` from the peak.

    Parameters
    ----------
    trail_pct : float
        Maximum allowed retracement from high-water mark as a
        decimal fraction. Default is 0.05 (5%). Must be between
        0 and 1.

    Attributes
    ----------
    name : str
        Rule identifier, always ``"TrailingStopRule"``.

    Examples
    --------
    >>> strategy = Strategy(
    ...     order_rules=[TrailingStopRule(trail_pct=0.05)],
    ...     ...
    ... )
    """

    name = "TrailingStopRule"

    def __init__(self, trail_pct: float = 0.05) -> None:
        self.trail_pct = trail_pct

    def evaluate(
        self, symbol: str, row: pd.Series, portfolio: Portfolio,
    ) -> Order | None:
        """Generate a trailing stop order if a position is held.

        Returns
        -------
        Order or None
            A trailing_stop SELL order, or None if no position exists.
        """
        if symbol not in portfolio.positions:
            return None
        pos = portfolio.positions[symbol]
        return Order(
            symbol=symbol,
            side="SELL",
            shares=pos.shares,
            order_type="trailing_stop",
            trail_pct=self.trail_pct,
        )
