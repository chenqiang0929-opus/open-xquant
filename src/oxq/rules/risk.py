"""Risk rules — portfolio-level circuit breakers."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.core.types import Order, Portfolio


class MaxDrawdownRisk:
    """Portfolio-level circuit breaker based on maximum drawdown.

    Monitors the portfolio's peak-to-trough drawdown. When the drawdown
    exceeds ``max_drawdown``, liquidates all positions for the current
    symbol and freezes all subsequent rule stages (order rules, rebalance,
    exit, entry) for the current bar.

    The peak value is tracked across the entire backtest and only
    resets when a new high is reached.

    Parameters
    ----------
    max_drawdown : float
        Maximum allowed drawdown as a decimal fraction.
        Default is 0.15 (15%). Must be between 0 and 1.

    Attributes
    ----------
    name : str
        Rule identifier, always ``"MaxDrawdownRisk"``.

    Examples
    --------
    >>> strategy = Strategy(
    ...     risk_rules=[MaxDrawdownRisk(max_drawdown=0.15)],
    ...     ...
    ... )

    Notes
    -----
    This rule is evaluated once per symbol per bar. It computes a
    portfolio-level metric, so the drawdown check is the same for all
    symbols — but it generates a SELL order per symbol that has a
    position, ensuring all positions are liquidated when triggered.

    The ``hold`` signal freezes stages 2b-5 (order rules, rebalance,
    exit, entry) but does NOT prevent pending stop/limit orders from
    triggering in stage 2a (``process_pending_orders``).
    """

    name = "MaxDrawdownRisk"

    def __init__(self, max_drawdown: float = 0.15) -> None:
        self.max_drawdown = max_drawdown
        self._peak_value: Decimal = Decimal("0")

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> tuple[Order | None, bool]:
        """Evaluate drawdown risk.

        Parameters
        ----------
        prices : dict[str, Decimal] or None
            Current prices for all symbols. If None, falls back to
            using only the current symbol's close price.

        Returns
        -------
        tuple[Order | None, bool]
            (sell order if position exists, whether to freeze trading)
        """
        if prices is None:
            price = Decimal(str(float(row["close"])))
            if not price.is_finite():
                return None, False
            prices = {symbol: price}
        current_value = portfolio.total_value(prices)

        if current_value > self._peak_value:
            self._peak_value = current_value

        if self._peak_value == 0:
            return None, False

        drawdown = (self._peak_value - current_value) / self._peak_value

        if float(drawdown) >= self.max_drawdown:
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                return Order(symbol=symbol, side="SELL", shares=pos.shares), True
            return None, True

        return None, False


class DailyLossLimitRisk:
    """Freezes trading when single-day loss exceeds a threshold.

    Compares the portfolio value at the start of each trading day
    with the current value. If the intraday loss exceeds
    ``max_daily_loss``, freezes all subsequent rule stages for the
    remainder of the current bar. Does NOT liquidate positions —
    only prevents new orders from being generated.

    Parameters
    ----------
    max_daily_loss : float
        Maximum allowed single-day loss as a decimal fraction.
        Default is 0.03 (3%). Must be between 0 and 1.

    Attributes
    ----------
    name : str
        Rule identifier, always ``"DailyLossLimitRisk"``.

    Examples
    --------
    >>> strategy = Strategy(
    ...     risk_rules=[DailyLossLimitRisk(max_daily_loss=0.03)],
    ...     ...
    ... )

    Notes
    -----
    The "start of day" value is recorded on the first symbol
    evaluation of each new date. Subsequent symbol evaluations
    on the same date compare against this recorded value.
    """

    name = "DailyLossLimitRisk"

    def __init__(self, max_daily_loss: float = 0.03) -> None:
        self.max_daily_loss = max_daily_loss
        self._day_start_value: Decimal = Decimal("0")
        self._current_date: object = None

    def evaluate(
        self,
        symbol: str,
        row: pd.Series,
        portfolio: Portfolio,
        prices: dict[str, Decimal] | None = None,
    ) -> tuple[Order | None, bool]:
        """Evaluate daily loss limit.

        Parameters
        ----------
        prices : dict[str, Decimal] or None
            Current prices for all symbols. If None, falls back to
            using only the current symbol's close price.

        Returns
        -------
        tuple[Order | None, bool]
            (always None — no liquidation, whether to freeze trading)
        """
        bar_date = row.name if hasattr(row, "name") else None
        if prices is None:
            price = Decimal(str(float(row["close"])))
            if not price.is_finite():
                return None, False
            prices = {symbol: price}
        current_value = portfolio.total_value(prices)

        if bar_date != self._current_date:
            self._current_date = bar_date
            self._day_start_value = current_value

        if self._day_start_value == 0:
            return None, False

        daily_loss = (self._day_start_value - current_value) / self._day_start_value

        if float(daily_loss) >= self.max_daily_loss:
            return None, True

        return None, False
