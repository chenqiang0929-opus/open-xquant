"""OrderGenerator — generates trade plans from target weights."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from oxq.core.types import Order, Position


@dataclass(frozen=True)
class PlannedOrder:
    """A trade order with context for human review.

    Attributes
    ----------
    order : Order
        The actual oxq Order object.
    current_shares : int
        Current position size.
    target_shares : int
        Desired position size after execution.
    current_weight : Decimal
        Current portfolio weight for this symbol.
    target_weight : Decimal
        Target portfolio weight.
    estimated_amount : Decimal
        Estimated trade value (shares * price).
    """

    order: Order
    current_shares: int
    target_shares: int
    current_weight: Decimal
    target_weight: Decimal
    estimated_amount: Decimal


def generate_orders(
    target_weights: dict[str, Decimal],
    positions: dict[str, Position],
    prices: dict[str, Decimal],
    total_capital: Decimal,
    lot_size: int = 1,
    currency: str = "CNY",
) -> list[PlannedOrder]:
    """Generate a trade plan from target weights.

    Computes the difference between target and current positions,
    producing BUY/SELL orders with context for human review.

    Parameters
    ----------
    target_weights : dict[str, Decimal]
        Target portfolio weights keyed by symbol. Symbols in positions
        but not in target_weights will be fully sold.
    positions : dict[str, Position]
        Current portfolio positions.
    prices : dict[str, Decimal]
        Current market prices per symbol.
    total_capital : Decimal
        Total portfolio value (cash + positions market value).
    lot_size : int
        Minimum trade unit. Default 1 (US stocks). Use 100 for A-shares.

    Returns
    -------
    list[PlannedOrder]
        Ordered list of planned trades with context.
    """
    planned: list[PlannedOrder] = []

    # All symbols: union of targets and current positions
    all_symbols = set(target_weights.keys()) | set(positions.keys())

    for symbol in sorted(all_symbols):
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue

        target_weight = target_weights.get(symbol, Decimal("0"))
        current_shares = positions[symbol].shares if symbol in positions else 0
        current_value = price * current_shares
        current_weight = current_value / total_capital if total_capital > 0 else Decimal("0")

        # Compute target shares with lot_size rounding
        raw_target = total_capital * target_weight / price
        target_shares = int(raw_target / lot_size) * lot_size

        delta = target_shares - current_shares
        if delta == 0:
            continue

        side = "BUY" if delta > 0 else "SELL"
        shares = abs(delta)

        planned.append(
            PlannedOrder(
                order=Order(symbol=symbol, side=side, shares=shares, currency=currency),
                current_shares=current_shares,
                target_shares=target_shares,
                current_weight=current_weight,
                target_weight=target_weight,
                estimated_amount=price * shares,
            )
        )

    return planned
