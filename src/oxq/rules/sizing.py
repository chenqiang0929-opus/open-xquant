"""Position sizing functions for order size clipping."""

from __future__ import annotations

from decimal import Decimal

from oxq.core.types import Portfolio


def clip_to_max_position(
    shares: int,
    symbol: str,
    portfolio: Portfolio,
    max_shares: int,
) -> int:
    """Clip order shares so total position does not exceed a maximum.

    Calculates the remaining capacity (``max_shares`` minus current
    holdings) and returns the smaller of the requested shares and
    the remaining capacity. Returns 0 if already at or above the
    limit.

    This is the open-xquant equivalent of quantstrat's ``osMaxPos``.

    Parameters
    ----------
    shares : int
        Requested number of shares to buy.
    symbol : str
        The symbol being traded.
    portfolio : Portfolio
        Current portfolio state.
    max_shares : int
        Maximum allowed shares for this symbol.

    Returns
    -------
    int
        Clipped number of shares (>= 0).
    """
    current = portfolio.positions[symbol].shares if symbol in portfolio.positions else 0
    remaining = max_shares - current
    return max(0, min(shares, remaining))


def clip_to_pct_equity(
    shares: int,
    symbol: str,
    price: Decimal,
    portfolio: Portfolio,
    prices: dict[str, Decimal],
    max_pct: float,
) -> int:
    """Clip order shares so position value does not exceed a percentage of equity.

    Calculates the maximum allowable position value as
    ``total_portfolio_value * max_pct``, subtracts the current
    position value, and clips the requested shares accordingly.
    Returns 0 if already at or above the limit.

    This is the open-xquant equivalent of quantstrat's ``osPctEquity``.

    Parameters
    ----------
    shares : int
        Requested number of shares to buy.
    symbol : str
        The symbol being traded.
    price : Decimal
        Current price of the symbol.
    portfolio : Portfolio
        Current portfolio state.
    prices : dict[str, Decimal]
        Current prices for all symbols (for total_value calculation).
    max_pct : float
        Maximum allowed position as a fraction of total equity.

    Returns
    -------
    int
        Clipped number of shares (>= 0).
    """
    total_value = portfolio.total_value(prices)
    max_value = total_value * Decimal(str(max_pct))
    current_value = Decimal("0")
    if symbol in portfolio.positions:
        current_value = portfolio.positions[symbol].shares * price
    room = max_value - current_value
    if room <= 0 or price <= 0:
        return 0
    max_shares = int(room / price)
    return max(0, min(shares, max_shares))
