"""Export RunResult to a directory of files for subprocess communication."""

from __future__ import annotations

from typing import Any

import pandas as pd

from oxq.core.types import BarSnapshot, Fill

_TRADE_COLUMNS = [
    "filled_at", "symbol", "side", "shares", "order_type",
    "limit_price", "stop_price", "filled_price", "fee",
]


def _flatten_trades(trades: list[Fill]) -> pd.DataFrame:
    """Flatten Fill list to a flat DataFrame."""
    if not trades:
        return pd.DataFrame(columns=_TRADE_COLUMNS)
    rows = []
    for f in trades:
        rows.append({
            "filled_at": f.filled_at,
            "symbol": f.order.symbol,
            "side": f.order.side,
            "shares": f.order.shares,
            "order_type": f.order.order_type,
            "limit_price": float(f.order.limit_price) if f.order.limit_price is not None else None,
            "stop_price": float(f.order.stop_price) if f.order.stop_price is not None else None,
            "filled_price": float(f.filled_price),
            "fee": float(f.fee),
        })
    return pd.DataFrame(rows, columns=_TRADE_COLUMNS)


def _flatten_snapshots(snapshots: list[BarSnapshot]) -> pd.DataFrame:
    """Flatten BarSnapshot list to a wide DataFrame with date index."""
    if not snapshots:
        return pd.DataFrame(columns=["date", "cash", "total_value"]).set_index("date")
    rows = []
    for snap in snapshots:
        row: dict[str, Any] = {
            "date": snap.date,
            "cash": snap.cash,
            "total_value": snap.total_value,
        }
        for sym in sorted(snap.target_weights):
            row[f"tw_{sym}"] = snap.target_weights.get(sym, 0.0)
            row[f"aw_{sym}"] = snap.adjusted_weights.get(sym, 0.0)
        for sym in sorted(snap.positions):
            ps = snap.positions[sym]
            row[f"pos_{sym}_shares"] = ps.shares
            row[f"pos_{sym}_avg_cost"] = ps.avg_cost
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _flatten_equity(equity_curve: list[tuple[Any, float]]) -> pd.DataFrame:
    """Flatten equity curve to a DataFrame with date index."""
    if not equity_curve:
        return pd.DataFrame(columns=["value"])
    dates, values = zip(*equity_curve)
    return pd.DataFrame({"value": values}, index=pd.Index(dates, name="date"))
