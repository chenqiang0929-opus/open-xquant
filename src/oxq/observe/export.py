"""Export RunResult to a directory of files for subprocess communication."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _flatten_equity(equity_curve: list[tuple[Any, float]]) -> pd.DataFrame:
    """Flatten equity curve to a DataFrame with date index."""
    if not equity_curve:
        return pd.DataFrame(columns=["value"])
    dates, values = zip(*equity_curve)
    return pd.DataFrame({"value": values}, index=pd.Index(dates, name="date"))
