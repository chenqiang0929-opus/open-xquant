"""Composite signal — combine multiple boolean signal columns with AND/OR."""

from __future__ import annotations

from functools import reduce

import pandas as pd


class Composite:
    """Combine boolean signal columns with ``logic`` ('and' or 'or')."""

    name = "Composite"

    def compute(
        self,
        mktdata: pd.DataFrame,
        signals: list[str] | None = None,
        logic: str = "and",
    ) -> pd.Series:
        if not signals:
            return pd.Series(dtype=bool)
        op = pd.Series.__and__ if logic == "and" else pd.Series.__or__
        return reduce(op, (mktdata[col] for col in signals))
