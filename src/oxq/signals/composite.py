"""Composite signal — combine multiple boolean signal columns with AND/OR."""

from __future__ import annotations

from functools import reduce

import pandas as pd


class Composite:
    """Combine boolean signal columns with ``logic`` ('and' or 'or')."""

    name = "Composite"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        signals: list[str] | None = None,
        logic: str = "and",
    ) -> dict[str, pd.Series]:
        if not signals:
            return {}
        op = pd.Series.__and__ if logic == "and" else pd.Series.__or__
        return {
            s: reduce(op, (df[col] for col in signals))
            for s, df in mktdata.items()
        }
