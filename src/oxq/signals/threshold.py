"""Threshold signal — fires when a column crosses above or below a value."""

from __future__ import annotations

import operator

import pandas as pd


class Threshold:
    """True when ``column`` satisfies ``relationship`` relative to ``threshold``."""

    name = "Threshold"

    def compute(
        self,
        mktdata: dict[str, pd.DataFrame],
        column: str = "",
        threshold: float = 0.0,
        relationship: str = "gt",
    ) -> dict[str, pd.Series]:
        ops = {
            "gt": operator.gt,
            "lt": operator.lt,
            "gte": operator.ge,
            "lte": operator.le,
        }
        op = ops[relationship]
        return {s: op(df[column], threshold) for s, df in mktdata.items()}
